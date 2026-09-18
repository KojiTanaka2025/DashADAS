from __future__ import annotations

import io
import os
import shutil
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from PIL import Image, ImageOps
from ultralytics import YOLO

from backend.lane_calib import calibrate_from_frames
from backend.qnn_sdk import find_converted_model, find_qnn_sdk, qnn_status
from backend.qnn_yolo import QnnCpuYolo
from backend.video import (
    create_job,
    ensure_sample_video,
    extract_frames,
    get_frame_path,
    get_job,
    is_video_upload,
    process_video_job,
    public_job,
)

ROOT = Path(__file__).resolve().parent.parent
FRONTEND_DIR = ROOT / "frontend"
SAMPLES_DIR = ROOT / "samples"
MODEL_NAME = os.environ.get("DASHADAS_MODEL", "yolo11n.pt")
CONF_THRESHOLD = float(os.environ.get("DASHADAS_CONF", "0.25"))
PERSON_CLASS_ID = 0
# COCO classes used for distance labels after calibration (QNN path stays person-only).
DETECT_CLASS_IDS = [0, 2, 3, 5, 7]  # person, car, motorcycle, bus, truck
COCO_LABELS = {
    0: "person",
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
}
MAX_IMAGE_BYTES = 12 * 1024 * 1024
MAX_VIDEO_BYTES = int(os.environ.get("DASHADAS_MAX_VIDEO_BYTES", str(1024 * 1024 * 1024)))
JOBS_DIR = Path(os.environ.get("DASHADAS_JOBS_DIR", "/tmp/dashadas-jobs"))
ALLOWED_IMAGE_TYPES = {
    "image/jpeg",
    "image/jpg",
    "image/png",
    "image/webp",
    "image/bmp",
}

model: YOLO | None = None
qnn_model: QnnCpuYolo | None = None
device = "cpu"
infer_lock = threading.Lock()


class DeviceRequest(BaseModel):
    device: str = Field(..., description="cpu, cuda, or qnn")


def cuda_available() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:
        return False


def gpu_name() -> str | None:
    if not cuda_available():
        return None
    try:
        import torch

        return torch.cuda.get_device_name(0)
    except Exception:
        return None


def available_devices() -> list[str]:
    devices = ["cpu"]
    if cuda_available():
        devices.append("cuda")
    if qnn_status()["usable"]:
        devices.append("qnn")
    return devices


def pick_device() -> str:
    requested = os.environ.get("DASHADAS_DEVICE", "auto").strip().lower()
    usable_qnn = qnn_status()["usable"]
    if requested in {"qnn"}:
        return "qnn" if usable_qnn else "cpu"
    if requested in {"cpu"}:
        return "cpu"
    if requested in {"cuda", "gpu"}:
        if cuda_available():
            return "cuda"
        return "cpu"
    return "cuda" if cuda_available() else "cpu"


def warmup_model() -> None:
    if model is None:
        return
    warmup = Image.new("RGB", (64, 64), color=(0, 0, 0))
    model.predict(
        warmup,
        classes=DETECT_CLASS_IDS,
        conf=CONF_THRESHOLD,
        device=device,
        save=False,
        verbose=False,
    )


def ensure_qnn() -> QnnCpuYolo:
    global qnn_model
    if qnn_model is None:
        qnn_model = QnnCpuYolo()
    return qnn_model


def apply_device(name: str) -> str:
    global device
    requested = (name or "").strip().lower()
    if requested not in {"cpu", "cuda", "qnn"}:
        raise HTTPException(status_code=400, detail="Device must be cpu, cuda, or qnn")
    if requested == "cuda" and not cuda_available():
        raise HTTPException(status_code=409, detail="CUDA is not available on this host")
    if requested == "qnn":
        status = qnn_status()
        if not status["usable"]:
            raise HTTPException(status_code=409, detail=status["reason"] or "QNN is not usable")
        with infer_lock:
            try:
                ensure_qnn()
            except Exception as exc:
                raise HTTPException(status_code=503, detail=str(exc)) from exc
            device = requested
        return device
    with infer_lock:
        device = requested
        if model is not None:
            model.to(device)
        warmup_model()
    return device


def load_image(data: bytes) -> Image.Image:
    try:
        image = Image.open(io.BytesIO(data))
        image = ImageOps.exif_transpose(image)
        return image.convert("RGB")
    except Exception as exc:
        raise HTTPException(status_code=400, detail="Could not read the image") from exc


def detect_persons(image: Image.Image) -> dict:
    # QNN converted graph is person-only; CPU/CUDA use the wider DETECT_CLASS_IDS set.
    if device == "qnn":
        with infer_lock:
            try:
                runner = ensure_qnn()
            except Exception as exc:
                raise HTTPException(status_code=503, detail=str(exc)) from exc
            return runner.detect(image)

    if model is None:
        raise HTTPException(status_code=503, detail="Model is not loaded yet")

    with infer_lock:
        started = time.perf_counter()
        results = model.predict(
            image,
            classes=DETECT_CLASS_IDS,
            conf=CONF_THRESHOLD,
            device=device,
            save=False,
            verbose=False,
        )
        inference_ms = round((time.perf_counter() - started) * 1000, 1)

    result = results[0]
    detections = []
    if result.boxes is not None:
        for box in result.boxes:
            xyxy = box.xyxy[0].tolist()
            class_id = int(box.cls[0]) if box.cls is not None else PERSON_CLASS_ID
            detections.append(
                {
                    "label": COCO_LABELS.get(class_id, "object"),
                    "score": round(float(box.conf[0]), 4),
                    "bbox": [round(v, 1) for v in xyxy],
                }
            )

    width, height = image.size
    return {
        "model": MODEL_NAME,
        "device": device,
        "inference_ms": inference_ms,
        "image": {"width": width, "height": height},
        "detections": detections,
    }


def detect_image_bytes(data: bytes) -> dict:
    return detect_persons(load_image(data))


def clamp_video_options(interval_sec: float, max_frames: int) -> tuple[float, int]:
    if interval_sec < 0.03 or interval_sec > 10:
        raise HTTPException(status_code=400, detail="Sampling interval must be between 0.03 and 10 seconds")
    if max_frames < 1 or max_frames > 1000:
        raise HTTPException(status_code=400, detail="Max frames must be between 1 and 1000")
    return interval_sec, max_frames


async def save_upload(file: UploadFile, dest: Path, max_bytes: int) -> int:
    dest.parent.mkdir(parents=True, exist_ok=True)
    size = 0
    try:
        with dest.open("wb") as handle:
            while True:
                chunk = await file.read(1024 * 1024)
                if not chunk:
                    break
                size += len(chunk)
                if size > max_bytes:
                    raise HTTPException(status_code=413, detail="File exceeds the size limit")
                handle.write(chunk)
    except HTTPException:
        dest.unlink(missing_ok=True)
        raise
    if size == 0:
        dest.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail="File is empty")
    return size


def start_video_job(source: Path, filename: str, interval_sec: float, max_frames: int) -> dict:
    job = create_job(filename, interval_sec, max_frames, JOBS_DIR)
    dest = Path(job["video_path"]).with_suffix(Path(filename).suffix.lower() or ".mp4")
    shutil.copyfile(source, dest)
    job["video_path"] = str(dest)
    threading.Thread(
        target=process_video_job,
        args=(job["id"], detect_image_bytes),
        daemon=True,
    ).start()
    return job


@asynccontextmanager
async def lifespan(_app: FastAPI):
    global model, device
    JOBS_DIR.mkdir(parents=True, exist_ok=True)
    device = pick_device()
    model = YOLO(MODEL_NAME)
    if device == "qnn":
        try:
            ensure_qnn()
        except Exception:
            device = "cpu"
            warmup_model()
    else:
        warmup_model()
    yield


app = FastAPI(title="DashADAS", lifespan=lifespan)


@app.get("/api/health")
def health() -> dict:
    qnn = qnn_status()
    qnn["model"] = str(find_converted_model(find_qnn_sdk()) or "")
    ready = model is not None if device != "qnn" else qnn_model is not None
    return {
        "ok": True,
        "model": MODEL_NAME if device != "qnn" else (qnn["model"] or MODEL_NAME),
        "device": device,
        "devices": available_devices(),
        "cuda_available": cuda_available(),
        "gpu_name": gpu_name(),
        "qnn": qnn,
        "ready": ready,
        "video": True,
        "max_video_mb": MAX_VIDEO_BYTES // (1024 * 1024),
    }


@app.post("/api/device")
def set_device(payload: DeviceRequest) -> dict:
    apply_device(payload.device)
    return health()


@app.post("/api/detect")
async def detect(file: UploadFile = File(...)) -> dict:
    if file.content_type and file.content_type.lower() not in ALLOWED_IMAGE_TYPES:
        raise HTTPException(status_code=415, detail="Please upload a JPEG, PNG, or WebP image")

    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="File is empty")
    if len(data) > MAX_IMAGE_BYTES:
        raise HTTPException(status_code=413, detail="Image must be 12MB or smaller")

    return detect_persons(load_image(data))


@app.post("/api/detect-sample")
def detect_sample() -> dict:
    sample_path = SAMPLES_DIR / "bus.jpg"
    if not sample_path.exists():
        raise HTTPException(status_code=404, detail="Sample image is missing")
    payload = detect_persons(load_image(sample_path.read_bytes()))
    payload["sample"] = "/samples/bus.jpg"
    return payload


@app.post("/api/detect-video")
async def detect_video(
    file: UploadFile = File(...),
    interval_sec: float = Form(1.0),
    max_frames: int = Form(300),
) -> dict:
    interval_sec, max_frames = clamp_video_options(interval_sec, max_frames)
    filename = file.filename or "upload.mp4"
    if not is_video_upload(filename, file.content_type):
        raise HTTPException(
            status_code=415,
            detail="Please upload a video such as MP4, MOV, MKV, or WebM",
        )

    job = create_job(filename, interval_sec, max_frames, JOBS_DIR)
    dest = Path(job["video_path"]).with_suffix(Path(filename).suffix.lower() or ".mp4")
    job["video_path"] = str(dest)
    await save_upload(file, dest, MAX_VIDEO_BYTES)
    threading.Thread(
        target=process_video_job,
        args=(job["id"], detect_image_bytes),
        daemon=True,
    ).start()
    return public_job(job)


@app.post("/api/detect-video-sample")
def detect_video_sample(
    interval_sec: float = Form(1 / 30),
    max_frames: int = Form(300),
) -> dict:
    interval_sec, max_frames = clamp_video_options(interval_sec, max_frames)
    try:
        sample_path = ensure_sample_video(JOBS_DIR)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    job = start_video_job(sample_path, "demo-pedestrians.mp4", interval_sec, max_frames)
    return public_job(job)


@app.post("/api/calibrate-video")
async def calibrate_video(
    file: UploadFile = File(...),
    interval_sec: float = Form(0.2),
    max_frames: int = Form(90),
) -> dict:
    """Lane-only calibration. Separate from Detect — no YOLO inference."""
    interval_sec, max_frames = clamp_video_options(interval_sec, max_frames)
    filename = file.filename or "calibrate.mp4"
    if not is_video_upload(filename, file.content_type):
        raise HTTPException(
            status_code=415,
            detail="Please upload a video such as MP4, MOV, MKV, or WebM",
        )

    work = JOBS_DIR / f"calib-{os.getpid()}-{threading.get_ident()}"
    video_path = work / ("source" + (Path(filename).suffix.lower() or ".mp4"))
    frames_dir = work / "frames"
    try:
        await save_upload(file, video_path, MAX_VIDEO_BYTES)
        paths = extract_frames(video_path, frames_dir, interval_sec, max_frames)
        images = [Image.open(path).convert("RGB") for path in paths]
        return calibrate_from_frames(images)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)[:300]) from exc
    finally:
        shutil.rmtree(work, ignore_errors=True)


@app.get("/api/jobs/{job_id}")
def job_status(job_id: str) -> dict:
    job = get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    return public_job(job)


@app.get("/api/jobs/{job_id}/frames/{index}")
def job_frame(job_id: str, index: int) -> FileResponse:
    path = get_frame_path(job_id, index)
    if path is None or not path.exists():
        raise HTTPException(status_code=404, detail="Frame not found")
    return FileResponse(path, media_type="image/jpeg")


if SAMPLES_DIR.exists():
    app.mount("/samples", StaticFiles(directory=SAMPLES_DIR), name="samples")

if FRONTEND_DIR.exists():
    app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")
