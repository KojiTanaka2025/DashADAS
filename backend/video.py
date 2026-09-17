from __future__ import annotations

import json
import os
import shutil
import subprocess
import threading
import urllib.error
import urllib.request
import uuid
from pathlib import Path

JOBS: dict[str, dict] = {}
JOBS_LOCK = threading.Lock()
MAX_STORED_JOBS = 4

VIDEO_EXTENSIONS = {
    ".mp4",
    ".mov",
    ".m4v",
    ".avi",
    ".mkv",
    ".webm",
    ".ts",
    ".mts",
    ".m2ts",
}

VIDEO_CONTENT_TYPES = {
    "video/mp4",
    "video/quicktime",
    "video/x-m4v",
    "video/x-msvideo",
    "video/webm",
    "video/x-matroska",
    "video/mp2t",
    "application/octet-stream",
}


def is_video_upload(filename: str | None, content_type: str | None) -> bool:
    suffix = Path(filename or "").suffix.lower()
    if suffix in VIDEO_EXTENSIONS:
        return True
    if content_type and content_type.lower() in VIDEO_CONTENT_TYPES:
        return True
    return False


def probe_video(path: Path) -> dict:
    command = [
        "ffprobe",
        "-v",
        "error",
        "-print_format",
        "json",
        "-show_format",
        "-show_streams",
        str(path),
    ]
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout or "ffprobe failed").strip()
        raise RuntimeError(detail.splitlines()[-1][:300])

    payload = json.loads(completed.stdout or "{}")
    video_stream = next(
        (stream for stream in payload.get("streams", []) if stream.get("codec_type") == "video"),
        {},
    )
    fmt = payload.get("format", {})
    fps = _parse_fps(video_stream.get("avg_frame_rate") or video_stream.get("r_frame_rate"))
    duration = float(fmt.get("duration") or video_stream.get("duration") or 0)
    return {
        "duration_sec": round(duration, 3),
        "fps": fps,
        "width": int(video_stream.get("width") or 0),
        "height": int(video_stream.get("height") or 0),
        "codec": video_stream.get("codec_name") or "unknown",
    }


def extract_frames(path: Path, out_dir: Path, interval_sec: float, max_frames: int) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    fps = 1.0 / interval_sec if interval_sec > 0 else 1.0
    pattern = out_dir / "frame_%06d.jpg"
    command = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(path),
        "-vf",
        f"fps={fps}",
        "-frames:v",
        str(max_frames),
        "-q:v",
        "3",
        str(pattern),
    ]
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout or "ffmpeg failed").strip()
        raise RuntimeError(detail.splitlines()[-1][:300])

    frames = sorted(out_dir.glob("frame_*.jpg"))
    if not frames:
        raise RuntimeError("Could not extract frames from the video. The codec may be unsupported.")
    return frames


SAMPLE_VIDEO_LOCK = threading.Lock()
DEFAULT_SAMPLE_VIDEO_URLS = (
    # Pexels, license: https://www.pexels.com/license/ — pedestrians in a city street.
    "https://videos.pexels.com/video-files/12144112/12144112-sd_640_360_30fps.mp4",
    # Ultralytics public plaza clip, used if the Pexels URL is unreachable.
    "https://github.com/ultralytics/assets/releases/download/v0.0.0/solutions_ci_demo.mp4",
)
SAMPLE_VIDEO_MAX_BYTES = 80 * 1024 * 1024
SAMPLE_VIDEO_SECONDS = float(os.environ.get("DASHADAS_SAMPLE_VIDEO_SECONDS", "8"))


def sample_video_urls() -> list[str]:
    extra = os.environ.get("DASHADAS_SAMPLE_VIDEO_URL", "").strip()
    urls = [extra] if extra else []
    urls.extend(DEFAULT_SAMPLE_VIDEO_URLS)
    seen: set[str] = set()
    unique: list[str] = []
    for url in urls:
        if url and url not in seen:
            seen.add(url)
            unique.append(url)
    return unique


def ensure_sample_video(cache_dir: Path) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    dest = cache_dir / "demo-pedestrians.mp4"
    with SAMPLE_VIDEO_LOCK:
        if dest.exists() and dest.stat().st_size > 10_000:
            return dest
        errors: list[str] = []
        for url in sample_video_urls():
            raw = dest.with_suffix(".download.mp4")
            try:
                _download_url(url, raw, SAMPLE_VIDEO_MAX_BYTES)
                _trim_sample_video(raw, dest, SAMPLE_VIDEO_SECONDS)
                raw.unlink(missing_ok=True)
                return dest
            except Exception as exc:
                raw.unlink(missing_ok=True)
                dest.unlink(missing_ok=True)
                errors.append(f"{url}: {exc}")
        detail = errors[-1] if errors else "No sample video URL configured"
        raise RuntimeError(f"Could not fetch a demo video. {detail}")


def _download_url(url: str, dest: Path, max_bytes: int) -> None:
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "DashADAS/1.0 (pedestrian demo clip)"},
    )
    dest.parent.mkdir(parents=True, exist_ok=True)
    size = 0
    try:
        with urllib.request.urlopen(request, timeout=60) as response, dest.open("wb") as handle:
            content_type = (response.headers.get("Content-Type") or "").lower()
            if content_type.startswith("text/html"):
                raise RuntimeError("Download returned a web page instead of a video")
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                size += len(chunk)
                if size > max_bytes:
                    raise RuntimeError("Demo video exceeds the size limit")
                handle.write(chunk)
    except urllib.error.URLError as exc:
        dest.unlink(missing_ok=True)
        raise RuntimeError(str(exc.reason if hasattr(exc, "reason") else exc)) from exc
    if size < 10_000:
        dest.unlink(missing_ok=True)
        raise RuntimeError("Downloaded file is too small to be a video")


def _trim_sample_video(source: Path, dest: Path, max_sec: float) -> None:
    command = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(source),
        "-t",
        f"{max_sec:.3f}",
        "-r",
        "30",
        "-an",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(dest),
    ]
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    if completed.returncode != 0 or not dest.exists() or dest.stat().st_size < 10_000:
        shutil.copyfile(source, dest)
        if not dest.exists() or dest.stat().st_size < 10_000:
            detail = (completed.stderr or completed.stdout or "ffmpeg failed").strip()
            raise RuntimeError(detail.splitlines()[-1][:300] if detail else "Could not prepare demo video")


def create_job(filename: str, interval_sec: float, max_frames: int, jobs_dir: Path) -> dict:
    job_id = uuid.uuid4().hex[:12]
    job_dir = jobs_dir / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    job = {
        "id": job_id,
        "status": "queued",
        "error": None,
        "filename": filename,
        "interval_sec": interval_sec,
        "max_frames": max_frames,
        "dir": str(job_dir),
        "video_path": str(job_dir / "source"),
        "video": None,
        "processed": 0,
        "total": 0,
        "person_frames": 0,
        "person_detections": 0,
        "inference_ms_total": 0.0,
        "frames": [],
    }
    with JOBS_LOCK:
        _prune_jobs_locked(jobs_dir)
        JOBS[job_id] = job
    return job


def public_job(job: dict) -> dict:
    return {
        "id": job["id"],
        "status": job["status"],
        "error": job["error"],
        "filename": job["filename"],
        "interval_sec": job["interval_sec"],
        "max_frames": job["max_frames"],
        "video": job["video"],
        "processed": job["processed"],
        "total": job["total"],
        "person_frames": job["person_frames"],
        "person_detections": job["person_detections"],
        "inference_ms_total": round(job["inference_ms_total"], 1),
        "frames": [
            {
                "index": frame["index"],
                "time_sec": frame["time_sec"],
                "url": f"/api/jobs/{job['id']}/frames/{frame['index']}",
                "detections": frame["detections"],
                "inference_ms": frame["inference_ms"],
                "image": frame["image"],
            }
            for frame in job["frames"]
        ],
    }


def get_job(job_id: str) -> dict | None:
    with JOBS_LOCK:
        job = JOBS.get(job_id)
        return dict(job) if job else None


def get_frame_path(job_id: str, index: int) -> Path | None:
    with JOBS_LOCK:
        job = JOBS.get(job_id)
        if not job or index < 0 or index >= len(job["frames"]):
            return None
        return Path(job["frames"][index]["path"])


def process_video_job(job_id: str, detect_fn) -> None:
    with JOBS_LOCK:
        job = JOBS.get(job_id)
        if job is None:
            return
        job["status"] = "running"
        video_path = Path(job["video_path"])
        job_dir = Path(job["dir"])
        interval_sec = job["interval_sec"]
        max_frames = job["max_frames"]

    try:
        info = probe_video(video_path)
        frames_dir = job_dir / "frames"
        frame_paths = extract_frames(video_path, frames_dir, interval_sec, max_frames)
        with JOBS_LOCK:
            job["video"] = info
            job["total"] = len(frame_paths)

        for index, frame_path in enumerate(frame_paths):
            result = detect_fn(frame_path.read_bytes())
            frame = {
                "index": index,
                "time_sec": round(index * interval_sec, 3),
                "path": str(frame_path),
                "detections": result["detections"],
                "inference_ms": result["inference_ms"],
                "image": result["image"],
            }
            with JOBS_LOCK:
                job["frames"].append(frame)
                job["processed"] = index + 1
                job["inference_ms_total"] += result["inference_ms"]
                if result["detections"]:
                    job["person_frames"] += 1
                    job["person_detections"] += len(result["detections"])

        with JOBS_LOCK:
            job["status"] = "done"
    except Exception as exc:
        with JOBS_LOCK:
            job["status"] = "error"
            job["error"] = str(exc)


def _parse_fps(value: str | None) -> float:
    if not value or value == "0/0":
        return 0.0
    if "/" in value:
        numerator, denominator = value.split("/", 1)
        try:
            denom = float(denominator)
            return round(float(numerator) / denom, 3) if denom else 0.0
        except ValueError:
            return 0.0
    try:
        return round(float(value), 3)
    except ValueError:
        return 0.0


def _prune_jobs_locked(jobs_dir: Path) -> None:
    finished = [job for job in JOBS.values() if job["status"] in {"done", "error"}]
    extra = len(finished) - (MAX_STORED_JOBS - 1)
    if extra <= 0:
        return
    for job in finished[:extra]:
        JOBS.pop(job["id"], None)
        shutil.rmtree(job["dir"], ignore_errors=True)
    jobs_dir.mkdir(parents=True, exist_ok=True)
