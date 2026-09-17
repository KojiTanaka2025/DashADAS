from __future__ import annotations

import os
import subprocess
import tempfile
import time
from pathlib import Path

import numpy as np
from PIL import Image

from backend.qnn_sdk import (
    apply_qnn_environment,
    find_converted_model,
    find_qnn_sdk,
    qnn_cpu_library,
    qnn_host_bin,
    qnn_status,
)
from backend.yolo_decode import decode_yolo_persons, letterbox

CONF_THRESHOLD = float(os.environ.get("DASHADAS_CONF", "0.25"))


class QnnCpuYolo:
    def __init__(self) -> None:
        status = qnn_status()
        if not status["usable"]:
            raise RuntimeError(status["reason"] or "QNN CPU emulation is not usable on this host")
        sdk = find_qnn_sdk()
        if sdk is None:
            raise RuntimeError("QNN SDK not found")
        model = find_converted_model(sdk)
        if model is None:
            raise RuntimeError(
                "Converted QNN model not found. On Linux amd64 run: ./scripts/convert-qnn-yolo.sh"
            )
        self.sdk = sdk
        self.model_path = model
        self._native = None
        apply_qnn_environment(sdk)
        self._backend_name = "qnn"
        self._try_native()

    def _try_native(self) -> None:
        if os.environ.get("DASHADAS_QNN_RUNTIME", "auto") == "cli":
            return
        try:
            from qti.aisw.tools.core.modules.api.definitions.common import BackendType, ModelConfig, Target
            from qti.aisw.tools.core.modules.net_runner.net_runner_module import (
                InferenceConfig,
                InferenceIdentifier,
                NetRunner,
                NetRunnerLoadArgConfig,
                NetRunnerRunArgConfig,
            )
            from qti.aisw.tools.core.utilities.devices.api.device_definitions import DevicePlatformType

            runner = NetRunner()
            identifier = InferenceIdentifier(
                model=ModelConfig(path=self.model_path),
                target=Target(type=DevicePlatformType.X86_64_LINUX),
                backend=BackendType.CPU,
            )
            loaded = runner.load(NetRunnerLoadArgConfig(identifier=identifier))
            self._native = (runner, loaded.handle, InferenceConfig, NetRunnerRunArgConfig)
        except Exception:
            self._native = None

    def detect(self, image: Image.Image) -> dict:
        tensor, scale, pad_x, pad_y = letterbox(image)
        started = time.perf_counter()
        raw = self._infer(_nchw_to_nhwc(tensor))
        inference_ms = round((time.perf_counter() - started) * 1000, 1)
        detections = decode_yolo_persons(
            raw,
            image.size,
            scale,
            pad_x,
            pad_y,
            conf=CONF_THRESHOLD,
        )
        return {
            "model": self.model_path.name,
            "device": self._backend_name,
            "inference_ms": inference_ms,
            "image": {"width": image.size[0], "height": image.size[1]},
            "detections": detections,
        }

    def _infer(self, tensor: np.ndarray) -> np.ndarray:
        payload = np.ascontiguousarray(tensor.astype(np.float32))
        if self._native is not None:
            return self._infer_native(payload)
        return self._infer_cli(payload)

    def _infer_native(self, tensor: np.ndarray) -> np.ndarray:
        runner, handle, inference_config_cls, run_arg_cls = self._native
        result = runner.run(
            run_arg_cls(
                identifier=handle,
                input_data=tensor,
                inference_config=inference_config_cls(use_native_input_data=True, use_native_output_data=True),
            )
        )
        outputs = result.output_data
        if isinstance(outputs, list) and outputs:
            first = outputs[0]
            if isinstance(first, dict) and first:
                return np.asarray(next(iter(first.values())))
        if isinstance(outputs, dict) and outputs:
            first = next(iter(outputs.values()))
            if isinstance(first, list) and first and isinstance(first[0], dict):
                return np.asarray(next(iter(first[0].values())))
        raise RuntimeError("QNN native runner returned no tensors")

    def _infer_cli(self, tensor: np.ndarray) -> np.ndarray:
        net_run = qnn_host_bin(self.sdk) / "qnn-net-run"
        backend = qnn_cpu_library(self.sdk)
        if not net_run.is_file():
            raise RuntimeError(f"Missing {net_run}")
        env = os.environ.copy()
        with tempfile.TemporaryDirectory(prefix="dashadas-qnn-") as tmp:
            tmp_path = Path(tmp)
            raw_path = tmp_path / "input.raw"
            tensor.tofile(raw_path)
            list_path = tmp_path / "input_list.txt"
            list_path.write_text(f"{raw_path}\n", encoding="utf-8")
            out_dir = tmp_path / "output"
            out_dir.mkdir()
            completed = subprocess.run(
                [
                    str(net_run),
                    "--backend",
                    str(backend),
                    "--model",
                    str(self.model_path),
                    "--input_list",
                    str(list_path),
                    "--output_dir",
                    str(out_dir),
                ],
                capture_output=True,
                text=True,
                check=False,
                env=env,
                cwd=tmp,
            )
            if completed.returncode != 0:
                detail = (completed.stderr or completed.stdout or "qnn-net-run failed").strip()
                raise RuntimeError(detail.splitlines()[-1][:400])
            raw_files = sorted(out_dir.rglob("*.raw"))
            if not raw_files:
                raise RuntimeError("qnn-net-run produced no .raw outputs")
            payload = np.fromfile(raw_files[0], dtype=np.float32)
        return _reshape_yolo_output(payload)


def _nchw_to_nhwc(tensor: np.ndarray) -> np.ndarray:
    # qnn-onnx-converter rewrites the graph to spatial-first (NHWC) even when
    # the ONNX input is NCHW. The converted yolo11n expects {1, 640, 640, 3}.
    if tensor.ndim == 4 and tensor.shape[1] in {1, 3} and tensor.shape[-1] not in {1, 3}:
        return np.transpose(tensor, (0, 2, 3, 1))
    return tensor


def _reshape_yolo_output(payload: np.ndarray) -> np.ndarray:
    count = int(payload.size)
    for channels in (84, 85):
        if count % channels == 0:
            anchors = count // channels
            return payload.reshape(1, channels, anchors)
    return payload.reshape(1, -1)
