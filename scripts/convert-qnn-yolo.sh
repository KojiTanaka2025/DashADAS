#!/usr/bin/env bash
# Convert YOLO11n to a QNN CPU model library.
# Linux amd64 only. Uses the DashADAS Docker image (Python 3.12).
set -euo pipefail

if [[ "$(uname -s)" != "Linux" || "$(uname -m)" != "x86_64" ]]; then
  echo "error: QNN conversion runs on Linux amd64 only." >&2
  echo "Build the Docker or CUDA environment on that host, copy QNN/<version>/, then run:" >&2
  echo "  ./scripts/convert-qnn-yolo.sh" >&2
  exit 1
fi

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
# shellcheck disable=SC1091
source "$ROOT/scripts/qnn-env.sh"

OUT_DIR="${DASHADAS_QNN_OUT:-$ROOT/QNN/models}"
mkdir -p "$OUT_DIR"

dashadas_image() {
  local name
  for name in dashadas:gpu dashadas:cpu dashadas; do
    if docker image inspect "$name" >/dev/null 2>&1; then
      echo "$name"
      return 0
    fi
  done
  return 1
}

if ! command -v docker >/dev/null 2>&1; then
  echo "error: Docker is required. Start the Docker (CPU) or CUDA environment first." >&2
  exit 1
fi

if ! image="$(dashadas_image)"; then
  echo "error: no DashADAS image found (dashadas:gpu or dashadas:cpu)." >&2
  echo "Build one environment first, then convert:" >&2
  echo "  docker compose -f compose.yaml -f compose.lan.yaml up --build -d" >&2
  echo "  docker compose -f compose.yaml -f compose.gpu.yaml up --build -d" >&2
  exit 1
fi

existing="$(find "$OUT_DIR" -name '*yolo11n*.so' -print -quit 2>/dev/null || true)"
if [[ -n "$existing" && -z "${DASHADAS_QNN_FORCE:-}" ]]; then
  echo "Already converted: $existing"
  echo "Set DASHADAS_QNN_FORCE=1 to rebuild."
  exit 0
fi

[[ -f "${QNN_SDK_ROOT}/bin/x86_64-linux-clang/qnn-onnx-converter" ]] || {
  echo "error: missing qnn-onnx-converter under $QNN_SDK_ROOT" >&2
  exit 1
}
[[ -f "${QNN_SDK_ROOT}/bin/x86_64-linux-clang/qnn-model-lib-generator" ]] || {
  echo "error: missing qnn-model-lib-generator under $QNN_SDK_ROOT" >&2
  exit 1
}

echo "Converting YOLO11n with ${image} (Python 3.12) ..."
docker run --rm -i \
  -v "$QNN_SDK_ROOT":/opt/qnn:ro \
  -v "$OUT_DIR":/work \
  -e QNN_SDK_ROOT=/opt/qnn \
  -e PYTHONPATH=/opt/qnn/lib/python \
  -e LD_LIBRARY_PATH=/opt/qnn/lib/x86_64-linux-clang:/opt/qnn/lib/python/qti/aisw/converters/common/linux-x86_64 \
  -e DEBIAN_FRONTEND=noninteractive \
  -w /work \
  "$image" \
  bash -s <<'EOS'
set -euo pipefail
apt-get update -qq
apt-get install -y -qq clang make libc++1 libc++abi1 libatomic1 >/dev/null
pip install -q "onnx==1.16.2" protobuf
if [[ ! -f /work/yolo11n.onnx ]]; then
  python -c 'from ultralytics import YOLO; YOLO("yolo11n.pt").export(format="onnx", imgsz=640, simplify=True, nms=False, opset=17, dynamic=False)'
fi
if [[ ! -f /work/yolo11n.cpp || ! -f /work/yolo11n.bin ]]; then
  python /opt/qnn/bin/x86_64-linux-clang/qnn-onnx-converter \
    --input_network /work/yolo11n.onnx \
    --input_dim images 1,3,640,640 \
    --input_layout images NCHW \
    --output_path /work/yolo11n.cpp
fi
python /opt/qnn/bin/x86_64-linux-clang/qnn-model-lib-generator \
  -c /work/yolo11n.cpp \
  -b /work/yolo11n.bin \
  -o /work \
  -t x86_64-linux-clang
EOS

echo
find "$OUT_DIR" -name '*yolo11n*.so' -print
echo "Start the QNN environment and choose device: qnn"
echo "  docker compose -f compose.yaml -f compose.lan.yaml -f compose.qnn.yaml up --build -d"
