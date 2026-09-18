#!/usr/bin/env bash
# Build libdashadas_perception.so on Linux amd64. Not supported on macOS.
set -euo pipefail

if [[ "$(uname -s)" != "Linux" || "$(uname -m)" != "x86_64" ]]; then
  echo "error: native QNN library builds only on Linux amd64 (this is $(uname -s) $(uname -m))." >&2
  echo "Build it on the QNN guest (for example dashadas-gpu), not on macOS." >&2
  exit 1
fi

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

need_apt=0
command -v cmake >/dev/null 2>&1 || need_apt=1
if ! command -v g++ >/dev/null 2>&1 && ! command -v clang++ >/dev/null 2>&1; then
  need_apt=1
fi
if ! ldconfig -p 2>/dev/null | grep -q 'libc++\.so\.1'; then
  need_apt=1
fi
if ! ldconfig -p 2>/dev/null | grep -q 'libunwind\.so\.1'; then
  need_apt=1
fi

if [[ "$need_apt" -eq 1 ]]; then
  if [[ "$(id -u)" -ne 0 ]] && ! command -v sudo >/dev/null 2>&1; then
    echo "error: install cmake, a C++ compiler, libc++1, libc++abi1, and LLVM libunwind (libunwind-19)." >&2
    exit 1
  fi
  SUDO=""
  if [[ "$(id -u)" -ne 0 ]]; then
    SUDO="sudo"
  fi
  export DEBIAN_FRONTEND=noninteractive
  $SUDO apt-get update -qq
  pkgs=(cmake g++ libc++1 libc++abi1)
  if apt-cache show libunwind-19 >/dev/null 2>&1; then
    pkgs+=(libunwind-19)
  elif apt-cache show libunwind-18 >/dev/null 2>&1; then
    pkgs+=(libunwind-18)
  else
    echo "error: no LLVM libunwind package (libunwind.so.1) for libQnnCpu.so" >&2
    exit 1
  fi
  $SUDO apt-get install -y -qq "${pkgs[@]}"
fi

# shellcheck disable=SC1091
source "$ROOT/scripts/qnn-env.sh"

if [[ -z "${DASHADAS_QNN_MODEL:-}" ]]; then
  if [[ -f "$ROOT/QNN/models/x86_64-linux-clang/libyolo11n.so" ]]; then
    export DASHADAS_QNN_MODEL="$ROOT/QNN/models/x86_64-linux-clang/libyolo11n.so"
  fi
fi

BUILD="$ROOT/native/build"
cmake -S "$ROOT/native" -B "$BUILD" \
  -DCMAKE_BUILD_TYPE=Release \
  -DQNN_SDK_ROOT="$QNN_SDK_ROOT"
cmake --build "$BUILD" -j"$(nproc)"

echo "Built $BUILD/libdashadas_perception.so"
if [[ -n "${DASHADAS_QNN_MODEL:-}" ]]; then
  echo "Example:"
  echo "  LD_LIBRARY_PATH=\"$QNN_SDK_ROOT/lib/x86_64-linux-clang:\${LD_LIBRARY_PATH:-}\" \\"
  echo "    $BUILD/dashadas_detect_cli --sdk \"$QNN_SDK_ROOT\" --model \"$DASHADAS_QNN_MODEL\" --image samples/bus.jpg"
fi
