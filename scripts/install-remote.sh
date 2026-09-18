#!/usr/bin/env bash
# Install and start DashADAS on a remote VM/CT that you can already SSH into.
#
#   ./scripts/install-remote.sh gpu-host
#   ./scripts/install-remote.sh user@192.168.1.10
#   ./scripts/install-remote.sh --sync user@192.168.1.10
#
# --sync copies this checkout instead of cloning GitHub (useful before a push).
# --cpu / --gpu override autodetection. Default is auto.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SYNC=0
MODE=auto
TARGET=""

usage() {
  cat <<'EOF'
Usage: install-remote.sh [options] [user@]host

Install DashADAS over SSH. Passwordless SSH (or an ssh-agent key) must
already work for the given host.

Guest OS must be Ubuntu 22.04/24.04 or Debian 12/13 on amd64. The
account needs root or passwordless sudo. GPU mode needs nvidia-smi
already working in the guest.

Options:
  --sync      Copy this local checkout to the host instead of git clone
  --auto      Use NVIDIA if nvidia-smi works, otherwise CPU (default)
  --cpu       Force the CPU image
  --gpu       Force the CUDA image (nvidia-smi must work)
  -h, --help  Show this help
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --sync) SYNC=1 ;;
    --auto|--cpu|--gpu) MODE="${1#--}" ;;
    -h|--help)
      usage
      exit 0
      ;;
    --)
      shift
      break
      ;;
    -*)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 1
      ;;
    *)
      if [[ -n "$TARGET" ]]; then
        echo "Unexpected argument: $1" >&2
        usage >&2
        exit 1
      fi
      TARGET="$1"
      ;;
  esac
  shift
done

if [[ -z "$TARGET" ]]; then
  usage >&2
  exit 1
fi

need_cmd() {
  command -v "$1" >/dev/null 2>&1 || {
    echo "error: need '$1' on this machine." >&2
    exit 1
  }
}

need_cmd ssh
need_cmd scp
if [[ "$SYNC" -eq 1 ]]; then
  need_cmd rsync
fi

echo "Checking SSH to ${TARGET}..."
ssh -o BatchMode=yes -o ConnectTimeout=8 "$TARGET" 'echo connected; hostname; whoami' \
  || {
    echo "error: passwordless SSH failed for ${TARGET}.
Fix key auth first (BatchMode must work). If root says 'Too many authentication
failures', use IdentitiesOnly=yes and one key in ~/.ssh/config." >&2
    exit 1
  }

echo "Checking guest OS / arch..."
ssh -o BatchMode=yes "$TARGET" "MODE=${MODE} bash -s" <<'EOF'
set -euo pipefail
if [[ ! -f /etc/os-release ]]; then
  echo "error: missing /etc/os-release" >&2
  exit 1
fi
. /etc/os-release
arch="$(dpkg --print-architecture 2>/dev/null || uname -m)"
ok=0
case "${ID:-}:${VERSION_ID:-}" in
  ubuntu:22.04|ubuntu:24.04|debian:12|debian:13) ok=1 ;;
esac
if [[ "$arch" != "amd64" && "$arch" != "x86_64" ]]; then
  echo "error: guest must be amd64/x86_64 (this host is ${arch})." >&2
  exit 1
fi
if [[ "$ok" -ne 1 ]]; then
  echo "error: unsupported OS ${ID:-unknown} ${VERSION_ID:-}. Use Ubuntu 22.04/24.04 or Debian 12/13." >&2
  exit 1
fi
if [[ "$(id -u)" -ne 0 ]] && ! sudo -n true >/dev/null 2>&1; then
  echo "error: installer needs root or passwordless sudo on this guest." >&2
  exit 1
fi
gpu_ready=0
if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi >/dev/null 2>&1; then
  gpu_ready=1
fi
if [[ "${MODE}" == "gpu" && "$gpu_ready" -ne 1 ]]; then
  echo "error: --gpu requested but nvidia-smi is not working on this guest." >&2
  exit 1
fi
echo "Guest OK: ${ID} ${VERSION_ID} ${arch} (nvidia-smi=$([[ $gpu_ready -eq 1 ]] && echo yes || echo no))"
EOF

if [[ "$SYNC" -eq 1 ]]; then
  echo "Copying sources to ${TARGET}:~/DashADAS ..."
  ssh "$TARGET" "mkdir -p ~/DashADAS"
  rsync -az --delete \
    --exclude '.git/' \
    --exclude '.cursor/' \
    --exclude '.venv/' \
    --exclude '__pycache__/' \
    --exclude '.DS_Store' \
    --exclude '*.mp4' \
    --exclude '*.pt' \
    --exclude '*.onnx' \
    --exclude '*.engine' \
    --exclude 'models/*.pt' \
    --exclude 'QNN/2.*/' \
    --exclude 'QNN/models/' \
    --filter 'P QNN/2.*/' \
    --filter 'P QNN/models/' \
    "$ROOT/" "$TARGET:DashADAS/"
else
  echo "The host will clone https://github.com/KojiTanaka2025/DashADAS.git"
fi

scp -q "$ROOT/scripts/bootstrap.sh" "$TARGET:/tmp/dashadas-bootstrap.sh"
ssh "$TARGET" "chmod +x /tmp/dashadas-bootstrap.sh && DASHADAS_MODE=${MODE} DASHADAS_DIR=\$HOME/DashADAS /tmp/dashadas-bootstrap.sh"
