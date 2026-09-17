#!/usr/bin/env bash
# Install Docker and start DashADAS on this machine (Debian/Ubuntu amd64).
# Intended to run as root or with sudo after SSH access is already working.
set -euo pipefail

REPO_URL="${DASHADAS_REPO:-https://github.com/KojiTanaka2025/DashADAS.git}"
INSTALL_DIR="${DASHADAS_DIR:-$HOME/DashADAS}"
MODE="${DASHADAS_MODE:-auto}" # auto | cpu | gpu
CPU_DISK_GB="${DASHADAS_CPU_DISK_GB:-15}"
GPU_DISK_GB="${DASHADAS_GPU_DISK_GB:-25}"

need_root() {
  if [[ "$(id -u)" -eq 0 ]]; then
    "$@"
  else
    sudo "$@"
  fi
}

fail() {
  echo "error: $*" >&2
  exit 1
}

os_supported() {
  local id="$1" version="$2"
  case "$id" in
    ubuntu)
      [[ "$version" == "22.04" || "$version" == "24.04" ]]
      ;;
    debian)
      [[ "$version" == "12" || "$version" == "13" ]]
      ;;
    *)
      return 1
      ;;
  esac
}

free_disk_gb() {
  df -P -B1G / | awk 'NR==2 {print $4}'
}

if [[ ! -f /etc/os-release ]]; then
  fail "This installer supports Ubuntu 22.04/24.04 and Debian 12/13 only."
fi
# shellcheck disable=SC1091
. /etc/os-release

arch="$(dpkg --print-architecture 2>/dev/null || uname -m)"
[[ "$arch" == "amd64" || "$arch" == "x86_64" ]] || fail "Need amd64/x86_64. This host is ${arch}."

os_supported "${ID:-}" "${VERSION_ID:-}" \
  || fail "Unsupported OS ${ID:-unknown} ${VERSION_ID:-}. Use Ubuntu 22.04/24.04 or Debian 12/13 (amd64)."

gpu_ready=0
if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi >/dev/null 2>&1; then
  gpu_ready=1
fi
if [[ "$MODE" == "gpu" || ( "$MODE" == "auto" && "$gpu_ready" -eq 1 ) ]]; then
  IMAGE_KIND=gpu
else
  IMAGE_KIND=cpu
fi
if [[ "$MODE" == "gpu" && "$gpu_ready" -ne 1 ]]; then
  fail "GPU mode requested, but nvidia-smi is not working.
Install NVIDIA userspace drivers in this guest first (same version as the host).
Inside Proxmox LXC: privileged container, Nesting=on, /dev/nvidia0 present,
then NVIDIA .run installer with --no-kernel-module. Do not PCI-passthrough
a consumer GPU into a VM if other LXCs already share the host GPU."
fi
if [[ "$IMAGE_KIND" == "gpu" && ! -e /dev/nvidia0 ]]; then
  fail "/dev/nvidia0 is missing. Pass NVIDIA devices into this guest before GPU install."
fi

need_gb="$CPU_DISK_GB"
[[ "$IMAGE_KIND" == "gpu" ]] && need_gb="$GPU_DISK_GB"
have_gb="$(free_disk_gb)"
if [[ "${have_gb:-0}" -lt "$need_gb" ]]; then
  fail "Not enough free disk on /. Have ${have_gb}G, need ${need_gb}G for ${IMAGE_KIND} image build."
fi
if [[ ! -d /run/systemd/system ]]; then
  fail "Need systemd (this installer starts Docker with systemctl)."
fi
if [[ "$(id -u)" -ne 0 ]] && ! sudo -n true >/dev/null 2>&1; then
  fail "Need root or passwordless sudo. This script is meant to run over non-interactive SSH."
fi

echo "Preflight OK: ${ID} ${VERSION_ID} amd64, ${IMAGE_KIND} install, ${have_gb}G free on /."

export DEBIAN_FRONTEND=noninteractive
need_root apt-get update -y
need_root apt-get install -y ca-certificates curl git gnupg rsync iproute2

if ! curl -fsSI --max-time 8 https://download.docker.com/linux >/dev/null 2>&1; then
  echo "warning: cannot reach https://download.docker.com — Docker install/build may fail." >&2
fi

if command -v docker >/dev/null 2>&1; then
  docker_bin="$(command -v docker)"
  if [[ "$docker_bin" == /snap/* ]]; then
    fail "Snap Docker is not supported. Remove the snap, then re-run so Docker CE can be installed from download.docker.com."
  fi
  if ! docker compose version >/dev/null 2>&1 && ! need_root docker compose version >/dev/null 2>&1; then
    fail "Docker is present but Compose v2 is missing. Install Docker CE from download.docker.com (not docker.io / snap)."
  fi
else
  need_root install -m 0755 -d /etc/apt/keyrings
  curl -fsSL "https://download.docker.com/linux/${ID}/gpg" | need_root tee /etc/apt/keyrings/docker.asc >/dev/null
  need_root chmod a+r /etc/apt/keyrings/docker.asc
  echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/${ID} ${VERSION_CODENAME} stable" \
    | need_root tee /etc/apt/sources.list.d/docker.list >/dev/null
  need_root apt-get update -y
  need_root apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
fi

if command -v ss >/dev/null 2>&1 && ss -lnt | awk '$4 ~ /:8080$/ {found=1} END {exit !found}'; then
  fail "TCP port 8080 is already in use on this host."
fi

if [[ "$(id -u)" -ne 0 ]]; then
  need_root usermod -aG docker "$USER" || true
fi

if [[ "$IMAGE_KIND" == "gpu" ]]; then
  if ! dpkg -s nvidia-container-toolkit >/dev/null 2>&1; then
    curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
      | need_root gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
    curl -fsSL https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
      | sed "s#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g" \
      | need_root tee /etc/apt/sources.list.d/nvidia-container-toolkit.list >/dev/null
    need_root apt-get update -y
    need_root apt-get install -y nvidia-container-toolkit
    need_root nvidia-ctk runtime configure --runtime=docker
    need_root systemctl restart docker
  fi
  COMPOSE_FILES=( -f compose.yaml -f compose.gpu.yaml )
else
  COMPOSE_FILES=( -f compose.yaml -f compose.lan.yaml )
fi

if [[ -d "$INSTALL_DIR/.git" ]]; then
  git -C "$INSTALL_DIR" fetch --quiet origin
  git -C "$INSTALL_DIR" pull --ff-only
elif [[ -f "$INSTALL_DIR/compose.yaml" ]]; then
  echo "Using existing sources in $INSTALL_DIR"
else
  git clone "$REPO_URL" "$INSTALL_DIR"
fi

cd "$INSTALL_DIR"
# Optional QNN overlay: only when a Linux QNN SDK is already on this guest.
# Conversion (./scripts/convert-qnn-yolo.sh) is a separate QNN-environment step.
if [[ -f compose.qnn.yaml && -d QNN ]] && find QNN -name 'libQnnCpu.so' -print -quit | grep -q .; then
  COMPOSE_FILES+=( -f compose.qnn.yaml )
fi
need_root docker compose "${COMPOSE_FILES[@]}" up --build -d

echo "Waiting for http://127.0.0.1:8080/api/health ..."
ok=0
for _ in $(seq 1 60); do
  if curl -fsS http://127.0.0.1:8080/api/health >/dev/null 2>&1; then
    ok=1
    break
  fi
  sleep 2
done
[[ "$ok" -eq 1 ]] || fail "Container started but /api/health did not become ready. Check: docker compose logs"

lan_ip="$(hostname -I 2>/dev/null | awk '{print $1}')"
echo
echo "DashADAS (${IMAGE_KIND}) is running."
echo "Health: $(curl -fsS http://127.0.0.1:8080/api/health)"
if [[ -n "$lan_ip" ]]; then
  echo "Open the demo UI from another machine on this LAN:"
  echo "  http://${lan_ip}:8080"
fi
echo
echo "The API has no login. Keep it on a private network."
