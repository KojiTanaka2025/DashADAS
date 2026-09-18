#!/usr/bin/env bash
# Download YOLOPv2 TorchScript weights into models/yolopv2.pt (~149MB).
# Run on the Proxmox guest (dashadas-gpu), not on the Mac mini.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DEST="${DASHADAS_YOLOPV2_WEIGHTS:-$ROOT/models/yolopv2.pt}"
URL="${DASHADAS_YOLOPV2_URL:-https://github.com/CAIC-AD/YOLOPv2/releases/download/V0.0.1/yolopv2.pt}"

mkdir -p "$(dirname "$DEST")"
if [[ -f "$DEST" ]] && [[ "$(stat -c%s "$DEST" 2>/dev/null || stat -f%z "$DEST")" -gt 1000000 ]]; then
  echo "Already present: $DEST ($(du -h "$DEST" | awk '{print $1}'))"
  exit 0
fi

tmp="${DEST}.partial"
echo "Downloading YOLOPv2 weights to ${DEST} ..."
curl -fL --progress-bar -o "$tmp" "$URL"
mv "$tmp" "$DEST"
echo "Saved $(du -h "$DEST" | awk '{print $1}') -> $DEST"
