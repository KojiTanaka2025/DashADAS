# QNN environment (not in Git)

This folder is the drop-in for a **Linux x86_64** Qualcomm AI Runtime (QAIRT / QNN) SDK. Do not commit the SDK. GitHub must never receive these files.

QNN in DashADAS is a **separate run environment** from Docker (CPU) and CUDA. It only works on Linux amd64. This checkout expects **QAIRT 2.50.0.260828** (other 2.x versions may work).

Qualcomm verifies Ubuntu 22.04 (Python 3.10) and Ubuntu 24.04 (Python 3.12). Conversion here always uses the DashADAS Docker image (Python 3.12), so a Debian 12/13 amd64 Docker host is enough.

## What to copy

After you unzip the Qualcomm package, copy the **version directory** that contains `QNN_README.txt` (or `bin/envsetup.sh`):

```text
QNN/
  README.md                 (this file; tracked by git)
  2.50.0.260828/            (SDK; gitignored)
    QNN_README.txt
    bin/
    lib/
    include/
  models/                   (converted libyolo11n.so; gitignored)
```

If the zip extracts as `qairt/2.50.0.260828/`, copy that inner version folder here.

**Important:** never `rsync --delete` from a Mac checkout into a guest that already has
`QNN/<version>/` unless you protect those paths. `scripts/install-remote.sh --sync`
excludes and protects `QNN/2.*/` and `QNN/models/` so the proprietary SDK is not wiped.

## Convert and run (Linux amd64 only)

On the Linux host, build Docker (CPU) or CUDA first so `dashadas:cpu` or `dashadas:gpu` exists. Then:

```bash
./scripts/convert-qnn-yolo.sh
docker compose -f compose.yaml -f compose.lan.yaml -f compose.qnn.yaml up --build -d
```

The converter writes `QNN/models/` (ignored by git). Open the Web UI and select **qnn (CPU emu)**.

QNN does not need a GPU. On a GPU guest, `compose.gpu.yaml` may replace `compose.lan.yaml` if you also want `cuda` in the Device menu.

Do not run the converter on macOS.

## Check that git will not upload the SDK

```bash
git check-ignore -v QNN/2.50.0.260828/lib/x86_64-linux-clang/libQnnCpu.so
```

The SDK path must be ignored. Only `QNN/README.md` may be tracked.
