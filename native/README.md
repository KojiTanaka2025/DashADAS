# Native perception library

Shared library (`libdashadas_perception.so`) with a **C ABI** so other C or C++ programs can run the same YOLO11n person detector as the Python service. A header-only C++ wrapper is in `include/dashadas/perception.hpp`.

**API specification:** [`API.md`](API.md) (types, functions, errors, tensor shapes, threading).

Build this on **Linux amd64** (the QNN guest, for example `dashadas-gpu`). Do not build it on macOS: `libQnnCpu.so` is ELF and the QAIRT SDK is not kept on the Mac.

The Web UI / FastAPI path is unchanged. Native CPU/CUDA backends are not implemented yet; this library talks to **QNN CPU emulation** (`libQnnCpu.so` + converted `libyolo11n.so`).

## Layout

| Path | Role |
| --- | --- |
| `API.md` | API specification |
| `include/dashadas/perception.h` | Public C API |
| `include/dashadas/perception.hpp` | C++ wrapper around the C handle |
| `src/` | Letterbox, YOLO decode/NMS, QNN backend (`dlopen`) |
| `examples/sample_client.cpp` | Minimal C++ caller |
| `examples/detect_cli.cpp` | CLI: load an image, print boxes |

## Build

On the Linux guest, with the QAIRT SDK in `QNN/<version>/` and a converted model in `QNN/models/`:

```bash
./scripts/build-native.sh
```

That installs a small compiler toolchain if needed (`cmake`, `g++` or `clang++`, `libc++` for `libQnnCpu.so`), then writes:

```text
native/build/libdashadas_perception.so
native/build/dashadas_detect_cli
native/build/dashadas_sample_client
```

## Call it from another C++ program

```cpp
#include "dashadas/perception.hpp"

DashadasConfig cfg = dashadas::default_config();
cfg.qnn_sdk_root = std::getenv("QNN_SDK_ROOT");
cfg.qnn_model = "/path/to/libyolo11n.so";

dashadas::Perception det(DASHADAS_DEVICE_QNN, cfg);
float inference_ms = 0.0f;
std::vector<DashadasDetection> boxes = det.detect(rgb, width, height, &inference_ms);
```

`rgb` is packed 8-bit RGB, `width * height * 3` bytes, no padding.

Link:

```bash
c++ -std=c++17 my_app.cpp \
  -I native/include \
  -L native/build -ldashadas_perception \
  -Wl,-rpath,$PWD/native/build
```

The same symbols are available from C via `perception.h` (`dashadas_create`, `dashadas_detect`, `dashadas_destroy`).

At runtime the process still needs the QNN backend libraries on the linker path, for example:

```bash
export LD_LIBRARY_PATH="$QNN_SDK_ROOT/lib/x86_64-linux-clang:${LD_LIBRARY_PATH:-}"
```

`libQnnCpu.so` also needs the system `libc++` / `libc++abi` / LLVM `libunwind` packages (the build script installs them).

## CLI check

```bash
source scripts/qnn-env.sh
./native/build/dashadas_detect_cli \
  --sdk "$QNN_SDK_ROOT" \
  --model QNN/models/x86_64-linux-clang/libyolo11n.so \
  --image samples/bus.jpg
```

Expect on the order of **four person boxes** for Ultralytics `bus.jpg`, matching the Python QNN path (NHWC input, output `(1, 84, 8400)`, person class only).

## Notes

- Not thread-safe across concurrent `dashadas_detect` calls on the same handle (the handle serializes them). Use one handle per thread if you need overlap.
- This is the same **x86 QNN CPU** graph as the lab demo. It is not Hexagon HTP, Snapdragon Ride, QNX, or ISO 26262.
- Do not commit the QAIRT SDK or `QNN/models/*.so`.
