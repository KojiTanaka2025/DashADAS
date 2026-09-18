# Native perception API

Specification for `libdashadas_perception.so`. Callers should treat this document plus `include/dashadas/perception.h` as the contract. Implementation details in `src/` may change without a version bump if the C ABI stays the same.

Library version: **1** (`DASHADAS_PERCEPTION_VERSION` / `dashadas_version()`).

## Purpose

Run the same YOLO11n **person-only** detector as the Python FastAPI service, from another C or C++ program, without Python. The current backend is Qualcomm QNN **CPU emulation** on Linux amd64 (`libQnnCpu.so` + converted `libyolo11n.so`).

This is a lab API. It is not Hexagon HTP, Snapdragon Ride, QNX, or ISO 26262.

## Library and headers

| Item | Value |
| --- | --- |
| Shared object | `libdashadas_perception.so` (SONAME `libdashadas_perception.so.0`) |
| C header | `dashadas/perception.h` |
| C++ wrapper | `dashadas/perception.hpp` (header-only; still link the `.so`) |
| Language | C++17 implementation, **C ABI** exports |
| pkg-config | `dashadas_perception` after install |

Link:

```text
-ldashadas_perception
```

At runtime set `LD_LIBRARY_PATH` so the dynamic linker can find `$QNN_SDK_ROOT/lib/x86_64-linux-clang` (for `libQnnCpu.so`). The library also `dlopen`s system `libc++` / `libunwind` because `libQnnCpu.so` is an LLVM C++ binary.

## Lifecycle

```text
dashadas_config_init(&cfg)
  → fill cfg.qnn_sdk_root and cfg.qnn_model (and optional fields)
  → ctx = dashadas_create(DASHADAS_DEVICE_QNN, &cfg)
  → loop: dashadas_detect(ctx, rgb, w, h, out, max_out, &n, &ms)
  → dashadas_destroy(ctx)
```

`dashadas_create` copies config scalars. Path **pointers** must remain valid for the duration of `dashadas_create` (the backend copies the strings internally). After create returns, the caller may free its own config buffers.

`dashadas_destroy(NULL)` is safe.

## Sample calls

`rgb` is packed RGB8, `width * height * 3` bytes, no row padding. Set `QNN_SDK_ROOT` and the model path to the converted `libyolo11n.so`. Runnable copies live in `examples/sample_client.cpp` (C++) and `examples/detect_cli.cpp` (C API + image load).

### C

```c
#include "dashadas/perception.h"

DashadasConfig cfg;
dashadas_config_init(&cfg);
cfg.qnn_sdk_root = "/path/to/QNN/2.50.0.260828";
cfg.qnn_model = "/path/to/libyolo11n.so";

DashadasPerception *ctx = dashadas_create(DASHADAS_DEVICE_QNN, &cfg);
if (ctx == NULL) {
  fprintf(stderr, "%s\n", dashadas_last_create_error());
  return 1;
}

DashadasDetection boxes[64];
int n = 0;
float ms = 0.f;
int rc = dashadas_detect(ctx, rgb, width, height, boxes, 64, &n, &ms);
if (rc != DASHADAS_OK) {
  fprintf(stderr, "%s\n", dashadas_last_error(ctx));
  dashadas_destroy(ctx);
  return 1;
}

printf("%d person(s) in %.1f ms\n", n, ms);
for (int i = 0; i < n; ++i) {
  printf("person %.3f  [%.1f, %.1f, %.1f, %.1f]\n",
         boxes[i].score, boxes[i].x1, boxes[i].y1, boxes[i].x2, boxes[i].y2);
}
dashadas_destroy(ctx);
```

### C++

```cpp
#include "dashadas/perception.hpp"

DashadasConfig cfg = dashadas::default_config();
cfg.qnn_sdk_root = std::getenv("QNN_SDK_ROOT");
cfg.qnn_model = "/path/to/libyolo11n.so";

dashadas::Perception det(DASHADAS_DEVICE_QNN, cfg);
float ms = 0.f;
std::vector<DashadasDetection> boxes = det.detect(rgb, width, height, &ms);
for (const DashadasDetection &box : boxes) {
  std::printf("person %.3f  [%.1f, %.1f, %.1f, %.1f]\n",
              box.score, box.x1, box.y1, box.x2, box.y2);
}
```

The wrapper throws `std::runtime_error` if create or detect fails.

## Types

### `DashadasDevice`

| Value | Name | Native backend |
| --- | --- | --- |
| 0 | `DASHADAS_DEVICE_CPU` | not implemented (`create` fails) |
| 1 | `DASHADAS_DEVICE_CUDA` | not implemented (`create` fails) |
| 2 | `DASHADAS_DEVICE_QNN` | QNN CPU emulation |

### `DashadasStatus` (`dashadas_detect` return)

| Value | Name | Meaning |
| --- | --- | --- |
| 0 | `DASHADAS_OK` | `*out_count` boxes written |
| -1 | `DASHADAS_ERR_INVALID` | null handle, bad size, or bad pointers |
| -2 | `DASHADAS_ERR_NO_MEMORY` | allocation failed during setup |
| -3 | `DASHADAS_ERR_UNSUPPORTED` | reserved / unused backend path |
| -4 | `DASHADAS_ERR_LOAD` | `dlopen` / QNN graph setup failed (`create` also uses this via NULL) |
| -5 | `DASHADAS_ERR_INFER` | `graphExecute` failed or empty output |

`dashadas_create` does **not** return these codes. It returns `NULL` and stores a message in `dashadas_last_create_error()`.

### `DashadasDetection`

| Field | Unit | Notes |
| --- | --- | --- |
| `score` | 0–1 | COCO `person` class score |
| `x1,y1,x2,y2` | pixels | original image coordinates, clamped to `[0, width]` / `[0, height]` |

Class is always person. There is no label field.

### `DashadasConfig`

| Field | Default | Required for QNN |
| --- | --- | --- |
| `qnn_sdk_root` | NULL | yes |
| `qnn_model` | NULL | yes (path to `libyolo11n.so`) |
| `onnx_model` | NULL | no (ignored) |
| `qnn_backend_lib` | NULL → `<sdk>/lib/x86_64-linux-clang/libQnnCpu.so` | no |
| `conf` | 0.25 | no |
| `iou` | 0.45 | no |
| `input_size` | 640 | no; must match the converted graph (640) |

`<= 0` for `conf` / `iou` / `input_size` is replaced by the default at create time.

## Functions

### `void dashadas_config_init(DashadasConfig *config)`

Zeroes the struct and writes defaults. No-op if `config` is NULL.

### `DashadasPerception *dashadas_create(DashadasDevice device, const DashadasConfig *config)`

Loads the backend and finalizes the QNN graph. `config` may be NULL (defaults only; QNN will then fail because paths are missing).

On failure: returns NULL. `dashadas_last_create_error()` is a NUL-terminated English string. The pointer is valid until the next `dashadas_create` in the process.

### `void dashadas_destroy(DashadasPerception *ctx)`

Releases QNN context, tensors, and `dlopen` handles.

### `int dashadas_detect(...)`

```c
int dashadas_detect(DashadasPerception *ctx,
                    const uint8_t *rgb,
                    int width,
                    int height,
                    DashadasDetection *out,
                    int max_out,
                    int *out_count,
                    float *inference_ms);
```

| Argument | Rules |
| --- | --- |
| `rgb` | packed RGB8, row-major, `width * height * 3` bytes, no row padding |
| `width`, `height` | `> 0` |
| `out` | caller-owned array of at least `max_out` elements |
| `max_out` | `> 0` |
| `out_count` | required; set to 0 before returning on error |
| `inference_ms` | optional; milliseconds for **graph execute only** |

Pipeline: letterbox to `input_size` (gray 114 pad, bilinear) → NHWC float32 `[0,1]` feed → QNN `graphExecute` → YOLO decode (`(1, 84, 8400)` layout) → person class + NMS → map boxes back to the original image.

The C++ wrapper `dashadas::Perception::detect` uses `max_out = 256`.

Concurrent `dashadas_detect` on the **same** handle is serialized. Use one handle per thread for overlap. `dashadas_last_error` / `dashadas_device_name` / `dashadas_model_name` return pointers owned by the handle; they are invalid after `destroy`.

### Query helpers

| Function | Returns |
| --- | --- |
| `dashadas_version` | `DASHADAS_PERCEPTION_VERSION` |
| `dashadas_device_name` | `"qnn"` / `"cpu"` / `"cuda"`; `""` if `ctx` is NULL |
| `dashadas_model_name` | path passed as `qnn_model` |
| `dashadas_last_error` | last `detect` error, or `"null perception handle"` |

## C++ wrapper

`dashadas::Perception` is move-only. It owns the C handle and caps `detect` at 256 boxes. See **Sample calls** above for a complete example.

## Errors and logging

QNN backend logs are swallowed (silent callback) so a host application is not flooded. Failures surface only as status codes and `last_error` strings.

## Model I/O (QNN)

| Tensor | Shape | Layout | Dtype |
| --- | --- | --- | --- |
| input `images` | `{1, 640, 640, 3}` | NHWC | float32 |
| output `output0` | `{1, 84, 8400}` | channels × anchors | float32 |

Channel layout is YOLO11: `x, y, w, h` then 80 COCO classes. Only class 0 (`person`) is kept.

## Web UI JSON (Python service)

Unrelated to the `.so`, but the same detector: each still or video frame includes `inference_ms` (model time). The Web UI shows it as **Frame time**.

## Out of scope

- Native CPU / CUDA (ONNX Runtime / TensorRT)
- HTP / quantized context binaries
- Thread-safe graph execute on one context
- Authentication, vehicle control, or safety claims
