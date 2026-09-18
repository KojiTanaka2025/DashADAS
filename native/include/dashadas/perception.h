#ifndef DASHADAS_PERCEPTION_H
#define DASHADAS_PERCEPTION_H

/*
 * DashADAS native perception API (C ABI).
 *
 * Other C or C++ programs load libdashadas_perception.so and call these
 * functions to run YOLO11n person detection. The C++ wrapper is
 * include/dashadas/perception.hpp. The full specification is native/API.md.
 *
 * Typical sequence:
 *   DashadasConfig cfg; dashadas_config_init(&cfg);
 *   cfg.qnn_sdk_root = ...; cfg.qnn_model = ...;
 *   DashadasPerception *ctx = dashadas_create(DASHADAS_DEVICE_QNN, &cfg);
 *   dashadas_detect(ctx, rgb, w, h, out, max_out, &n, &ms);
 *   dashadas_destroy(ctx);
 */

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#ifndef DASHADAS_API
#if defined(_WIN32)
#ifdef DASHADAS_BUILD
#define DASHADAS_API __declspec(dllexport)
#else
#define DASHADAS_API __declspec(dllimport)
#endif
#else
#define DASHADAS_API __attribute__((visibility("default")))
#endif
#endif

/* Bump when the C ABI breaks. Callers can compare dashadas_version(). */
#define DASHADAS_PERCEPTION_VERSION 1

/* Runtime selector. Only DASHADAS_DEVICE_QNN is implemented in this library. */
typedef enum DashadasDevice {
  DASHADAS_DEVICE_CPU = 0,
  DASHADAS_DEVICE_CUDA = 1,
  DASHADAS_DEVICE_QNN = 2
} DashadasDevice;

/* Return codes for dashadas_detect. dashadas_create returns NULL on failure. */
enum DashadasStatus {
  DASHADAS_OK = 0,
  DASHADAS_ERR_INVALID = -1,
  DASHADAS_ERR_NO_MEMORY = -2,
  DASHADAS_ERR_UNSUPPORTED = -3,
  DASHADAS_ERR_LOAD = -4,
  DASHADAS_ERR_INFER = -5
};

/* One person box in original image pixels (inclusive-exclusive xyxy). */
typedef struct DashadasDetection {
  float score; /* person class score in [0, 1] */
  float x1;
  float y1;
  float x2;
  float y2;
} DashadasDetection;

/*
 * Optional paths and thresholds. Pointers must stay valid through
 * dashadas_create; the library copies the values it needs.
 * Zero / NULL fields are filled by dashadas_config_init defaults.
 */
typedef struct DashadasConfig {
  const char *qnn_sdk_root;     /* QAIRT SDK root (required for QNN) */
  const char *qnn_model;        /* converted libyolo11n.so (required for QNN) */
  const char *onnx_model;       /* unused until a native ONNX backend exists */
  const char *qnn_backend_lib;  /* default: <sdk>/lib/x86_64-linux-clang/libQnnCpu.so */
  float conf;                   /* person score floor; default 0.25 */
  float iou;                    /* NMS IoU; default 0.45 */
  int input_size;               /* letterbox size; default 640 */
} DashadasConfig;

typedef struct DashadasPerception DashadasPerception;

DASHADAS_API void dashadas_config_init(DashadasConfig *config);

/* Returns NULL on failure. Read dashadas_last_create_error() for the reason. */
DASHADAS_API DashadasPerception *dashadas_create(DashadasDevice device, const DashadasConfig *config);
DASHADAS_API void dashadas_destroy(DashadasPerception *ctx);
DASHADAS_API const char *dashadas_last_create_error(void);
DASHADAS_API int dashadas_version(void);

/*
 * Detect people in one packed RGB8 frame (row-major, width * height * 3).
 * Writes up to max_out boxes into out. out_count is always set (0 on error).
 * inference_ms, if non-NULL, is model execute time only (not letterbox/NMS).
 */
DASHADAS_API int dashadas_detect(DashadasPerception *ctx,
                                 const uint8_t *rgb,
                                 int width,
                                 int height,
                                 DashadasDetection *out,
                                 int max_out,
                                 int *out_count,
                                 float *inference_ms);

DASHADAS_API const char *dashadas_last_error(const DashadasPerception *ctx);
DASHADAS_API const char *dashadas_device_name(const DashadasPerception *ctx);
DASHADAS_API const char *dashadas_model_name(const DashadasPerception *ctx);

#ifdef __cplusplus
}
#endif

#endif
