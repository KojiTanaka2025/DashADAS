#ifndef DASHADAS_PERCEPTION_H
#define DASHADAS_PERCEPTION_H

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

#define DASHADAS_PERCEPTION_VERSION 1

typedef enum DashadasDevice {
  DASHADAS_DEVICE_CPU = 0,
  DASHADAS_DEVICE_CUDA = 1,
  DASHADAS_DEVICE_QNN = 2
} DashadasDevice;

enum DashadasStatus {
  DASHADAS_OK = 0,
  DASHADAS_ERR_INVALID = -1,
  DASHADAS_ERR_NO_MEMORY = -2,
  DASHADAS_ERR_UNSUPPORTED = -3,
  DASHADAS_ERR_LOAD = -4,
  DASHADAS_ERR_INFER = -5
};

typedef struct DashadasDetection {
  float score;
  float x1;
  float y1;
  float x2;
  float y2;
} DashadasDetection;

typedef struct DashadasConfig {
  const char *qnn_sdk_root;
  const char *qnn_model;
  const char *onnx_model;
  const char *qnn_backend_lib;
  float conf;
  float iou;
  int input_size;
} DashadasConfig;

typedef struct DashadasPerception DashadasPerception;

DASHADAS_API void dashadas_config_init(DashadasConfig *config);

DASHADAS_API DashadasPerception *dashadas_create(DashadasDevice device, const DashadasConfig *config);
DASHADAS_API void dashadas_destroy(DashadasPerception *ctx);
DASHADAS_API const char *dashadas_last_create_error(void);
DASHADAS_API int dashadas_version(void);

/* rgb is packed 8-bit RGB, row-major, width * height * 3 bytes. */
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
