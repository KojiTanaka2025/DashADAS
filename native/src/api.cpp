#include "dashadas/perception.h"

#include "backend.hpp"
#include "letterbox.hpp"
#include "yolo_decode.hpp"

#include <algorithm>
#include <chrono>
#include <cstring>
#include <memory>
#include <mutex>
#include <string>

namespace {

std::mutex g_create_mu;
std::string g_create_error;

const char *device_label(DashadasDevice device) {
  switch (device) {
    case DASHADAS_DEVICE_CPU:
      return "cpu";
    case DASHADAS_DEVICE_CUDA:
      return "cuda";
    case DASHADAS_DEVICE_QNN:
      return "qnn";
    default:
      return "unknown";
  }
}

void set_create_error(const std::string &msg) {
  std::lock_guard<std::mutex> lock(g_create_mu);
  g_create_error = msg;
}

}  // namespace

/* Opaque C handle: one detector instance, one backend, one mutex. */
struct DashadasPerception {
  DashadasDevice device = DASHADAS_DEVICE_QNN;
  float conf = 0.25f;
  float iou = 0.45f;
  int input_size = 640;
  std::unique_ptr<dashadas::Backend> backend;
  std::string last_error;
  std::string device_name;
  std::mutex mu;
};

extern "C" {

void dashadas_config_init(DashadasConfig *config) {
  if (config == nullptr) {
    return;
  }
  std::memset(config, 0, sizeof(*config));
  config->conf = 0.25f;
  config->iou = 0.45f;
  config->input_size = 640;
}

int dashadas_version(void) { return DASHADAS_PERCEPTION_VERSION; }

const char *dashadas_last_create_error(void) {
  std::lock_guard<std::mutex> lock(g_create_mu);
  return g_create_error.c_str();
}

/* Only QNN is wired; CPU/CUDA stay in the enum so the public ABI can grow later. */
DashadasPerception *dashadas_create(DashadasDevice device, const DashadasConfig *config) {
  DashadasConfig local;
  dashadas_config_init(&local);
  if (config != nullptr) {
    local = *config;
  }
  if (local.conf <= 0.0f) {
    local.conf = 0.25f;
  }
  if (local.iou <= 0.0f) {
    local.iou = 0.45f;
  }
  if (local.input_size <= 0) {
    local.input_size = 640;
  }

  auto ctx = std::make_unique<DashadasPerception>();
  ctx->device = device;
  ctx->conf = local.conf;
  ctx->iou = local.iou;
  ctx->input_size = local.input_size;
  ctx->device_name = device_label(device);

  std::string error;
  if (device == DASHADAS_DEVICE_QNN) {
    ctx->backend = dashadas::create_qnn_backend(local, &error);
  } else {
    error = std::string(device_label(device)) +
            " is not implemented in the native library yet; use DASHADAS_DEVICE_QNN";
  }
  if (!ctx->backend) {
    if (error.empty()) {
      error = "failed to create perception backend";
    }
    set_create_error(error);
    return nullptr;
  }
  set_create_error("");
  return ctx.release();
}

void dashadas_destroy(DashadasPerception *ctx) { delete ctx; }

int dashadas_detect(DashadasPerception *ctx,
                    const uint8_t *rgb,
                    int width,
                    int height,
                    DashadasDetection *out,
                    int max_out,
                    int *out_count,
                    float *inference_ms) {
  if (out_count != nullptr) {
    *out_count = 0;
  }
  if (ctx == nullptr || rgb == nullptr || out == nullptr || out_count == nullptr || width <= 0 ||
      height <= 0 || max_out <= 0) {
    if (ctx != nullptr) {
      ctx->last_error = "invalid detect arguments";
    }
    return DASHADAS_ERR_INVALID;
  }

  std::lock_guard<std::mutex> lock(ctx->mu);
  /* Letterbox first; inference_ms below excludes this pre-process. */
  const dashadas::Letterbox input = dashadas::letterbox_rgb(rgb, width, height, ctx->input_size);
  std::vector<float> raw;
  int channels = 0;
  int anchors = 0;
  const auto started = std::chrono::steady_clock::now();
  const int infer_rc = ctx->backend->uses_nhwc()
                           ? ctx->backend->infer_nhwc(input, &raw, &channels, &anchors)
                           : ctx->backend->infer_nchw(input, &raw, &channels, &anchors);
  const auto elapsed = std::chrono::steady_clock::now() - started;
  if (inference_ms != nullptr) {
    *inference_ms = std::chrono::duration<float, std::milli>(elapsed).count();
  }
  if (infer_rc != DASHADAS_OK) {
    ctx->last_error = "inference failed";
    return infer_rc;
  }
  if (raw.empty() || channels <= 4 || anchors <= 0) {
    ctx->last_error = "empty model output";
    return DASHADAS_ERR_INFER;
  }

  const std::vector<DashadasDetection> dets = dashadas::decode_yolo_persons(
      raw.data(), channels, anchors, width, height, input.scale, input.pad_x, input.pad_y, ctx->conf,
      ctx->iou);
  const int n = std::min(max_out, static_cast<int>(dets.size()));
  if (n > 0) {
    std::memcpy(out, dets.data(), static_cast<size_t>(n) * sizeof(DashadasDetection));
  }
  *out_count = n;
  ctx->last_error.clear();
  return DASHADAS_OK;
}

const char *dashadas_last_error(const DashadasPerception *ctx) {
  if (ctx == nullptr) {
    return "null perception handle";
  }
  return ctx->last_error.c_str();
}

const char *dashadas_device_name(const DashadasPerception *ctx) {
  if (ctx == nullptr) {
    return "";
  }
  return ctx->device_name.c_str();
}

const char *dashadas_model_name(const DashadasPerception *ctx) {
  if (ctx == nullptr || ctx->backend == nullptr) {
    return "";
  }
  const char *name = ctx->backend->model_name();
  return name != nullptr ? name : "";
}

}  // extern "C"
