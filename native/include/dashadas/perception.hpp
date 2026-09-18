#pragma once

/*
 * Header-only C++ wrapper around the C ABI in perception.h.
 * The process still links libdashadas_perception.so; this type only owns the
 * C handle and converts detections to std::vector.
 */

#include "dashadas/perception.h"

#include <stdexcept>
#include <string>
#include <vector>

namespace dashadas {

inline DashadasConfig default_config() {
  DashadasConfig config;
  dashadas_config_init(&config);
  return config;
}

class Perception {
 public:
  Perception(DashadasDevice device, const DashadasConfig &config) {
    ctx_ = dashadas_create(device, &config);
    if (ctx_ == nullptr) {
      const char *err = dashadas_last_create_error();
      throw std::runtime_error(err && err[0] ? err : "dashadas_create failed");
    }
  }

  Perception(const Perception &) = delete;
  Perception &operator=(const Perception &) = delete;

  Perception(Perception &&other) noexcept : ctx_(other.ctx_) { other.ctx_ = nullptr; }
  Perception &operator=(Perception &&other) noexcept {
    if (this != &other) {
      dashadas_destroy(ctx_);
      ctx_ = other.ctx_;
      other.ctx_ = nullptr;
    }
    return *this;
  }

  ~Perception() { dashadas_destroy(ctx_); }

  /* Caps at 256 boxes, matching a typical dashcam person count. */
  std::vector<DashadasDetection> detect(const uint8_t *rgb,
                                        int width,
                                        int height,
                                        float *inference_ms = nullptr) {
    DashadasDetection tmp[256];
    int count = 0;
    const int rc = dashadas_detect(ctx_, rgb, width, height, tmp, 256, &count, inference_ms);
    if (rc != DASHADAS_OK) {
      throw std::runtime_error(dashadas_last_error(ctx_));
    }
    return std::vector<DashadasDetection>(tmp, tmp + count);
  }

  const char *device_name() const { return dashadas_device_name(ctx_); }
  const char *model_name() const { return dashadas_model_name(ctx_); }
  const char *last_error() const { return dashadas_last_error(ctx_); }

  /* Escape hatch for C-only callees that need the raw handle. */
  DashadasPerception *c_handle() const { return ctx_; }

 private:
  DashadasPerception *ctx_ = nullptr;
};

}  // namespace dashadas
