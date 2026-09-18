#pragma once

#include <cstdint>
#include <vector>

namespace dashadas {

struct Letterbox {
  std::vector<float> nchw; /* 1*3*size*size */
  std::vector<float> nhwc; /* 1*size*size*3 */
  float scale = 1.0f;
  float pad_x = 0.0f;
  float pad_y = 0.0f;
};

Letterbox letterbox_rgb(const uint8_t *rgb, int width, int height, int size);

}  // namespace dashadas
