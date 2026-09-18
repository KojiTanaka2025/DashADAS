#include "letterbox.hpp"

#include <algorithm>
#include <cmath>

namespace dashadas {

Letterbox letterbox_rgb(const uint8_t *rgb, int width, int height, int size) {
  Letterbox out;
  const float scale = std::min(static_cast<float>(size) / static_cast<float>(height),
                               static_cast<float>(size) / static_cast<float>(width));
  const int new_w = static_cast<int>(std::round(static_cast<float>(width) * scale));
  const int new_h = static_cast<int>(std::round(static_cast<float>(height) * scale));
  out.scale = scale;
  out.pad_x = (static_cast<float>(size) - static_cast<float>(new_w)) / 2.0f;
  out.pad_y = (static_cast<float>(size) - static_cast<float>(new_h)) / 2.0f;
  const int left = static_cast<int>(std::round(out.pad_x - 0.1f));
  const int top = static_cast<int>(std::round(out.pad_y - 0.1f));

  std::vector<uint8_t> canvas(static_cast<size_t>(size) * static_cast<size_t>(size) * 3u, 114);
  for (int y = 0; y < new_h; ++y) {
    const float src_y = (static_cast<float>(y) + 0.5f) / scale - 0.5f;
    const int y0 = std::max(0, std::min(height - 1, static_cast<int>(std::floor(src_y))));
    const int y1 = std::max(0, std::min(height - 1, y0 + 1));
    const float fy = src_y - static_cast<float>(y0);
    for (int x = 0; x < new_w; ++x) {
      const float src_x = (static_cast<float>(x) + 0.5f) / scale - 0.5f;
      const int x0 = std::max(0, std::min(width - 1, static_cast<int>(std::floor(src_x))));
      const int x1 = std::max(0, std::min(width - 1, x0 + 1));
      const float fx = src_x - static_cast<float>(x0);
      for (int c = 0; c < 3; ++c) {
        const float v00 = rgb[(y0 * width + x0) * 3 + c];
        const float v10 = rgb[(y0 * width + x1) * 3 + c];
        const float v01 = rgb[(y1 * width + x0) * 3 + c];
        const float v11 = rgb[(y1 * width + x1) * 3 + c];
        const float v0 = v00 + (v10 - v00) * fx;
        const float v1 = v01 + (v11 - v01) * fx;
        const float v = v0 + (v1 - v0) * fy;
        canvas[((top + y) * size + (left + x)) * 3 + c] = static_cast<uint8_t>(std::clamp(v, 0.0f, 255.0f));
      }
    }
  }

  const size_t plane = static_cast<size_t>(size) * static_cast<size_t>(size);
  out.nchw.assign(plane * 3u, 0.0f);
  out.nhwc.assign(plane * 3u, 0.0f);
  for (int y = 0; y < size; ++y) {
    for (int x = 0; x < size; ++x) {
      for (int c = 0; c < 3; ++c) {
        const float v = canvas[(y * size + x) * 3 + c] / 255.0f;
        out.nhwc[(y * size + x) * 3 + c] = v;
        out.nchw[c * plane + y * size + x] = v;
      }
    }
  }
  return out;
}

}  // namespace dashadas
