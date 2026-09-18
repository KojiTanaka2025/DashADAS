#include "yolo_decode.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <vector>

namespace dashadas {
namespace {

constexpr int kPersonClass = 0;

std::vector<int> nms(const std::vector<float> &x1,
                     const std::vector<float> &y1,
                     const std::vector<float> &x2,
                     const std::vector<float> &y2,
                     const std::vector<float> &scores,
                     float iou_thres) {
  std::vector<int> order(scores.size());
  std::iota(order.begin(), order.end(), 0);
  std::sort(order.begin(), order.end(), [&](int a, int b) { return scores[a] > scores[b]; });
  std::vector<int> keep;
  std::vector<char> suppressed(order.size(), 0);
  for (size_t i = 0; i < order.size(); ++i) {
    const int idx = order[i];
    if (suppressed[i]) {
      continue;
    }
    keep.push_back(idx);
    const float area_i = (x2[idx] - x1[idx]) * (y2[idx] - y1[idx]);
    for (size_t j = i + 1; j < order.size(); ++j) {
      if (suppressed[j]) {
        continue;
      }
      const int r = order[j];
      const float xx1 = std::max(x1[idx], x1[r]);
      const float yy1 = std::max(y1[idx], y1[r]);
      const float xx2 = std::min(x2[idx], x2[r]);
      const float yy2 = std::min(y2[idx], y2[r]);
      const float inter = std::max(0.0f, xx2 - xx1) * std::max(0.0f, yy2 - yy1);
      const float area_r = (x2[r] - x1[r]) * (y2[r] - y1[r]);
      const float iou = inter / (area_i + area_r - inter + 1e-6f);
      if (iou > iou_thres) {
        suppressed[j] = 1;
      }
    }
  }
  return keep;
}

}  // namespace

std::vector<DashadasDetection> decode_yolo_persons(const float *raw,
                                                   int channels,
                                                   int anchors,
                                                   int image_w,
                                                   int image_h,
                                                   float scale,
                                                   float pad_x,
                                                   float pad_y,
                                                   float conf,
                                                   float iou) {
  /* Layout is (1, C, A) with C = 4 + classes. */
  std::vector<float> x1s;
  std::vector<float> y1s;
  std::vector<float> x2s;
  std::vector<float> y2s;
  std::vector<float> scores;
  x1s.reserve(256);
  for (int a = 0; a < anchors; ++a) {
    const float cx = raw[0 * anchors + a];
    const float cy = raw[1 * anchors + a];
    const float w = raw[2 * anchors + a];
    const float h = raw[3 * anchors + a];
    const float person = (channels > 4) ? raw[(4 + kPersonClass) * anchors + a] : 0.0f;
    if (person < conf) {
      continue;
    }
    x1s.push_back(cx - w * 0.5f);
    y1s.push_back(cy - h * 0.5f);
    x2s.push_back(cx + w * 0.5f);
    y2s.push_back(cy + h * 0.5f);
    scores.push_back(person);
  }
  const auto keep = nms(x1s, y1s, x2s, y2s, scores, iou);
  std::vector<DashadasDetection> dets;
  dets.reserve(keep.size());
  for (int idx : keep) {
    float x1 = (x1s[idx] - pad_x) / scale;
    float y1 = (y1s[idx] - pad_y) / scale;
    float x2 = (x2s[idx] - pad_x) / scale;
    float y2 = (y2s[idx] - pad_y) / scale;
    x1 = std::clamp(x1, 0.0f, static_cast<float>(image_w));
    y1 = std::clamp(y1, 0.0f, static_cast<float>(image_h));
    x2 = std::clamp(x2, 0.0f, static_cast<float>(image_w));
    y2 = std::clamp(y2, 0.0f, static_cast<float>(image_h));
    DashadasDetection det{};
    det.score = scores[idx];
    det.x1 = x1;
    det.y1 = y1;
    det.x2 = x2;
    det.y2 = y2;
    dets.push_back(det);
  }
  return dets;
}

}  // namespace dashadas
