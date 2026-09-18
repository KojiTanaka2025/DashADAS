/* Example of another C++ program calling libdashadas_perception. */

#include "dashadas/perception.hpp"

#include <cstdio>
#include <cstdlib>
#include <vector>

int main() {
  DashadasConfig config = dashadas::default_config();
  const char *sdk = std::getenv("QNN_SDK_ROOT");
  const char *model = std::getenv("DASHADAS_QNN_MODEL");
  if (sdk == nullptr || model == nullptr) {
    std::fprintf(stderr, "Set QNN_SDK_ROOT and DASHADAS_QNN_MODEL\n");
    return 2;
  }
  config.qnn_sdk_root = sdk;
  config.qnn_model = model;

  try {
    dashadas::Perception det(DASHADAS_DEVICE_QNN, config);
    constexpr int kSize = 64;
    std::vector<uint8_t> rgb(static_cast<size_t>(kSize * kSize * 3), 114);
    float ms = 0.0f;
    const std::vector<DashadasDetection> boxes = det.detect(rgb.data(), kSize, kSize, &ms);
    std::printf("device=%s model=%s detections=%zu inference_ms=%.1f\n", det.device_name(),
                det.model_name(), boxes.size(), static_cast<double>(ms));
    for (const DashadasDetection &box : boxes) {
      std::printf("person %.3f  [%.1f, %.1f, %.1f, %.1f]\n", static_cast<double>(box.score),
                  static_cast<double>(box.x1), static_cast<double>(box.y1),
                  static_cast<double>(box.x2), static_cast<double>(box.y2));
    }
  } catch (const std::exception &ex) {
    std::fprintf(stderr, "sample_client: %s\n", ex.what());
    return 1;
  }
  return 0;
}
