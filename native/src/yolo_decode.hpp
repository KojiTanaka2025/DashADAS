#pragma once

#include "dashadas/perception.h"

#include <vector>

namespace dashadas {

/* raw is (C, A) with C = 4 + classes. Only COCO person (class 0) is kept. */
std::vector<DashadasDetection> decode_yolo_persons(const float *raw,
                                                   int channels,
                                                   int anchors,
                                                   int image_w,
                                                   int image_h,
                                                   float scale,
                                                   float pad_x,
                                                   float pad_y,
                                                   float conf,
                                                   float iou);

}  // namespace dashadas
