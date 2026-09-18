#pragma once

#include "dashadas/perception.h"
#include "letterbox.hpp"

#include <memory>
#include <string>
#include <vector>

namespace dashadas {

/* Internal inference plugin. Public callers never see this type. */
class Backend {
 public:
  virtual ~Backend() = default;
  virtual int infer_nhwc(const Letterbox &input, std::vector<float> *output, int *channels, int *anchors) = 0;
  virtual int infer_nchw(const Letterbox &input, std::vector<float> *output, int *channels, int *anchors) {
    (void)input;
    (void)output;
    (void)channels;
    (void)anchors;
    return DASHADAS_ERR_UNSUPPORTED;
  }
  virtual const char *model_name() const = 0;
  virtual bool uses_nhwc() const { return false; }
};

std::unique_ptr<Backend> create_qnn_backend(const DashadasConfig &config, std::string *error);

}  // namespace dashadas
