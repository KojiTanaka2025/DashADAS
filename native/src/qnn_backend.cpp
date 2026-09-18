#include "backend.hpp"

#include <dlfcn.h>

#include <cstdarg>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "QnnBackend.h"
#include "QnnCommon.h"
#include "QnnContext.h"
#include "QnnDevice.h"
#include "QnnGraph.h"
#include "QnnInterface.h"
#include "QnnLog.h"
#include "QnnTensor.h"
#include "QnnTypes.h"

namespace dashadas {
namespace {

using QnnInterfaceGetProvidersFn = Qnn_ErrorHandle_t (*)(const QnnInterface_t ***, uint32_t *);

enum ModelError {
  MODEL_NO_ERROR = 0
};

/* Layout must match qnn_wrapper_api::GraphInfo_t in the converted model .so. */
struct GraphInfo {
  Qnn_GraphHandle_t graph;
  char *graphName;
  Qnn_Tensor_t *inputTensors;
  uint32_t numInputTensors;
  Qnn_Tensor_t *outputTensors;
  uint32_t numOutputTensors;
};

struct GraphConfigInfo {
  char *graphName;
  const QnnGraph_Config_t **graphConfigs;
};

using ComposeGraphsFn = ModelError (*)(Qnn_BackendHandle_t,
                                       QNN_INTERFACE_VER_TYPE,
                                       Qnn_ContextHandle_t,
                                       const GraphConfigInfo **,
                                       const uint32_t,
                                       GraphInfo ***,
                                       uint32_t *,
                                       bool,
                                       QnnLog_Callback_t,
                                       QnnLog_Level_t);
using FreeGraphsFn = ModelError (*)(GraphInfo ***, uint32_t);

uint32_t tensor_rank(const Qnn_Tensor_t &t) {
  return t.version == QNN_TENSOR_VERSION_2 ? t.v2.rank : t.v1.rank;
}

uint32_t *tensor_dims(Qnn_Tensor_t &t) {
  return t.version == QNN_TENSOR_VERSION_2 ? t.v2.dimensions : t.v1.dimensions;
}

const uint32_t *tensor_dims_const(const Qnn_Tensor_t &t) {
  return t.version == QNN_TENSOR_VERSION_2 ? t.v2.dimensions : t.v1.dimensions;
}

Qnn_ClientBuffer_t *tensor_client(Qnn_Tensor_t *t) {
  return t->version == QNN_TENSOR_VERSION_2 ? &t->v2.clientBuf : &t->v1.clientBuf;
}

void tensor_set_mem_raw(Qnn_Tensor_t *t) {
  if (t->version == QNN_TENSOR_VERSION_2) {
    t->v2.memType = QNN_TENSORMEMTYPE_RAW;
  } else {
    t->v1.memType = QNN_TENSORMEMTYPE_RAW;
  }
}

size_t tensor_float_count(const Qnn_Tensor_t &t) {
  const uint32_t rank = tensor_rank(t);
  const uint32_t *dims = tensor_dims_const(t);
  size_t count = 1;
  for (uint32_t i = 0; i < rank; ++i) {
    count *= dims[i];
  }
  return count;
}

bool clone_exec_tensor(Qnn_Tensor_t *dst, const Qnn_Tensor_t &src) {
  *dst = src;
  const uint32_t rank = tensor_rank(src);
  auto *dims = static_cast<uint32_t *>(std::malloc(sizeof(uint32_t) * rank));
  if (dims == nullptr) {
    return false;
  }
  std::memcpy(dims, tensor_dims_const(src), sizeof(uint32_t) * rank);
  if (dst->version == QNN_TENSOR_VERSION_2) {
    dst->v2.dimensions = dims;
  } else {
    dst->v1.dimensions = dims;
  }
  tensor_set_mem_raw(dst);
  const size_t count = tensor_float_count(*dst);
  const size_t bytes = count * sizeof(float);
  void *data = std::malloc(bytes);
  if (data == nullptr) {
    std::free(dims);
    return false;
  }
  std::memset(data, 0, bytes);
  Qnn_ClientBuffer_t *buf = tensor_client(dst);
  buf->data = data;
  buf->dataSize = static_cast<uint32_t>(bytes);
  return true;
}

void free_exec_tensor(Qnn_Tensor_t *t) {
  if (t == nullptr) {
    return;
  }
  Qnn_ClientBuffer_t *buf = tensor_client(t);
  std::free(buf->data);
  buf->data = nullptr;
  std::free(tensor_dims(*t));
  if (t->version == QNN_TENSOR_VERSION_2) {
    t->v2.dimensions = nullptr;
  } else {
    t->v1.dimensions = nullptr;
  }
}

std::string join_path(const std::string &a, const std::string &b) {
  if (a.empty()) {
    return b;
  }
  if (a.back() == '/') {
    return a + b;
  }
  return a + "/" + b;
}

void silent_qnn_log(const char *fmt, QnnLog_Level_t level, uint64_t timestamp, va_list args) {
  (void)fmt;
  (void)level;
  (void)timestamp;
  (void)args;
}

/* libQnnCpu.so needs LLVM libc++ / libunwind. Changing LD_LIBRARY_PATH after
   process start does not help dlopen, so load those SONAMEs first. */
void preload_runtime_deps() {
  const char *sonames[] = {"libc++.so.1", "libc++abi.so.1", "libunwind.so.1", nullptr};
  for (int i = 0; sonames[i] != nullptr; ++i) {
    dlopen(sonames[i], RTLD_NOW | RTLD_GLOBAL);
  }
  const char *paths[] = {"/usr/lib/llvm-19/lib/libunwind.so.1",
                         "/usr/lib/llvm-18/lib/libunwind.so.1",
                         "/usr/lib/x86_64-linux-gnu/libunwind.so.1",
                         nullptr};
  for (int i = 0; paths[i] != nullptr; ++i) {
    dlopen(paths[i], RTLD_NOW | RTLD_GLOBAL);
  }
}

}  // namespace

class QnnBackend final : public Backend {
 public:
  ~QnnBackend() override { shutdown(); }

  /* SampleApp sequence: dlopen backend → getProviders → log/backend/device/context
     → dlopen model → composeGraphs → graphFinalize → allocate IO tensors. */
  int init(const DashadasConfig &config, std::string *error) {
    sdk_root_ = config.qnn_sdk_root ? config.qnn_sdk_root : "";
    model_path_ = config.qnn_model ? config.qnn_model : "";
    if (sdk_root_.empty() || model_path_.empty()) {
      *error = "QNN sdk root and model path are required";
      return DASHADAS_ERR_INVALID;
    }
    const std::string libdir = join_path(sdk_root_, "lib/x86_64-linux-clang");
    backend_path_ = config.qnn_backend_lib && config.qnn_backend_lib[0]
                        ? std::string(config.qnn_backend_lib)
                        : join_path(libdir, "libQnnCpu.so");

    preload_runtime_deps();
    backend_lib_ = dlopen(backend_path_.c_str(), RTLD_NOW | RTLD_GLOBAL);
    if (backend_lib_ == nullptr) {
      *error = std::string("dlopen backend failed: ") + dlerror();
      return DASHADAS_ERR_LOAD;
    }
    auto get_providers =
        reinterpret_cast<QnnInterfaceGetProvidersFn>(dlsym(backend_lib_, "QnnInterface_getProviders"));
    if (get_providers == nullptr) {
      *error = "QnnInterface_getProviders missing";
      return DASHADAS_ERR_LOAD;
    }
    const QnnInterface_t **providers = nullptr;
    uint32_t n_providers = 0;
    if (get_providers(&providers, &n_providers) != QNN_SUCCESS || providers == nullptr || n_providers == 0) {
      *error = "QnnInterface_getProviders failed";
      return DASHADAS_ERR_LOAD;
    }
    bool found = false;
    for (uint32_t i = 0; i < n_providers; ++i) {
      if (providers[i]->apiVersion.coreApiVersion.major == QNN_API_VERSION_MAJOR &&
          providers[i]->apiVersion.coreApiVersion.minor >= QNN_API_VERSION_MINOR) {
        iface_ = providers[i]->QNN_INTERFACE_VER_NAME;
        found = true;
        break;
      }
    }
    if (!found) {
      *error = "no compatible QNN interface";
      return DASHADAS_ERR_LOAD;
    }

    if (iface_.logCreate != nullptr) {
      if (iface_.logCreate(silent_qnn_log, QNN_LOG_LEVEL_ERROR, &log_) != QNN_SUCCESS) {
        log_ = nullptr;
      }
    }
    if (iface_.backendCreate == nullptr ||
        iface_.backendCreate(log_, nullptr, &backend_) != QNN_BACKEND_NO_ERROR) {
      *error = "QNN backendCreate failed";
      return DASHADAS_ERR_LOAD;
    }
    if (iface_.deviceCreate != nullptr) {
      const Qnn_ErrorHandle_t device_status = iface_.deviceCreate(log_, nullptr, &device_);
      if (device_status != QNN_SUCCESS) {
        device_ = nullptr;
      }
    }
    if (iface_.contextCreate == nullptr ||
        iface_.contextCreate(backend_, device_, nullptr, &context_) != QNN_CONTEXT_NO_ERROR) {
      *error = "QNN contextCreate failed";
      return DASHADAS_ERR_LOAD;
    }

    model_lib_ = dlopen(model_path_.c_str(), RTLD_NOW | RTLD_LOCAL);
    if (model_lib_ == nullptr) {
      *error = std::string("dlopen model failed: ") + dlerror();
      return DASHADAS_ERR_LOAD;
    }
    compose_ = reinterpret_cast<ComposeGraphsFn>(dlsym(model_lib_, "QnnModel_composeGraphs"));
    free_graphs_ = reinterpret_cast<FreeGraphsFn>(dlsym(model_lib_, "QnnModel_freeGraphsInfo"));
    if (compose_ == nullptr || free_graphs_ == nullptr) {
      *error = "model is missing QnnModel_composeGraphs";
      return DASHADAS_ERR_LOAD;
    }
    if (compose_(backend_, iface_, context_, nullptr, 0, &graphs_, &graph_count_, false, silent_qnn_log,
                 QNN_LOG_LEVEL_ERROR) != MODEL_NO_ERROR ||
        graphs_ == nullptr || graph_count_ == 0) {
      *error = "QnnModel_composeGraphs failed";
      return DASHADAS_ERR_LOAD;
    }
    if (graphs_[0] == nullptr) {
      *error = "QnnModel_composeGraphs returned an empty graph";
      return DASHADAS_ERR_LOAD;
    }
    GraphInfo *g = graphs_[0];
    if (iface_.graphFinalize == nullptr ||
        iface_.graphFinalize(g->graph, nullptr, nullptr) != QNN_GRAPH_NO_ERROR) {
      *error = "QNN graphFinalize failed";
      return DASHADAS_ERR_LOAD;
    }
    if (g->numInputTensors < 1 || g->numOutputTensors < 1) {
      *error = "QNN graph has no IO tensors";
      return DASHADAS_ERR_LOAD;
    }
    inputs_ = static_cast<Qnn_Tensor_t *>(std::calloc(g->numInputTensors, sizeof(Qnn_Tensor_t)));
    outputs_ = static_cast<Qnn_Tensor_t *>(std::calloc(g->numOutputTensors, sizeof(Qnn_Tensor_t)));
    if (inputs_ == nullptr || outputs_ == nullptr) {
      *error = "out of memory";
      return DASHADAS_ERR_NO_MEMORY;
    }
    n_in_ = g->numInputTensors;
    n_out_ = g->numOutputTensors;
    for (uint32_t i = 0; i < n_in_; ++i) {
      if (!clone_exec_tensor(&inputs_[i], g->inputTensors[i])) {
        *error = "failed to allocate QNN input tensor";
        return DASHADAS_ERR_NO_MEMORY;
      }
    }
    for (uint32_t i = 0; i < n_out_; ++i) {
      if (!clone_exec_tensor(&outputs_[i], g->outputTensors[i])) {
        *error = "failed to allocate QNN output tensor";
        return DASHADAS_ERR_NO_MEMORY;
      }
    }
    const size_t in_count = tensor_float_count(inputs_[0]);
    const size_t out_count = tensor_float_count(outputs_[0]);
    if (in_count != 1ull * 640ull * 640ull * 3ull) {
      *error = "unexpected QNN input size";
      return DASHADAS_ERR_LOAD;
    }
    const uint32_t rank = tensor_rank(outputs_[0]);
    const uint32_t *dims = tensor_dims_const(outputs_[0]);
    if (rank >= 3 && dims[1] == 84) {
      out_channels_ = 84;
      out_anchors_ = static_cast<int>(dims[2]);
      out_transposed_ = false;
    } else if (rank >= 3 && dims[2] == 84) {
      out_channels_ = 84;
      out_anchors_ = static_cast<int>(dims[1]);
      out_transposed_ = true;
    } else if (out_count % 84u == 0) {
      out_channels_ = 84;
      out_anchors_ = static_cast<int>(out_count / 84u);
      out_transposed_ = false;
    } else {
      *error = "unexpected QNN output size";
      return DASHADAS_ERR_LOAD;
    }
    return DASHADAS_OK;
  }

  /* Converter rewrote the ONNX input to NHWC; feeding NCHW produces garbage boxes. */
  int infer_nhwc(const Letterbox &input, std::vector<float> *output, int *channels, int *anchors) override {
    Qnn_ClientBuffer_t *inbuf = tensor_client(&inputs_[0]);
    const size_t bytes = input.nhwc.size() * sizeof(float);
    if (inbuf->data == nullptr || inbuf->dataSize < bytes) {
      return DASHADAS_ERR_INFER;
    }
    std::memcpy(inbuf->data, input.nhwc.data(), bytes);
    GraphInfo *g = graphs_[0];
    const Qnn_ErrorHandle_t status =
        iface_.graphExecute(g->graph, inputs_, n_in_, outputs_, n_out_, nullptr, nullptr);
    if (status != QNN_GRAPH_NO_ERROR) {
      return DASHADAS_ERR_INFER;
    }
    Qnn_ClientBuffer_t *outbuf = tensor_client(&outputs_[0]);
    const size_t count = tensor_float_count(outputs_[0]);
    const float *src = static_cast<const float *>(outbuf->data);
    if (out_transposed_) {
      output->assign(static_cast<size_t>(out_channels_) * static_cast<size_t>(out_anchors_), 0.0f);
      for (int a = 0; a < out_anchors_; ++a) {
        for (int c = 0; c < out_channels_; ++c) {
          (*output)[c * out_anchors_ + a] = src[a * out_channels_ + c];
        }
      }
    } else {
      output->assign(src, src + static_cast<std::ptrdiff_t>(count));
    }
    *channels = out_channels_;
    *anchors = out_anchors_;
    return DASHADAS_OK;
  }

  const char *model_name() const override { return model_path_.c_str(); }
  bool uses_nhwc() const override { return true; }

 private:
  void shutdown() {
    if (inputs_ != nullptr) {
      for (uint32_t i = 0; i < n_in_; ++i) {
        free_exec_tensor(&inputs_[i]);
      }
      std::free(inputs_);
      inputs_ = nullptr;
    }
    if (outputs_ != nullptr) {
      for (uint32_t i = 0; i < n_out_; ++i) {
        free_exec_tensor(&outputs_[i]);
      }
      std::free(outputs_);
      outputs_ = nullptr;
    }
    if (free_graphs_ != nullptr && graphs_ != nullptr) {
      free_graphs_(&graphs_, graph_count_);
      graphs_ = nullptr;
    }
    if (iface_.contextFree != nullptr && context_ != nullptr) {
      iface_.contextFree(context_, nullptr);
      context_ = nullptr;
    }
    if (iface_.deviceFree != nullptr && device_ != nullptr) {
      iface_.deviceFree(device_);
      device_ = nullptr;
    }
    if (iface_.backendFree != nullptr && backend_ != nullptr) {
      iface_.backendFree(backend_);
      backend_ = nullptr;
    }
    if (iface_.logFree != nullptr && log_ != nullptr) {
      iface_.logFree(log_);
      log_ = nullptr;
    }
    if (model_lib_ != nullptr) {
      dlclose(model_lib_);
      model_lib_ = nullptr;
    }
    if (backend_lib_ != nullptr) {
      dlclose(backend_lib_);
      backend_lib_ = nullptr;
    }
  }

  std::string sdk_root_;
  std::string model_path_;
  std::string backend_path_;
  void *backend_lib_ = nullptr;
  void *model_lib_ = nullptr;
  QNN_INTERFACE_VER_TYPE iface_{};
  Qnn_LogHandle_t log_ = nullptr;
  Qnn_BackendHandle_t backend_ = nullptr;
  Qnn_DeviceHandle_t device_ = nullptr;
  Qnn_ContextHandle_t context_ = nullptr;
  GraphInfo **graphs_ = nullptr;
  uint32_t graph_count_ = 0;
  ComposeGraphsFn compose_ = nullptr;
  FreeGraphsFn free_graphs_ = nullptr;
  Qnn_Tensor_t *inputs_ = nullptr;
  Qnn_Tensor_t *outputs_ = nullptr;
  uint32_t n_in_ = 0;
  uint32_t n_out_ = 0;
  int out_channels_ = 84;
  int out_anchors_ = 8400;
  bool out_transposed_ = false;
};

std::unique_ptr<Backend> create_qnn_backend(const DashadasConfig &config, std::string *error) {
  auto backend = std::make_unique<QnnBackend>();
  const int rc = backend->init(config, error);
  if (rc != DASHADAS_OK) {
    return nullptr;
  }
  return backend;
}

}  // namespace dashadas
