/* CLI over the C API: load an image, print person boxes. See native/API.md. */
#include "dashadas/perception.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#ifdef DASHADAS_HAS_STB
#define STB_IMAGE_IMPLEMENTATION
#define STBI_ONLY_JPEG
#define STBI_ONLY_PNG
#define STBI_ONLY_BMP
#define STBI_ONLY_PPM
#include "stb_image.h"
#endif

namespace {

bool load_ppm(const char *path, std::vector<uint8_t> *rgb, int *width, int *height) {
  FILE *fp = std::fopen(path, "rb");
  if (fp == nullptr) {
    return false;
  }
  char magic[8] = {0};
  if (std::fscanf(fp, "%7s", magic) != 1 || std::strcmp(magic, "P6") != 0) {
    std::fclose(fp);
    return false;
  }
  int maxval = 0;
  if (std::fscanf(fp, "%d %d %d", width, height, &maxval) != 3 || *width <= 0 || *height <= 0 ||
      maxval != 255) {
    std::fclose(fp);
    return false;
  }
  const int ch = std::fgetc(fp);
  if (ch != '\n' && ch != ' ' && ch != '\r') {
    std::fclose(fp);
    return false;
  }
  const size_t nbytes = static_cast<size_t>(*width) * static_cast<size_t>(*height) * 3u;
  rgb->resize(nbytes);
  const size_t nread = std::fread(rgb->data(), 1, nbytes, fp);
  std::fclose(fp);
  return nread == nbytes;
}

bool load_image(const char *path, std::vector<uint8_t> *rgb, int *width, int *height) {
#ifdef DASHADAS_HAS_STB
  int w = 0;
  int h = 0;
  int c = 0;
  unsigned char *data = stbi_load(path, &w, &h, &c, 3);
  if (data != nullptr) {
    rgb->assign(data, data + static_cast<size_t>(w) * static_cast<size_t>(h) * 3u);
    stbi_image_free(data);
    *width = w;
    *height = h;
    return true;
  }
#endif
  return load_ppm(path, rgb, width, height);
}

const char *arg_value(int argc, char **argv, const char *flag, const char *fallback) {
  const size_t n = std::strlen(flag);
  for (int i = 1; i < argc; ++i) {
    if (std::strcmp(argv[i], flag) == 0 && i + 1 < argc) {
      return argv[i + 1];
    }
    if (std::strncmp(argv[i], flag, n) == 0 && argv[i][n] == '=') {
      return argv[i] + n + 1;
    }
  }
  return fallback;
}

bool has_flag(int argc, char **argv, const char *flag) {
  for (int i = 1; i < argc; ++i) {
    if (std::strcmp(argv[i], flag) == 0) {
      return true;
    }
  }
  return false;
}

}  // namespace

int main(int argc, char **argv) {
  if (has_flag(argc, argv, "-h") || has_flag(argc, argv, "--help") || argc < 2) {
    std::fprintf(stderr,
                 "Usage: %s --image path.jpg [--sdk QNN_SDK_ROOT] [--model libyolo11n.so]\n"
                 "          [--backend libQnnCpu.so] [--conf 0.25] [--iou 0.45]\n",
                 argv[0]);
    return 2;
  }

  const char *image = arg_value(argc, argv, "--image", nullptr);
  const char *sdk = arg_value(argc, argv, "--sdk", std::getenv("QNN_SDK_ROOT"));
  const char *model = arg_value(argc, argv, "--model", std::getenv("DASHADAS_QNN_MODEL"));
  const char *backend = arg_value(argc, argv, "--backend", std::getenv("DASHADAS_QNN_BACKEND"));
  const char *conf_s = arg_value(argc, argv, "--conf", "0.25");
  const char *iou_s = arg_value(argc, argv, "--iou", "0.45");
  if (image == nullptr) {
    std::fprintf(stderr, "error: --image is required\n");
    return 2;
  }
  if (sdk == nullptr || model == nullptr) {
    std::fprintf(stderr, "error: --sdk and --model are required (or QNN_SDK_ROOT / DASHADAS_QNN_MODEL)\n");
    return 2;
  }

  std::vector<uint8_t> rgb;
  int width = 0;
  int height = 0;
  if (!load_image(image, &rgb, &width, &height)) {
    std::fprintf(stderr, "error: failed to load image %s (JPEG/PNG needs stb; PPM P6 always works)\n",
                 image);
    return 1;
  }

  DashadasConfig config;
  dashadas_config_init(&config);
  config.qnn_sdk_root = sdk;
  config.qnn_model = model;
  config.qnn_backend_lib = backend;
  config.conf = std::strtof(conf_s, nullptr);
  config.iou = std::strtof(iou_s, nullptr);

  DashadasPerception *ctx = dashadas_create(DASHADAS_DEVICE_QNN, &config);
  if (ctx == nullptr) {
    std::fprintf(stderr, "error: dashadas_create: %s\n", dashadas_last_create_error());
    return 1;
  }

  DashadasDetection dets[256];
  int count = 0;
  float ms = 0.0f;
  const int rc = dashadas_detect(ctx, rgb.data(), width, height, dets, 256, &count, &ms);
  if (rc != DASHADAS_OK) {
    std::fprintf(stderr, "error: dashadas_detect: %s\n", dashadas_last_error(ctx));
    dashadas_destroy(ctx);
    return 1;
  }

  std::printf("device=%s  %dx%d  detections=%d  inference_ms=%.1f\n", dashadas_device_name(ctx),
              width, height, count, static_cast<double>(ms));
  for (int i = 0; i < count; ++i) {
    std::printf("person %.4f  [%.1f, %.1f, %.1f, %.1f]\n", static_cast<double>(dets[i].score),
                static_cast<double>(dets[i].x1), static_cast<double>(dets[i].y1),
                static_cast<double>(dets[i].x2), static_cast<double>(dets[i].y2));
  }
  dashadas_destroy(ctx);
  return 0;
}
