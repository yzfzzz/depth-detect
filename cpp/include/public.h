#ifndef PUBLIC_H
#define PUBLIC_H

#include <cuda_fp16.h>
#include <cuda_runtime_api.h>
#include <NvInfer.h>
#include <spdlog/spdlog.h>
#include <unistd.h>

#include <cmath>
#include <nvtx3/nvtx3.hpp>
#include <opencv2/opencv.hpp>
#include <string>

#define CHECK_CUDA(call)                                                              \
    do {                                                                              \
        cudaError_t status = call;                                                    \
        if (status != cudaSuccess) {                                                  \
            auto logger = spdlog::get("app");                                         \
            if (logger) {                                                             \
                logger->error("CUDA error at {}:{} - {}", __FILE__, __LINE__,         \
                              cudaGetErrorString(status));                            \
            } else {                                                                  \
                std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << " - " \
                          << cudaGetErrorString(status) << std::endl;                 \
            }                                                                         \
            exit(EXIT_FAILURE);                                                       \
        }                                                                             \
    } while (0)

const std::vector<std::string> V_CLASS_NAMES{ "person",        "bicycle",      "car",
                                              "motorcycle",    "airplane",     "bus",
                                              "train",         "truck",        "boat",
                                              "traffic light", "fire hydrant", "stop sign",
                                              "parking meter", "bench",        "bird",
                                              "cat",           "dog",          "horse",
                                              "sheep",         "cow",          "elephant",
                                              "bear",          "zebra",        "giraffe",
                                              "backpack",      "umbrella",     "handbag",
                                              "tie",           "suitcase",     "frisbee",
                                              "skis",          "snowboard",    "sports ball",
                                              "kite",          "baseball bat", "baseball glove",
                                              "skateboard",    "surfboard",    "tennis racket",
                                              "bottle",        "wine glass",   "cup",
                                              "fork",          "knife",        "spoon",
                                              "bowl",          "banana",       "apple",
                                              "sandwich",      "orange",       "broccoli",
                                              "carrot",        "hot dog",      "pizza",
                                              "donut",         "cake",         "chair",
                                              "couch",         "potted plant", "bed",
                                              "dining table",  "toilet",       "tv",
                                              "laptop",        "mouse",        "remote",
                                              "keyboard",      "cell phone",   "microwave",
                                              "oven",          "toaster",      "sink",
                                              "refrigerator",  "book",         "clock",
                                              "vase",          "scissors",     "teddy bear",
                                              "hair drier",    "toothbrush" };

#endif  // PUBLIC_H
