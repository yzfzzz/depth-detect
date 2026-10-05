#ifndef PUBLIC_H
#define PUBLIC_H

#include <spdlog/spdlog.h>
#include <unistd.h>

#include <cmath>
#include <opencv2/opencv.hpp>
#include <string>

#ifdef HAS_NVTX3
#    include <nvtx3/nvtx3.hpp>
#endif

// CHECK_CUDA 宏已迁至 cpp/inference/infer_backend/include/tensorrt/cuda_utils.h，
// 本头文件保持无 CUDA 依赖，CPU-only 构建不再被 CUDA 头文件阻断

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

enum COCO80 {
    PERSON     = 0,
    BICYCLE    = 1,
    CAR        = 2,
    MOTORCYCLE = 3,
    AIRPLANE   = 4,
    BUS        = 5,
    TRAIN      = 6,
    TRUCK      = 7
};
#endif  // PUBLIC_H
