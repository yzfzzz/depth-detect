#pragma once

#include <cstddef>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <vector>
typedef Eigen::Matrix<float, 1, 4, Eigen::RowMajor>                DETECTBOX;
typedef Eigen::Matrix<float, -1, 4, Eigen::RowMajor>               DETECTBOXSS;
typedef Eigen::Matrix<float, 1, 128, Eigen::RowMajor>              FEATURE;
typedef Eigen::Matrix<float, Eigen::Dynamic, 128, Eigen::RowMajor> FEATURESS;
// typedef std::vector<FEATURE> FEATURESS;

// 卡尔曼滤波器状态类型：均值、协方差、观测矩阵
typedef Eigen::Matrix<float, 1, 8, Eigen::RowMajor> KAL_MEAN;
typedef Eigen::Matrix<float, 8, 8, Eigen::RowMajor> KAL_COVA;
typedef Eigen::Matrix<float, 1, 4, Eigen::RowMajor> KAL_HMEAN;
typedef Eigen::Matrix<float, 4, 4, Eigen::RowMajor> KAL_HCOVA;
using KAL_DATA  = std::pair<KAL_MEAN, KAL_COVA>;
using KAL_HDATA = std::pair<KAL_HMEAN, KAL_HCOVA>;

// 线性分配结果：匹配对列表 + 未匹配索引

// 跟踪匹配中间结果
using MATCH_DATA = std::pair<int, int>;

typedef struct t {
    std::vector<MATCH_DATA> matches;
    std::vector<int>        unmatched_tracks;
    std::vector<int>        unmatched_detections;
} TRACHER_MATCHD;

// 线性分配代价矩阵类型 (动态大小)
