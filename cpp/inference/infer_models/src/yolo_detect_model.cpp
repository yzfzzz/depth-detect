#include "yolo_detect_model.h"

#include "postprocess.h"
#include "preprocess.h"
#include "public.h"

#include <cstring>

void YoloDetectModel::init(const std::string & model_path,
                           int                 raw_img_w,
                           int                 raw_img_h,
                           float               nms_thresh,
                           float               conf_thresh,
                           int                 num_class) {
    model_path_  = model_path;
    raw_img_w_   = raw_img_w;
    raw_img_h_   = raw_img_h;
    nms_thresh_  = nms_thresh;
    conf_thresh_ = conf_thresh;
    num_class_   = num_class;
    // 初始化基类
    if (!initInferenceBackend()) {
        APP_ERROR("Failed to initialize base model");
        return;
    }

    // 获取输入输出维度
    auto input_dims  = getInputDims();
    auto output_dims = getOutputDims();

    // 设置输入尺寸
    input_h_ = input_dims[2];
    input_w_ = input_dims[3];

    APP_INFO("YOLO model input size: {}x{}", input_w_, input_h_);

    // 计算输出候选框数量（YOLOv8 输出格式: [1, num_class+4, candidates]）
    output_candidates_ = output_dims[2];

    // 计算输出数据总大小
    int output_size = 1;
    for (int dim : output_dims) {
        output_size *= dim;
    }

    APP_INFO("YOLO model output candidates: {}", output_candidates_);
    APP_INFO("YOLO model output size: {}", output_size);

    // 定义分配固定主机内存的 lambda 函数
    auto alloc_cuda_pinned = [](size_t bytes) {
        void * ptr = nullptr;
        CHECK_CUDA(cudaHostAlloc(&ptr, bytes, cudaHostAllocDefault));
        return ptr;
    };

    // 准备主机输出数据空间
    // 格式: [count, bbox1, bbox2, ...]
    // 每个 bbox: [x1, y1, x2, y2, conf, class_id, keep_flag]
    size_t h_output_data_size = 1 + MAX_NUM_OUTPUT_BBOX * NUM_BOX_ELEMENT;
    h_output_data_.reset(
        static_cast<float *>(alloc_cuda_pinned(h_output_data_size * sizeof(float))));

    // 定义分配设备内存的 lambda 函数
    auto alloc_cuda = [](size_t bytes) {
        void * ptr = nullptr;
        CHECK_CUDA(cudaMalloc(&ptr, bytes));
        return ptr;
    };

    // 准备设备输入输出缓冲区
    // d_buffer_[0]: 输入缓冲区 [1, 3, H, W]
    d_buffer_[0].reset(alloc_cuda(3 * input_h_ * input_w_ * sizeof(float)));

    // d_buffer_[1]: 输出缓冲区 [1, num_class+4, candidates]
    d_buffer_[1].reset(alloc_cuda(output_size * sizeof(float)));

    // 转置缓冲区（用于后处理）
    d_transpose_.reset(static_cast<float *>(alloc_cuda(output_size * sizeof(float))));

    // 解码缓冲区（用于 NMS）
    d_decode_.reset(static_cast<float *>(
        alloc_cuda((1 + MAX_NUM_OUTPUT_BBOX * NUM_BOX_ELEMENT) * sizeof(float))));

    // 源数据缓冲区（原始图像数据）
    d_src_data_.reset(
        static_cast<uchar *>(alloc_cuda(sizeof(uchar) * raw_img_h_ * raw_img_w_ * 3)));

    // 中间数据缓冲区（预处理后的图像数据）
    d_mid_data_.reset(static_cast<uchar *>(alloc_cuda(sizeof(uchar) * input_h_ * input_w_ * 3)));

    APP_INFO("YOLO model initialized successfully");
}

void YoloDetectModel::cudaPreProcess(FrameInputContext & frame_input_context) {
    preprocess_v2(static_cast<float *>(d_buffer_[0].get()), frame_input_context.d_raw_img_.get(),
                  d_mid_data_.get(), raw_img_h_, raw_img_w_, input_h_, input_w_, stream_);
}

void YoloDetectModel::cudaPostProcess(FrameInputContext & frame_input_context) {
    // 转置
    transpose(static_cast<float *>(d_buffer_[1].get()), d_transpose_.get(), output_candidates_,
              num_class_ + 4, stream_);

    // 解码
    decode(d_transpose_.get(), d_decode_.get(), output_candidates_, num_class_, conf_thresh_,
           MAX_NUM_OUTPUT_BBOX, NUM_BOX_ELEMENT, stream_);

    // NMS
    nms(d_decode_.get(), nms_thresh_, MAX_NUM_OUTPUT_BBOX, NUM_BOX_ELEMENT, stream_);

    // 拷贝到主机
    CHECK_CUDA(cudaMemcpyAsync(h_output_data_.get(), d_decode_.get(),
                               (1 + MAX_NUM_OUTPUT_BBOX * NUM_BOX_ELEMENT) * sizeof(float),
                               cudaMemcpyDeviceToHost, stream_));
}

bool YoloDetectModel::runInferenceAsync(FrameInputContext & frame_input_context) {
    if (!isBackendInitialized()) {
        APP_ERROR("Model not initialized");
        return false;
    }

    // 异步预处理
    cudaPreProcess(frame_input_context);
    // 异步推理
    backend_->runInferenceAsync(d_buffer_[0].get(), d_buffer_[1].get(), stream_);
    // 异步后处理
    cudaPostProcess(frame_input_context);
    return true;
}

void YoloDetectModel::getInferOutputResult(InferOutputContext & infer_output_context) {
    waitAsync();
    std::vector<Detection> vDetections;
    int count = std::min(static_cast<int>(h_output_data_.get()[0]), MAX_NUM_OUTPUT_BBOX);

    for (int i = 0; i < count; i++) {
        int pos      = 1 + i * NUM_BOX_ELEMENT;
        int keepFlag = static_cast<int>(h_output_data_.get()[pos + 6]);

        if (keepFlag == 1) {
            Detection det;
            memcpy(det.bbox.data(), &h_output_data_.get()[pos], 4 * sizeof(float));
            det.conf    = h_output_data_.get()[pos + 4];
            det.classId = static_cast<int>(h_output_data_.get()[pos + 5]);
            float r_w   = input_w_ / (raw_img_w_ * 1.0);
            float r_h   = input_h_ / (raw_img_h_ * 1.0);
            float r     = std::min(r_w, r_h);
            float pad_h = (input_h_ - r * raw_img_h_) / 2;
            float pad_w = (input_w_ - r * raw_img_w_) / 2;
            det.bbox[0] = (det.bbox[0] - pad_w) / r;
            det.bbox[1] = (det.bbox[1] - pad_h) / r;
            det.bbox[2] = (det.bbox[2] - pad_w) / r;
            det.bbox[3] = (det.bbox[3] - pad_h) / r;
            vDetections.push_back(det);
        }
    }

    infer_output_context.detections = vDetections;
}

bool YoloDetectModel::runInference(FrameInputContext & frame_input_context) {
    if (!isBackendInitialized()) {
        APP_ERROR("Model not initialized");
        return false;
    }
    APP_INFO("Running synchronous inference for YOLO model...");
    return false;
}

void YoloDetectModel::cvMatPreProcess(FrameInputContext & frame_input_context) {
    APP_INFO("Running CPU pre-processing for YOLO model...");
}

void YoloDetectModel::cvMatPostProcess(FrameInputContext & frame_input_context) {
    APP_INFO("Running CPU post-processing for YOLO model...");
}
