#include "yolo_detect_model.h"

#include "logger_manager.h"
#include "postprocess.h"
#include "preprocess.h"
#include "public.h"

#include <cstddef>
#include <cstring>
#include <opencv2/dnn.hpp>
#include <unordered_set>

void YoloDetectModel::init(std::map<std::string, std::string> model_path,
                           int                                raw_img_w,
                           int                                raw_img_h,
                           float                              nms_thresh,
                           float                              conf_thresh,
                           int                                num_class,
                           bool                               use_gpu) {
    APP_INFO(
        "YOLO model init: raw_img_w: {}, raw_img_h: {}, nms_thresh: {}, "
        "conf_thresh: {}, num_class: {}, use_gpu: {}",
        raw_img_w, raw_img_h, nms_thresh, conf_thresh, num_class, use_gpu);
    BaseModel::init(model_path, raw_img_w, raw_img_h, use_gpu);
    nms_thresh_  = nms_thresh;
    conf_thresh_ = conf_thresh;
    num_class_   = num_class;

    // 计算输出候选框数量（YOLOv8 输出格式: [1, num_class+4, candidates]）
    output_candidates_ = getOutputDims()[2];

    APP_INFO("YOLO model output candidates: {}", output_candidates_);
    size_t h_output_data_size = 1 + MAX_NUM_OUTPUT_BBOX * NUM_BOX_ELEMENT;
    if (backend_->getBackendType() == BackendType::TensorRT) {
        // 计算输出数据总大小
        size_t output_size       = getOutputByteSize(0);
        // 定义分配固定主机内存的 lambda 函数
        auto   alloc_cuda_pinned = [](size_t bytes) {
            void * ptr = nullptr;
            CHECK_CUDA(cudaHostAlloc(&ptr, bytes, cudaHostAllocDefault));
            return ptr;
        };

        // 准备主机输出数据空间
        // 格式: [count, bbox1, bbox2, ...]
        // 每个 bbox: [x1, y1, x2, y2, conf, class_id, keep_flag]

        h_infer_out_pinned_.reset(
            static_cast<float *>(alloc_cuda_pinned(h_output_data_size * sizeof(float))));

        // 定义分配设备内存的 lambda 函数
        auto alloc_cuda = [](size_t bytes) {
            void * ptr = nullptr;
            CHECK_CUDA(cudaMalloc(&ptr, bytes));
            return ptr;
        };

        // 准备设备输入输出缓冲区
        // d_infer_io_[0]: 输入缓冲区 [1, 3, H, W]
        d_infer_io_.resize(getNumOutputs() + 1);  // 输入 + 输出
        d_infer_io_[0].reset(alloc_cuda(3 * input_h_ * input_w_ * sizeof(float)));

        // d_infer_io_[1]: 输出缓冲区 [1, num_class+4, candidates]
        d_infer_io_[1].reset(alloc_cuda(output_size));

        // 转置缓冲区（用于后处理）
        d_transpose_.reset(static_cast<float *>(alloc_cuda(output_size)));

        // 解码缓冲区（用于 NMS）
        d_decode_.reset(static_cast<float *>(
            alloc_cuda((1 + MAX_NUM_OUTPUT_BBOX * NUM_BOX_ELEMENT) * sizeof(float))));

        // 源数据缓冲区（原始图像数据）
        d_src_data_.reset(
            static_cast<uchar *>(alloc_cuda(sizeof(uchar) * raw_img_h_ * raw_img_w_ * 3)));

        // 中间数据缓冲区（预处理后的图像数据）
        d_mid_data_.reset(
            static_cast<uchar *>(alloc_cuda(sizeof(uchar) * input_h_ * input_w_ * 3)));

    } else if (backend_->getBackendType() == BackendType::OnnxRuntime) {
        // ONNX Runtime CPU 后端，准备主机输出数据空间
        int output_num = getNumOutputs();
        h_infer_out_.resize(output_num);
        for (int i = 0; i < output_num; ++i) {
            auto output_size = getOutputByteSize(i);
            h_infer_out_[i].resize(output_size / sizeof(float));
        }
    }

    APP_INFO("YOLO model initialized successfully");
}

void YoloDetectModel::cudaPreProcess(FrameInputContext & frame_input_context) {
    if (frame_input_context.d_raw_img_ == nullptr) {
        APP_ERROR("Input image buffer is not allocated on GPU");
        return;
    }
    preprocess_v2(static_cast<float *>(d_infer_io_[0].get()), frame_input_context.d_raw_img_.get(),
                  d_mid_data_.get(), raw_img_h_, raw_img_w_, input_h_, input_w_, stream_);
}

void YoloDetectModel::cudaPostProcess(FrameInputContext & frame_input_context) {
    // YOLO GPU 后处理流水线：转置 → 解码（提取类别/置信度） → NMS → 异步拷贝回主机
    transpose(static_cast<float *>(d_infer_io_[getOutputIndexFromName("output0")].get()),
              d_transpose_.get(), output_candidates_, num_class_ + 4, stream_);

    decode(d_transpose_.get(), d_decode_.get(), output_candidates_, num_class_, conf_thresh_,
           MAX_NUM_OUTPUT_BBOX, NUM_BOX_ELEMENT, stream_);

    nms(d_decode_.get(), nms_thresh_, MAX_NUM_OUTPUT_BBOX, NUM_BOX_ELEMENT, stream_);

    // 拷贝到主机
    CHECK_CUDA(cudaMemcpyAsync(h_infer_out_pinned_.get(), d_decode_.get(),
                               (1 + MAX_NUM_OUTPUT_BBOX * NUM_BOX_ELEMENT) * sizeof(float),
                               cudaMemcpyDeviceToHost, stream_));
}

void YoloDetectModel::getInferOutputResult(InferOutputContext & infer_output_context) {
    synchronizeStream();
    std::vector<Detection> vDetections;
    int count = std::min(static_cast<int>(h_infer_out_pinned_.get()[0]), MAX_NUM_OUTPUT_BBOX);

    for (int i = 0; i < count; i++) {
        int pos      = 1 + i * NUM_BOX_ELEMENT;
        int keepFlag = static_cast<int>(h_infer_out_pinned_.get()[pos + 6]);

        if (keepFlag == 1) {
            Detection det;
            memcpy(det.bbox.data(), &h_infer_out_pinned_.get()[pos], 4 * sizeof(float));
            det.conf    = h_infer_out_pinned_.get()[pos + 4];
            det.classId = static_cast<int>(h_infer_out_pinned_.get()[pos + 5]);
            // 将模型坐标系下的 bbox 反算回原始图像坐标系
            // 模型输入经过了 letterbox 缩放 + 灰边填充，此处逆向：先减去灰边偏移，再除以缩放比例
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

std::vector<float> YoloDetectModel::cvMatPreProcess(FrameInputContext & frame_input_context) {
    // Letterbox 缩放：保持宽高比缩放到模型输入尺寸，不足部分用灰色（128）填充
    // 这样避免直接拉伸导致的形变，保留物体真实比例
    float scale = std::min((float) input_w_ / frame_input_context.raw_img.cols,
                           (float) input_h_ / frame_input_context.raw_img.rows);
    int   w     = (int) (frame_input_context.raw_img.cols * scale);
    int   h     = (int) (frame_input_context.raw_img.rows * scale);

    cv::Mat resized, padded;
    // 先按 letterbox 比例缩放到实际尺寸，再四周补灰边到目标尺寸
    cv::resize(frame_input_context.raw_img, resized, cv::Size(w, h));
    cv::copyMakeBorder(resized, padded, (input_h_ - h) / 2, input_h_ - h - (input_h_ - h) / 2,
                       (input_w_ - w) / 2, input_w_ - w - (input_w_ - w) / 2, cv::BORDER_CONSTANT,
                       cv::Scalar(128, 128, 128));

    // BGR→RGB 通道交换 + HWC→CHW 布局转换 + /255 归一化，合并为一次遍历
    std::vector<float> onnx_input_tensor;
    for (int c = 0; c < 3; ++c) {
        for (int y = 0; y < input_h_; ++y) {
            const uchar * row = padded.ptr<uchar>(y);
            for (int x = 0; x < input_w_; ++x) {
                onnx_input_tensor.emplace_back(row[x * 3 + (2 - c)] / 255.0f);
            }
        }
    }
    return onnx_input_tensor;
}

void YoloDetectModel::cvMatPostProcess(InferOutputContext & infer_output_context) {
    int           num_elements = num_class_ + 4;
    int           num_bboxes   = output_candidates_;
    const float * raw_output   = h_infer_out_[getOutputIndexFromName("output0")].data();

    // 解码：直接从 [84, 8400] 格式输出中提取检测框
    // 每个候选框有 84 个值：前 4 个是 bbox 坐标，后 80 个是类别置信度
    std::vector<cv::Rect2d> boxes;
    std::vector<float>      scores;
    std::vector<int>        classIds;
    std::vector<int>        original_indices;  // 记录满足阈值的原始候选框索引

    for (int i = 0; i < num_bboxes; ++i) {
        float confidence = 0;
        int   label      = 0;
        for (int j = 0; j < num_class_; j++) {
            float cls_score = raw_output[(j + 4) * num_bboxes + i];
            if (cls_score > confidence) {
                confidence = cls_score;
                label      = j;
            }
        }

        if (confidence < conf_thresh_) {
            continue;
        }

        float cx = raw_output[0 * num_bboxes + i];
        float cy = raw_output[1 * num_bboxes + i];
        float w  = raw_output[2 * num_bboxes + i];
        float h  = raw_output[3 * num_bboxes + i];

        boxes.emplace_back(cx - w * 0.5, cy - h * 0.5, w, h);
        scores.push_back(confidence);
        classIds.push_back(label);
        original_indices.push_back(i);

        if (boxes.size() >= MAX_NUM_OUTPUT_BBOX) {
            break;
        }
    }

    // 2. NMS：使用 cv::Rect2d 配合 double 类型的阈值 0.0
    std::vector<int> indices;
    if (!boxes.empty()) {
        cv::dnn::NMSBoxes(boxes, scores, 0.0, static_cast<double>(nms_thresh_), indices);
    }

    // 3. 组装结果：直接构造 Detection 并回写，无需中间 h_decode 缓存
    std::unordered_set<int> kept_indices(indices.begin(), indices.end());
    std::vector<Detection>  vDetections;

    for (size_t i = 0; i < boxes.size(); ++i) {
        if (kept_indices.count(i) == 0) {
            continue;
        }

        Detection det;
        det.bbox[0] = boxes[i].x;
        det.bbox[1] = boxes[i].y;
        det.bbox[2] = boxes[i].x + boxes[i].width;
        det.bbox[3] = boxes[i].y + boxes[i].height;
        det.conf    = scores[i];
        det.classId = classIds[i];

        // 坐标还原（与 getInferOutputResult 逻辑保持一致）
        float r_w   = input_w_ / (raw_img_w_ * 1.0f);
        float r_h   = input_h_ / (raw_img_h_ * 1.0f);
        float r     = std::min(r_w, r_h);
        float pad_h = (input_h_ - r * raw_img_h_) / 2.0f;
        float pad_w = (input_w_ - r * raw_img_w_) / 2.0f;
        det.bbox[0] = (det.bbox[0] - pad_w) / r;
        det.bbox[1] = (det.bbox[1] - pad_h) / r;
        det.bbox[2] = (det.bbox[2] - pad_w) / r;
        det.bbox[3] = (det.bbox[3] - pad_h) / r;

        vDetections.push_back(det);
    }

    infer_output_context.detections = vDetections;
}
