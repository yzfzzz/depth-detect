#include "yolo_detect_model.h"

#include "postprocess.h"
#include "preprocess.h"
#include "public.h"

#include <NvOnnxParser.h>
#include <opencv2/core/hal/interface.h>

#include <cassert>
#include <fstream>
#include <iostream>
#include <vector>

YoloDetectModel::YoloDetectModel(const std::string trtFile,
                                 int               raw_img_w,
                                 int               raw_img_h,
                                 int               gpuId,
                                 float             nmsThresh,
                                 float             confThresh,
                                 int               numClass) :
    trtFile_(trtFile),
    nmsThresh_(nmsThresh),
    confThresh_(confThresh),
    numClass_(numClass),
    raw_img_w_(raw_img_w),
    raw_img_h_(raw_img_h) {
    g_logger_ = Logger(nvinfer1::ILogger::Severity::kERROR);
    cudaSetDevice(gpuId);

    CHECK_CUDA(cudaStreamCreate(&stream_));

    // load engine
    getEngine();

    context_.reset(engine_->createExecutionContext());
    // Set input dimensions - use appropriate API based on platform/TensorRT version
#if defined(__aarch64__) || defined(__arm__) || NV_TENSORRT_MAJOR < 10
    // For Jetson Nano (ARM64) and older TensorRT versions
    context_->setBindingDimensions(0, nvinfer1::Dims{
                                          4, { 1, 3, input_h_, input_w_ }
    });

    // Get output dimensions using binding index
    nvinfer1::Dims outDims = engine_->getBindingDimensions(1);  // [1, 84, 8400]
#else
    // For newer TensorRT versions on x86_64
    context_->setInputShape("images", nvinfer1::Dims{
                                          4, { 1, 3, input_h_, input_w_ }
    });

    // Get output dimensions using tensor name
    nvinfer1::Dims outDims =
        context_->getTensorShape(engine_->getIOTensorName(1));  // [1, 84, 8400]
#endif

    // yolov8: [1, 30, 8400]
    // yolo26: [1, 300, 6]
    OUTPUT_CANDIDATES_ = outDims.d[2];
    int outputSize     = 1;  //yolov8 = 84*8400, yolov26 = 6*300
    for (int i = 0; i < outDims.nbDims; i++) {
        outputSize *= outDims.d[i];
    }
    if (OUTPUT_CANDIDATES_ == 6) {
        // 走yolo26推理，输出候选框较少，且已经经过nms处理，不需要再做一次nms了
        is_need_nms_                = false;
        yolo26_max_num_output_bbox_ = outDims.d[1];
        yolo26_num_box_element_     = outDims.d[2];
        std::cout << "Running YOLOv26 inference, output candidates num: " << OUTPUT_CANDIDATES_
                  << std::endl;
    }
    auto alloc_cuda_pinned = [](size_t bytes) {
        void * ptr = nullptr;
        CHECK_CUDA(cudaHostAlloc(&ptr, bytes, cudaHostAllocDefault));
        return ptr;
    };
    // prepare output data space on host
    if (!is_need_nms_) {
        h_output_data_size_ = yolo26_max_num_output_bbox_ * yolo26_num_box_element_;
    } else {
        h_output_data_size_ = 1 + MAX_NUM_OUTPUT_BBOX * NUM_BOX_ELEMENT;
    }
    h_output_data_.reset(
        static_cast<float *>(alloc_cuda_pinned(h_output_data_size_ * sizeof(float))));

    // prepare input and output space on device
    auto alloc_cuda = [](size_t bytes) {
        void * ptr = nullptr;
        CHECK_CUDA(cudaMalloc(&ptr, bytes));
        return ptr;
    };

    d_buffer_[0].reset(alloc_cuda(3 * input_h_ * input_w_ * sizeof(float)));
    d_buffer_[1].reset(alloc_cuda(outputSize * sizeof(float)));

    d_transpose_.reset(static_cast<float *>(alloc_cuda(outputSize * sizeof(float))));
    d_decode_.reset(static_cast<float *>(
        alloc_cuda((1 + MAX_NUM_OUTPUT_BBOX * NUM_BOX_ELEMENT) * sizeof(float))));

    d_src_data_.reset(
        static_cast<uchar *>(alloc_cuda(sizeof(uchar) * raw_img_h_ * raw_img_w_ * 3)));
    d_mid_data_.reset(static_cast<uchar *>(alloc_cuda(sizeof(uchar) * input_h_ * input_w_ * 3)));
}

void YoloDetectModel::getEngine() {
    if (access(trtFile_.c_str(), F_OK) == 0) {
        std::ifstream engineFile(trtFile_, std::ios::binary);
        long int      fsize = 0;

        engineFile.seekg(0, engineFile.end);
        fsize = engineFile.tellg();
        engineFile.seekg(0, engineFile.beg);
        std::vector<char> engineString(fsize);
        engineFile.read(engineString.data(), fsize);
        if (engineString.size() == 0) {
            std::cout << "Failed getting serialized engine!" << std::endl;
            return;
        }
        std::cout << "Succeeded getting serialized engine!" << std::endl;

        runtime_.reset(nvinfer1::createInferRuntime(g_logger_));
        engine_.reset(runtime_->deserializeCudaEngine(engineString.data(), fsize));
#if NV_TENSORRT_MAJOR < 10
        // Define input dimensions
        auto input_dims = engine_->getBindingDimensions(0);
        input_h_        = input_dims.d[2];
        input_w_        = input_dims.d[3];
#else
        auto input_dims = engine_->getTensorShape(engine_->getIOTensorName(0));
        input_h_        = input_dims.d[2];
        input_w_        = input_dims.d[3];
#endif
        assert(input_h_ > 0 && input_w_ > 0 &&
               "Input dimensions must be positive! Check engine binding or dynamic shape setting.");

        if (engine_ == nullptr) {
            std::cout << "Failed loading engine!" << std::endl;
            return;
        }
        std::cout << "Succeeded loading engine!" << std::endl;
    } else {
        std::cerr << "Failed loading engine file: " << trtFile_ << std::endl;
    }
}

YoloDetectModel::~YoloDetectModel() {
    context_.reset();
    engine_.reset();
    runtime_.reset();

    if (stream_ != 0) {
        cudaStreamSynchronize(stream_);
        cudaStreamDestroy(stream_);
        stream_ = 0;
    }
}

std::vector<Detection> YoloDetectModel::inference(const cv::Mat & img) {
    if (img.empty()) {
        return {};
    }

    // put input on device, then letterbox、bgr to rgb、hwc to chw、normalize.
    preprocess(img, static_cast<float *>(d_buffer_[0].get()), d_src_data_.get(), d_mid_data_.get(),
               raw_img_h_, raw_img_w_, input_h_, input_w_, stream_);
    cudaStreamSynchronize(stream_);

    // TensorRT inference - use appropriate API based on platform/TensorRT version
    void * bingding_buffers[2] = { d_buffer_[0].get(), d_buffer_[1].get() };
    bool   status              = context_->executeV2(bingding_buffers);
    if (!status) {
        std::cerr << "TensorRT enqueueV3 failed!" << std::endl;
        return {};
    }

    if (!is_need_nms_) {
        // 走yolo26推理，输出候选框较少，且已经经过nms处理，不需要再做一次nms了
        // [1 1801]
        CHECK_CUDA(
            cudaMemcpy(h_output_data_.get(), d_buffer_[1].get(),
                       (yolo26_max_num_output_bbox_ * yolo26_num_box_element_) * sizeof(float),
                       cudaMemcpyDeviceToHost));

    } else {
        // 走yolo8推理，输出候选框较多，需要做一次nms处理

        // transpose [1 84 8400] convert to [1 8400 84]
        transpose(static_cast<float *>(d_buffer_[1].get()), d_transpose_.get(), OUTPUT_CANDIDATES_,
                  numClass_ + 4, stream_);
        // convert [1 8400 84] to [1 7001]
        decode(d_transpose_.get(), d_decode_.get(), OUTPUT_CANDIDATES_, numClass_, confThresh_,
               MAX_NUM_OUTPUT_BBOX, NUM_BOX_ELEMENT, stream_);
        // cuda nms
        nms(d_decode_.get(), nmsThresh_, MAX_NUM_OUTPUT_BBOX, NUM_BOX_ELEMENT, stream_);
        cudaStreamSynchronize(stream_);

        CHECK_CUDA(cudaMemcpy(h_output_data_.get(), d_decode_.get(),
                              (1 + MAX_NUM_OUTPUT_BBOX * NUM_BOX_ELEMENT) * sizeof(float),
                              cudaMemcpyDeviceToHost));
    }

    return postProcess(img);
}

std::vector<Detection> YoloDetectModel::postProcess(const cv::Mat & img) {
    std::vector<Detection> vDetections;
    int                    count;
    if (!is_need_nms_) {
        count = std::min(yolo26_max_num_output_bbox_, MAX_NUM_OUTPUT_BBOX);
    } else {
        count = std::min((int) h_output_data_.get()[0], MAX_NUM_OUTPUT_BBOX);
    }
    for (int i = 0; i < count; i++) {
        int       pos;
        Detection det;
        auto      get_effective_detection = [&]() {
            memcpy(det.bbox.data(), &h_output_data_.get()[pos], 4 * sizeof(float));
            det.conf    = h_output_data_.get()[pos + 4];
            det.classId = (int) h_output_data_.get()[pos + 5];
            vDetections.push_back(det);
        };
        if (!is_need_nms_) {
            pos = i * yolo26_num_box_element_;
            if (h_output_data_.get()[pos + 4] > confThresh_) {
                get_effective_detection();
            }

        } else {
            pos          = 1 + i * NUM_BOX_ELEMENT;
            int keepFlag = (int) h_output_data_.get()[pos + 6];
            if (keepFlag == 1) {
                get_effective_detection();
            }
        }
    }

    for (size_t j = 0; j < vDetections.size(); j++) {
        scale_bbox(img, vDetections[j].bbox.data(), input_w_, input_h_);
    }

    return vDetections;
}

void YoloDetectModel::inferenceAsync(uchar * d_image) {
    {
        nvtx3::scoped_range tracker_scope("yolo preprocess");
        preprocess_v2(static_cast<float *>(d_buffer_[0].get()), d_image, d_mid_data_.get(),
                      raw_img_h_, raw_img_w_, input_h_, input_w_, stream_);
    }
    {
        nvtx3::scoped_range tracker_scope("yolo enqueue");
#if defined(__aarch64__) || defined(__arm__) || NV_TENSORRT_MAJOR < 10
        // For Jetson Nano (ARM64) and older TensorRT versions
        void * bindings[] = { d_buffer_[0].get(), d_buffer_[1].get() };
        bool   status     = context_->enqueueV2(bindings, stream_, nullptr);
#else
        // For newer TensorRT versions on x86_64
        context_->setTensorAddress("images", d_buffer_[0].get());
        context_->setTensorAddress("output0", d_buffer_[1].get());
        bool status = context_->enqueueV3(stream_);
#endif

        if (!status) {
            std::cerr << "TensorRT enqueue failed!" << std::endl;
            return;
        }
    }

    if (!is_need_nms_) {
        // 走yolo26推理，输出候选框较少，且已经经过nms处理，不需要再做一次nms了
        // [1 1801]
        CHECK_CUDA(
            cudaMemcpyAsync(h_output_data_.get(), d_buffer_[1].get(),
                            (yolo26_max_num_output_bbox_ * yolo26_num_box_element_) * sizeof(float),
                            cudaMemcpyDeviceToHost, stream_));
    } else {
        nvtx3::scoped_range tracker_scope("yolo nms");
        // 走yolo8推理，输出候选框较多，需要做一次nms处
        // transpose [1 84 8400] convert to [1 8400 84]
        transpose(static_cast<float *>(d_buffer_[1].get()), d_transpose_.get(), OUTPUT_CANDIDATES_,
                  numClass_ + 4, stream_);
        // convert [1 8400 84] to [1 7001]
        decode(d_transpose_.get(), d_decode_.get(), OUTPUT_CANDIDATES_, numClass_, confThresh_,
               MAX_NUM_OUTPUT_BBOX, NUM_BOX_ELEMENT, stream_);
        // cuda nms
        nms(d_decode_.get(), nmsThresh_, MAX_NUM_OUTPUT_BBOX, NUM_BOX_ELEMENT, stream_);

        CHECK_CUDA(cudaMemcpyAsync(h_output_data_.get(), d_decode_.get(),
                                   h_output_data_size_ * sizeof(float), cudaMemcpyDeviceToHost,
                                   stream_));
    }
}

void YoloDetectModel::waitAsync() {
    cudaStreamSynchronize(stream_);
}

std::vector<Detection> YoloDetectModel::getInferResultAsync(const cv::Mat & img) {
    return postProcess(img);
}
