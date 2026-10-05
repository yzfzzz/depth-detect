#include "base_model.h"

#include "backend_factory.h"
#include "logger_manager.h"

#include <fstream>

BaseModel::~BaseModel() {
#ifdef HAS_CUDA
    if (stream_ != 0) {
        CHECK_CUDA(cudaStreamSynchronize(stream_));
        CHECK_CUDA(cudaStreamDestroy(stream_));
    }
#endif
}

void BaseModel::init(std::map<std::string, std::string> model_path,
                     int                                raw_img_w,
                     int                                raw_img_h,
                     bool                               use_gpu,
                     const std::string &                preferred_backend) {
    initialized_ = false;
#ifdef HAS_CUDA
    stream_ = 0;
#endif
    if (raw_img_h <= 0 || raw_img_w <= 0) {
        APP_ERROR("Invalid image dimensions: {}x{}", raw_img_w, raw_img_h);
        return;
    }
    raw_img_w_         = raw_img_w;
    raw_img_h_         = raw_img_h;
    preferred_backend_ = preferred_backend;
    bool status        = initInferenceBackend(model_path, use_gpu, preferred_backend);
    if (!status) {
        APP_ERROR("Failed to initialize inference backend");
        return;
    }
    // 从后端获取模型要求的输入尺寸（如 640x640），用于后续预处理 resize
    auto input_dims  = getInputDims();
    auto output_dims = getOutputDims();
    input_h_         = input_dims[2];
    input_w_         = input_dims[3];
}

std::unique_ptr<InferenceBackend> BaseModel::createBackend(
    std::map<std::string, std::string> model_path,
    bool                               use_gpu) {
    // 后端选择收拢在 backend_factory：显式配置只走指定后端；
    // auto 保持历史回退链「GPU 可用且配置了 .engine -> TensorRT 优先，.onnx 兜底」
    const std::vector<std::string> candidates =
        backendCandidateChain(model_path, use_gpu, preferred_backend_);
    if (candidates.empty()) {
        APP_ERROR("No candidate backend resolved from config (use_gpu={}, preferred={})", use_gpu,
                  preferred_backend_);
        return nullptr;
    }

    for (const auto & type : candidates) {
        const std::string path_key = backendPathKeyForBackend(type);
        auto              it       = model_path.find(path_key);
        if (it == model_path.end()) {
            APP_WARN("Model path key '{}' missing for backend '{}'", path_key, type);
            continue;
        }
        auto backend = createBackendByType(type, 0);
        if (!backend) {
            continue;  // 该后端未参与本次编译，尝试下一个候选
        }
        if (!backend->loadModel(it->second)) {
            APP_WARN("Backend '{}' failed to load model from {}, trying next candidate", type,
                     it->second);
            continue;
        }
        APP_INFO("Backend [{}] initialized successfully", backend->getBackendTypeName());
        // CUDA 路径的模型需要与后端共享同一条流：预处理 kernel、推理提交、
        // 后处理 kernel / D2H 拷贝全部在该流上按提交顺序串行化
#ifdef HAS_CUDA
        if (backend->getTensorLocation() == TensorLocation::CudaDevice) {
            CHECK_CUDA(cudaSetDevice(0));
            CHECK_CUDA(cudaStreamCreate(&stream_));
            APP_INFO("CUDA stream created successfully");
        }
#endif
        return backend;
    }
    return nullptr;
}

bool BaseModel::initInferenceBackend(std::map<std::string, std::string> model_path,
                                     bool                               use_gpu,
                                     const std::string &                preferred_backend) {
    if (initialized_) {
        APP_WARN("Model already initialized");
        return true;
    }
    preferred_backend_ = preferred_backend;

    // 检查模型文件是否存在 engine/onnx 是「优选 + 兜底」双臂，实际选哪个由 createBackend
    // 决定，因此单个文件缺失只警告、不致命，否则缺失的兜底文件会让整个后端初始化失败
    bool any_readable = false;
    for (auto & item : model_path) {
        const auto &  type = item.first;
        const auto &  path = item.second;
        std::ifstream file(path);
        if (!file.good()) {
            APP_WARN("Model file not found: type={}, path: {}", type, path);
            continue;
        }
        file.close();
        any_readable = true;
    }
    if (!any_readable) {
        APP_ERROR("No readable model file in config, failed to initialize any backend");
        return false;
    }

    // 创建后端
    backend_ = createBackend(model_path, use_gpu);
    if (!backend_) {
        APP_ERROR("Failed to create inference backend");
        return false;
    }

    initialized_ = true;
    APP_INFO("Model initialized successfully with backend: {}", backend_->getBackendTypeName());
    return true;
}

// 执行异步推理（供子类调用）
bool BaseModel::runInferenceAsync(FrameInputContext & frame_input_context) {
    if (!isBackendInitialized()) {
        APP_ERROR("Model not initialized");
        return false;
    }
    // 按后端的 capability 查询决定
    const std::string async_reason = backend_->asyncUnsupportedReason();
    if (!async_reason.empty()) {
        APP_ERROR("Async inference unavailable: backend [{}], reason: {}; use runInference instead",
                  backendTypeName(), async_reason);
        return false;
    }
#ifdef HAS_CUDA
    // 真异步路径（当前只有 TensorRT）：预处理→推理→后处理全在 GPU Stream 上排队，CPU 不等待
    cudaPreProcess(frame_input_context);
    std::vector<void *> output_buffers;
    output_buffers.reserve(d_infer_io_.size() - 1);
    std::transform(d_infer_io_.begin() + 1, d_infer_io_.end(), std::back_inserter(output_buffers),
                   [](auto & ptr) { return ptr.get(); });
    backend_->runInferenceAsync(d_infer_io_[0].get(), output_buffers, stream_);
    // 异步后处理
    cudaPostProcess(frame_input_context);
    return true;
#else
    APP_ERROR("Async inference unavailable: this build has no CUDA backend");
    return false;
#endif
}

// 执行同步推理（供子类调用）
// 执行路径按后端的 TensorLocation 分派，不按后端类型分派：
// Host 路径（ONNX Runtime / 未来的 QNN）走 OpenCV 前后处理 + 主机缓冲；
// CudaDevice 路径（TensorRT）走 CUDA 前后处理 + 设备缓冲
bool BaseModel::runInference(FrameInputContext &  frame_input_context,
                             InferOutputContext & infer_output_context) {
    if (!isBackendInitialized()) {
        APP_ERROR("Model not initialized");
        return false;
    }
    std::vector<void *> output_buffers;
    output_buffers.reserve(getNumOutputs());
    if (backend_->getTensorLocation() == TensorLocation::Host) {
        std::vector<float> host_input_tensor = cvMatPreProcess(frame_input_context);
        std::transform(h_infer_out_.begin(), h_infer_out_.end(), std::back_inserter(output_buffers),
                       [](auto & v) { return v.data(); });
        backend_->runInference(host_input_tensor.data(), output_buffers);
        cvMatPostProcess(infer_output_context);
        return true;
    }
#ifdef HAS_CUDA
    else if (backend_->getTensorLocation() == TensorLocation::CudaDevice) {
        // CUDA 路径：异步预处理 → 等待完成 → 同步推理 → 异步后处理 → 取结果
        cudaPreProcess(frame_input_context);
        synchronizeStream();  // 确保预处理数据就绪后再推理
        std::transform(d_infer_io_.begin() + 1, d_infer_io_.end(),
                       std::back_inserter(output_buffers), [](auto & ptr) { return ptr.get(); });
        backend_->runInference(d_infer_io_[0].get(), output_buffers);
        // 异步后处理
        cudaPostProcess(frame_input_context);
        getInferOutputResult(infer_output_context);
        return true;
    }
#endif
    APP_ERROR("Unsupported tensor location for backend: {}", backendTypeName());
    return false;
}
