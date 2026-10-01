#include "backend_factory.h"

#include "logger_manager.h"

#ifdef HAS_CUDA
#    include "tensorrt/cuda_utils.h"
#endif

#ifdef USE_TENSORRT
#    include "tensorrt/tensorrt_backend.h"
#endif
#ifdef USE_ONNXRUNTIME
#    include "onnx/onnxruntime_backend.h"
#endif
#ifdef ENABLE_QNN
#    include "qnn_backend.h"  // QNN 后端落地时提供（USE_QNN 开关 + QNN SDK）
#endif

std::string backendPathKeyForBackend(const std::string & backend_type) {
    if (backend_type == "tensorrt") {
        return "engine";
    }
    if (backend_type == "qnn") {
        return "qnn";
    }
    return "onnx";  // onnxruntime 及未知类型统一回落到 onnx 条目
}

std::unique_ptr<InferenceBackend> createBackendByType(const std::string & backend_type,
                                                      int                 gpu_id) {
#ifdef USE_TENSORRT
    if (backend_type == "tensorrt") {
        return std::make_unique<TensorRTBackend>(gpu_id);
    }
#endif
#ifdef USE_ONNXRUNTIME
    if (backend_type == "onnxruntime") {
        return std::make_unique<OnnxRuntimeBackend>();
    }
#endif
#ifdef ENABLE_QNN
    if (backend_type == "qnn") {
        return std::make_unique<QnnBackend>();
    }
#endif
    APP_ERROR("Backend type '{}' unknown or not compiled in this build", backend_type);
    return nullptr;
}

std::vector<std::string> backendCandidateChain(
    const std::map<std::string, std::string> & model_paths,
    bool                                       use_gpu,
    const std::string &                        preferred) {
    // 显式指定：只信配置，不做静默回退
    if (!preferred.empty() && preferred != "auto") {
        return { preferred };
    }

    // auto：保持历史回退链—：GPU 可用 + 偏好 GPU + 配置了 engine -> TensorRT 优先
    std::vector<std::string> candidates;
#ifdef HAS_CUDA
    if (use_gpu) {
        int         device_count = 0;
        cudaError_t error        = cudaGetDeviceCount(&device_count);
        if (error == cudaSuccess && device_count > 0 &&
            model_paths.find("engine") != model_paths.end()) {
            candidates.push_back("tensorrt");
        }
    }
#endif
    if (model_paths.find("onnx") != model_paths.end()) {
        candidates.push_back("onnxruntime");
    }
    return candidates;
}
