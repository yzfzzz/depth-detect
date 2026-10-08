#include "tensorrt/tensorrt_backend.h"

#include "logger_manager.h"

#include <cstdint>
#include <fstream>
#include <iostream>

namespace inference {

TensorRTBackend::TensorRTBackend(int gpu_id) : gpu_id_(gpu_id) {
    CHECK_CUDA(cudaSetDevice(gpu_id_));
}

TensorRTBackend::~TensorRTBackend() {
#ifdef HAS_CUDA_GRAPH
    // 模型析构时 BaseModel 已经同步并销毁了流，这里只释放图句柄
    if (graph_exec_ != nullptr) {
        cudaGraphExecDestroy(graph_exec_);
        graph_exec_ = nullptr;
    }
#endif
}

bool TensorRTBackend::isAvailable() const {
    int         device_count = 0;
    cudaError_t error        = cudaGetDeviceCount(&device_count);
    return error == cudaSuccess && device_count > 0;
}

bool TensorRTBackend::loadModel(const std::string &        model_path,
                                const BackendInitOptions & options) {
    use_cuda_graph_ = options.cuda_graph_enabled;
    if (options.cuda_graph_warmup_frames < 1) {
        APP_WARN(
            "cuda_graph_warmup_frames={} is invalid: TensorRT requires at least one enqueue before "
            "graph capture; clamped to 1",
            options.cuda_graph_warmup_frames);
        graph_warmup_frames_ = 1;
    } else {
        graph_warmup_frames_ = options.cuda_graph_warmup_frames;
    }
#ifndef HAS_CUDA_GRAPH
    if (use_cuda_graph_) {
        APP_WARN(
            "CUDA graph requested but this build has no CUDA graph support (needs CUDA >= 10.0); "
            "falling back to per-op enqueue");
        use_cuda_graph_ = false;
    }
#endif

    if (!isAvailable()) {
        APP_ERROR("TensorRT backend not available: No GPU detected");
        return false;
    }

    if (!loadEngine(model_path)) {
        return false;
    }
    // GPU 环境信息随加载一起打印，避免上层调用方依赖 TRT 特有方法
    getCudaDeviceInfo();
    if (use_cuda_graph_) {
        APP_INFO("CUDA graph enabled for engine [{}]: capture after {} warmup inference(s)",
                 model_path, graph_warmup_frames_);
    }
    return true;
}

bool TensorRTBackend::loadEngine(const std::string & engine_path) {
    // 检查文件是否存在
    std::ifstream engineFile(engine_path, std::ios::binary);
    if (!engineFile.good()) {
        APP_ERROR("Cannot open engine file: {}", engine_path);
        return false;
    }

    // 读取引擎文件
    engineFile.seekg(0, std::ios::end);
    size_t fsize = engineFile.tellg();
    engineFile.seekg(0, std::ios::beg);
    std::vector<char> engineData(fsize);
    engineFile.read(engineData.data(), fsize);
    engineFile.close();

    if (engineData.empty()) {
        APP_ERROR("Engine file is empty: {}", engine_path);
        return false;
    }

    // 创建运行时和引擎
    runtime_.reset(nvinfer1::createInferRuntime(logger_));
    if (!runtime_) {
        APP_ERROR("Failed to create TensorRT runtime");
        return false;
    }

    engine_.reset(runtime_->deserializeCudaEngine(engineData.data(), fsize));
    if (!engine_) {
        APP_ERROR("Failed to deserialize TensorRT engine");
        return false;
    }

    // 创建执行上下文
    context_.reset(engine_->createExecutionContext());
    if (!context_) {
        APP_ERROR("Failed to create TensorRT execution context");
        return false;
    }

    // 设置输入输出维度
    setupInputOutputDims();

    APP_INFO("TensorRT engine loaded successfully from: {}", engine_path);
    APP_INFO("Input dims: [{}, {}, {}, {}], Input byte size: {} bytes", input_dims_[0],
             input_dims_[1], input_dims_[2], input_dims_[3], input_byte_size_);
    APP_INFO("Output tensor count: {}", output_tensor_.size());
    for (int i = 0; i < output_tensor_.size(); i++) {
        std::string dims_str;
        for (size_t j = 0; j < output_tensor_[i].dims.size(); ++j) {
            if (j > 0) {
                dims_str += ", ";
            }
            dims_str += std::to_string(output_tensor_[i].dims[j]);
        }
        APP_INFO("[{}] Output dim: ['{}']: dims: [{}], byte size: {} bytes", i,
                 output_tensor_[i].name, dims_str, output_tensor_[i].byte_size);
    }

    return true;
}

void TensorRTBackend::setupInputOutputDims() {
#if NV_TENSORRT_MAJOR < 10
    // TRT 8.x 及以下版本
    int nb_bindings    = engine_->getNbBindings();
    input_tensor_name_ = engine_->getBindingName(0);
    num_outputs_       = nb_bindings - 1;  // 第0个是输入，其余是输出

    output_tensor_.resize(num_outputs_);

    for (int i = 0; i < num_outputs_; ++i) {
        output_tensor_[i].name = engine_->getBindingName(i + 1);
        auto od                = engine_->getBindingDimensions(i + 1);
        output_tensor_[i].dims.clear();
        for (int j = 0; j < od.nbDims; ++j) {
            output_tensor_[i].dims.push_back(od.d[j]);
        }
    }

    auto input_dims = engine_->getBindingDimensions(0);
#else
    // TRT 10.x
    int nb_bindings    = engine_->getNbIOTensors();
    input_tensor_name_ = engine_->getIOTensorName(0);
    num_outputs_       = nb_bindings - 1;

    output_tensor_.resize(num_outputs_);
    for (int i = 0; i < num_outputs_; ++i) {
        output_tensor_[i].name = engine_->getIOTensorName(i + 1);
    }

    for (int i = 0; i < num_outputs_; ++i) {
        auto od = engine_->getTensorShape(output_tensor_[i].name.c_str());
        output_tensor_[i].dims.clear();
        for (int j = 0; j < od.nbDims; ++j) {
            output_tensor_[i].dims.push_back(od.d[j]);
        }
    }
    auto input_dims = engine_->getTensorShape(input_tensor_name_.c_str());
#endif
    // 保存输入维度
    input_dims_.clear();
    for (int i = 0; i < input_dims.nbDims; ++i) {
        input_dims_.push_back(input_dims.d[i]);
    }

    // 计算数据大小
    input_byte_size_ = 1;
    for (int dim : input_dims_) {
        input_byte_size_ *= dim;
    }
    input_byte_size_ *= sizeof(float);

    for (int i = 0; i < output_tensor_.size(); i++) {
        size_t byte_size = 1;
        for (int dim : output_tensor_[i].dims) {
            byte_size *= dim;
        }
        output_tensor_[i].byte_size = byte_size * sizeof(float);
    }

    // 设置输入维度（动态形状）
#if NV_TENSORRT_MAJOR < 10
    context_->setBindingDimensions(0, input_dims);
#else
    context_->setInputShape(input_tensor_name_.c_str(), input_dims);
#endif
}

bool TensorRTBackend::runInference(void * input_data, std::vector<void *> output_data) {
    if (!context_) {
        APP_ERROR("TensorRT context not initialized");
        return false;
    }
    if (output_tensor_.size() != output_data.size()) {
        APP_ERROR("output_data size mismatch: expected {}, got {}", output_tensor_.size(),
                  output_data.size());
        return false;
    }
    std::vector<void *> buffers(output_data.size() + 1);
    buffers[0] = input_data;
    for (size_t i = 0; i < output_data.size(); i++) {
        buffers[i + 1] = output_data[i];
    }
    bool status = context_->executeV2(buffers.data());
    if (!status) {
        APP_ERROR("TensorRT inference failed");
        return false;
    }
    return true;
}

// 真异步：TRT 10.x 用 enqueueV3、TRT 8.x 用 enqueueV2，都是提交即返回，
// 输出就绪顺序由 stream 保证，因此这里不做降级（能力查询 asyncUnsupportedReason() 返回空串）。
// stream_handle 是基类接口透传的原生流句柄，CUDA 后端按 cudaStream_t 解释。
// 开启 CUDA Graph 后：预热若干次再把提交序列捕获成一张图，此后每次调用只发一次 cudaGraphLaunch；捕获/发射失败自动回退逐算子提交
bool TensorRTBackend::runInferenceAsync(void *              input_data,
                                        std::vector<void *> output_data,
                                        void *              stream_handle) {
    if (!context_) {
        APP_ERROR("TensorRT context not initialized");
        return false;
    }
    if (output_tensor_.size() != output_data.size()) {
        APP_ERROR("output_data size mismatch: expected {}, got {}", output_tensor_.size(),
                  output_data.size());
        return false;
    }
    cudaStream_t stream = static_cast<cudaStream_t>(stream_handle);
#ifdef HAS_CUDA_GRAPH
    if (use_cuda_graph_ && !graph_disabled_) {
        // 绑定地址在捕获后发生变化（缓冲重建）：图里记的是旧地址，直接停用图
        if (graph_exec_ != nullptr && !bindingsMatch(input_data, output_data)) {
            APP_WARN("TensorRT bindings changed after CUDA graph capture; disabling CUDA graph");
            graph_disabled_ = true;
            destroyCudaGraph(stream);
        }
        if (!graph_disabled_) {
            if (graph_exec_ == nullptr) {
                // 预热：让 TRT 内部资源分配、CUDA 首次 module load 等一次性初始化先发生，
                // 否则这些操作落在捕获窗口内会让捕获失败
                if (graph_warmup_count_ < graph_warmup_frames_) {
                    ++graph_warmup_count_;
                } else if (!captureCudaGraph(stream, input_data, output_data)) {
                    graph_disabled_ = true;  // 只告警一次，之后永久逐算子提交
                }
            }
            if (graph_exec_ != nullptr) {
                cudaError_t status = cudaGraphLaunch(graph_exec_, stream);
                if (status == cudaSuccess) {
                    return true;
                }
                APP_WARN("cudaGraphLaunch failed: {}; falling back to per-op enqueue",
                         cudaGetErrorString(status));
                graph_disabled_ = true;
                destroyCudaGraph(stream);
            }
        }
    }
#endif
    return enqueue(stream, input_data, output_data);
}

// 逐算子提交
bool TensorRTBackend::enqueue(cudaStream_t                stream,
                              void *                      input_data,
                              const std::vector<void *> & output_data) {
#if NV_TENSORRT_MAJOR >= 10
    // TRT 10.x: Explicitly set tensor addresses before enqueueV3
    // 捕获窗口内只记录设备侧提交
    cudaStreamCaptureStatus capture_status = cudaStreamCaptureStatusNone;
    const bool capturing = cudaStreamIsCapturing(stream, &capture_status) == cudaSuccess &&
                           capture_status != cudaStreamCaptureStatusNone;
    if (!capturing) {
        context_->setTensorAddress(input_tensor_name_.c_str(), input_data);
        for (size_t i = 0; i < output_data.size(); i++) {
            context_->setTensorAddress(output_tensor_[i].name.c_str(), output_data[i]);
        }
    }
    bool status = context_->enqueueV3(stream);
#else
    // TRT 8.x: Uses buffer array with enqueueV2
    std::vector<void *> buffers(output_data.size() + 1);
    buffers[0] = input_data;
    for (size_t i = 0; i < output_data.size(); i++) {
        buffers[i + 1] = output_data[i];
    }
    bool status = context_->enqueueV2(buffers.data(), stream, nullptr);
#endif
    if (!status) {
        APP_ERROR("TensorRT async inference failed");
        return false;
    }
    return true;
}

#ifdef HAS_CUDA_GRAPH
// 捕获一次完整提交为 CUDA Graph。捕获不执行算子：本次推理的实际计算由调用方在捕获后立刻
// cudaGraphLaunch 完成；任何一步失败都返回 false，由调用方回退逐算子提交
bool TensorRTBackend::captureCudaGraph(cudaStream_t                stream,
                                       void *                      input_data,
                                       const std::vector<void *> & output_data) {
    // 捕获前把流清空：一次性 host 同步，位于捕获窗口之外
    cudaStreamSynchronize(stream);

    cudaError_t status = cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal);
    if (status != cudaSuccess) {
        APP_WARN("cudaStreamBeginCapture failed: {}; falling back to per-op enqueue",
                 cudaGetErrorString(status));
        return false;
    }

    enqueue(stream, input_data, output_data);

    cudaGraph_t graph = nullptr;
    status            = cudaStreamEndCapture(stream, &graph);
    if (status != cudaSuccess) {
        APP_WARN("CUDA graph capture failed: {}; falling back to per-op enqueue",
                 cudaGetErrorString(status));
        if (graph != nullptr) {
            cudaGraphDestroy(graph);
        }
        // 捕获可能仍处于进行态：查询并终止，避免后续提交被记进一个永远不会执行的"幽灵捕获"
        cudaStreamCaptureStatus capture_status = cudaStreamCaptureStatusNone;
        if (cudaStreamIsCapturing(stream, &capture_status) == cudaSuccess &&
            capture_status != cudaStreamCaptureStatusNone) {
            cudaGraph_t discarded = nullptr;
            cudaStreamEndCapture(stream, &discarded);
            if (discarded != nullptr) {
                cudaGraphDestroy(discarded);
            }
        }
        return false;
    }

    // 节点数按类型统计，日志里能直接看出图上有没有推理 kernel
    size_t node_total = 0, node_kernel = 0, node_memcpy = 0, node_memset = 0;
    cudaGraphGetNodes(graph, nullptr, &node_total);
    if (node_total > 0) {
        std::vector<cudaGraphNode_t> nodes(node_total);
        size_t                       fetched = node_total;
        if (cudaGraphGetNodes(graph, nodes.data(), &fetched) == cudaSuccess) {
            for (size_t i = 0; i < fetched; ++i) {
                cudaGraphNodeType type;
                if (cudaGraphNodeGetType(nodes[i], &type) != cudaSuccess) {
                    continue;
                }
                switch (type) {
                    case cudaGraphNodeTypeKernel:
                        ++node_kernel;
                        break;
                    case cudaGraphNodeTypeMemcpy:
                        ++node_memcpy;
                        break;
                    case cudaGraphNodeTypeMemset:
                        ++node_memset;
                        break;
                    default:
                        break;
                }
            }
        }
    }

    cudaGraphExec_t graph_exec = nullptr;
#    if CUDART_VERSION >= 12000
    status = cudaGraphInstantiate(&graph_exec, graph, 0);
#    else
    status = cudaGraphInstantiate(&graph_exec, graph, nullptr, nullptr, 0);
#    endif
    cudaGraphDestroy(graph);
    if (status != cudaSuccess) {
        APP_WARN("cudaGraphInstantiate failed: {}; falling back to per-op enqueue",
                 cudaGetErrorString(status));
        return false;
    }

    graph_exec_ = graph_exec;
    graph_bindings_.clear();
    graph_bindings_.reserve(output_data.size() + 1);
    graph_bindings_.push_back(input_data);
    graph_bindings_.insert(graph_bindings_.end(), output_data.begin(), output_data.end());
    APP_INFO("CUDA graph captured: {} nodes (kernel={}, memcpy={}, memset={})", node_total,
             node_kernel, node_memcpy, node_memset);
    return true;
}

void TensorRTBackend::destroyCudaGraph(cudaStream_t stream) {
    if (graph_exec_ != nullptr) {
        cudaStreamSynchronize(stream);  // 等流：避免释放仍在执行的图
        cudaGraphExecDestroy(graph_exec_);
        graph_exec_ = nullptr;
    }
    graph_bindings_.clear();
}

bool TensorRTBackend::bindingsMatch(void *                      input_data,
                                    const std::vector<void *> & output_data) const {
    if (graph_bindings_.size() != output_data.size() + 1 || graph_bindings_[0] != input_data) {
        return false;
    }
    for (size_t i = 0; i < output_data.size(); ++i) {
        if (graph_bindings_[i + 1] != output_data[i]) {
            return false;
        }
    }
    return true;
}
#endif  // HAS_CUDA_GRAPH

std::vector<int> TensorRTBackend::getInputDims() const {
    return input_dims_;
}

std::vector<int64_t> TensorRTBackend::getOutputDims(int output_index) const {
    if (output_index < 0 || output_index >= static_cast<int>(output_tensor_.size())) {
        APP_ERROR("Invalid output index: {}", output_index);
        return {};
    }
    return output_tensor_[output_index].dims;
}

size_t TensorRTBackend::getInputByteSize() const {
    return input_byte_size_;
}

size_t TensorRTBackend::getOutputByteSize(int output_index) const {
    if (output_index < 0 || output_index >= static_cast<int>(output_tensor_.size())) {
        APP_ERROR("Invalid output index: {}", output_index);
        return 0;
    }
    return output_tensor_[output_index].byte_size;
}

void TensorRTBackend::getCudaDeviceInfo() {
    int         device_count = 0;
    cudaError_t error        = cudaGetDeviceCount(&device_count);
    if (error != cudaSuccess || device_count <= 0) {
        APP_ERROR("No CUDA-capable devices found");
        return;
    }

    if (gpu_id_ < 0 || gpu_id_ >= device_count) {
        APP_ERROR("Invalid GPU ID: {}. Available devices: 0 to {}", gpu_id_, device_count - 1);
        return;
    }

    cudaDeviceProp device_prop;
    error = cudaGetDeviceProperties(&device_prop, gpu_id_);
    if (error != cudaSuccess) {
        APP_ERROR("Failed to get properties for GPU ID {}: {}", gpu_id_, cudaGetErrorString(error));
        return;
    }

    APP_INFO(
        "Using GPU ID {}: {}, Compute Capability: {}.{}, {} SMs, {} threads per SM, {} threads per "
        "block, {} registers per SM",
        gpu_id_, device_prop.name, device_prop.major, device_prop.minor,
        device_prop.multiProcessorCount, device_prop.maxThreadsPerMultiProcessor,
        device_prop.maxThreadsPerBlock, device_prop.regsPerMultiprocessor);
}

}  // namespace inference
