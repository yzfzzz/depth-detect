#include "tensorrt_backend.h"

#include "public.h"

#include <fstream>
#include <iostream>

TensorRTBackend::TensorRTBackend(int gpu_id) : gpu_id_(gpu_id) {
    CHECK_CUDA(cudaSetDevice(gpu_id_));
}

TensorRTBackend::~TensorRTBackend() {}

bool TensorRTBackend::isAvailable() const {
    int         device_count = 0;
    cudaError_t error        = cudaGetDeviceCount(&device_count);
    return error == cudaSuccess && device_count > 0;
}

bool TensorRTBackend::loadModel(const std::string & model_path) {
    if (!isAvailable()) {
        APP_ERROR("TensorRT backend not available: No GPU detected");
        return false;
    }

    return loadEngine(model_path);
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
    APP_INFO("Input dims: [{}, {}, {}, {}]", input_dims_[0], input_dims_[1], input_dims_[2],
             input_dims_[3]);
    APP_INFO("Output dims: [{}]", output_dims_.size());

    return true;
}

void TensorRTBackend::setupInputOutputDims() {
#if NV_TENSORRT_MAJOR < 10
    // TensorRT 8.x API
    int nb_bindings     = engine_->getNbBindings();
    input_tensor_name_  = engine_->getBindingName(0);
    output_tensor_name_ = engine_->getBindingName(1);

    auto input_dims  = engine_->getBindingDimensions(0);
    auto output_dims = engine_->getBindingDimensions(1);
#else
    // TensorRT 10.x API
    int nb_bindings     = engine_->getNbIOTensors();
    input_tensor_name_  = engine_->getIOTensorName(0);
    output_tensor_name_ = engine_->getIOTensorName(1);

    auto input_dims  = engine_->getTensorShape(input_tensor_name_.c_str());
    auto output_dims = engine_->getTensorShape(output_tensor_name_.c_str());
#endif

    // 保存输入维度
    input_dims_.clear();
    for (int i = 0; i < input_dims.nbDims; ++i) {
        input_dims_.push_back(input_dims.d[i]);
    }

    // 保存输出维度
    output_dims_.clear();
    for (int i = 0; i < output_dims.nbDims; ++i) {
        output_dims_.push_back(output_dims.d[i]);
    }

    // 计算数据大小
    input_byte_size_ = 1;
    for (int dim : input_dims_) {
        input_byte_size_ *= dim;
    }
    input_byte_size_ *= sizeof(float);

    output_byte_size_ = 1;
    for (int dim : output_dims_) {
        output_byte_size_ *= dim;
    }
    output_byte_size_ *= sizeof(float);

    // 设置输入维度（动态形状）
#if NV_TENSORRT_MAJOR < 10
    context_->setBindingDimensions(0, input_dims);
#else
    context_->setInputShape(input_tensor_name_.c_str(), input_dims);
#endif
    APP_INFO("TensorRT input size: {} bytes, output size: {} bytes", input_byte_size_,
             output_byte_size_);
}

bool TensorRTBackend::runInference(void * input_data, void * output_data) {
    if (!context_) {
        APP_ERROR("TensorRT context not initialized");
        return false;
    }

    void * buffers[2] = { input_data, output_data };
    bool   status     = context_->executeV2(buffers);

    if (!status) {
        APP_ERROR("TensorRT inference failed");
        return false;
    }

    return true;
}

bool TensorRTBackend::runInferenceAsync(void *       input_data,
                                        void *       output_data,
                                        cudaStream_t stream) {
    if (!context_) {
        APP_ERROR("TensorRT context not initialized");
        return false;
    }

#if NV_TENSORRT_MAJOR >= 10
    // TRT 10.x: Explicitly set tensor addresses before enqueueV3
    context_->setTensorAddress(input_tensor_name_.c_str(), input_data);
    context_->setTensorAddress(output_tensor_name_.c_str(), output_data);
    bool status = context_->enqueueV3(stream);
#else
    // TRT 8.x: Uses buffer array with enqueueV2
    void * buffers[2] = { input_data, output_data };
    bool   status     = context_->enqueueV2(buffers, stream, nullptr);
#endif

    if (!status) {
        APP_ERROR("TensorRT async inference failed");
        return false;
    }

    return true;
}

std::vector<int> TensorRTBackend::getInputDims() const {
    return input_dims_;
}

std::vector<int> TensorRTBackend::getOutputDims() const {
    return output_dims_;
}

size_t TensorRTBackend::getInputByteSize() const {
    return input_byte_size_;
}

size_t TensorRTBackend::getOutputByteSize() const {
    return output_byte_size_;
}
