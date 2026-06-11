#include "onnxruntime_backend.h"

#include "public.h"

#include <cstring>
#include <fstream>

OnnxRuntimeBackend::OnnxRuntimeBackend() :
    session_(nullptr),
    input_byte_size_(0),
    output_byte_size_(0) {}

OnnxRuntimeBackend::~OnnxRuntimeBackend() = default;

bool OnnxRuntimeBackend::isAvailable() const {
    // ONNX Runtime CPU 后端总是可用
    return true;
}

bool OnnxRuntimeBackend::loadModel(const std::string & model_path) {
    // 检查文件是否存在
    std::ifstream file(model_path);
    if (!file.good()) {
        APP_ERROR("Cannot open ONNX model file: {}", model_path);
        return false;
    }
    file.close();

    try {
        // 创建 ONNX Runtime 环境
        env_ = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "OnnxRuntimeBackend");

        // 配置 Session 选项
        session_options_ = std::make_unique<Ort::SessionOptions>();
        session_options_->SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
        session_options_->SetIntraOpNumThreads(4);

        // 加载模型
        session_ = Ort::Session(*env_, model_path.c_str(), *session_options_);

        // 创建内存信息（CPU）
        memory_info_ = std::make_unique<Ort::MemoryInfo>(
            Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault));

        // 获取输入信息
        Ort::AllocatorWithDefaultOptions allocator;
        char *                           raw_input_name = session_.GetInputName(0, allocator);
        input_name_                                     = raw_input_name;
        allocator.Free(raw_input_name);

        auto input_type_info   = session_.GetInputTypeInfo(0);
        auto input_tensor_info = input_type_info.GetTensorTypeAndShapeInfo();
        input_dims_            = input_tensor_info.GetShape();

        // 处理动态维度（如 YOLOv8 中的 -1），设为 1
        for (auto & dim : input_dims_) {
            if (dim <= 0) {
                dim = 1;
            }
        }

        // 获取输出信息
        char * raw_output_name = session_.GetOutputName(0, allocator);
        output_name_           = raw_output_name;
        allocator.Free(raw_output_name);

        auto output_type_info   = session_.GetOutputTypeInfo(0);
        auto output_tensor_info = output_type_info.GetTensorTypeAndShapeInfo();
        output_dims_            = output_tensor_info.GetShape();

        // 处理动态维度
        for (auto & dim : output_dims_) {
            if (dim <= 0) {
                dim = 1;
            }
        }

        // 计算输入/输出数据大小
        input_byte_size_ = 1;
        for (auto dim : input_dims_) {
            input_byte_size_ *= dim;
        }
        input_byte_size_ *= sizeof(float);

        output_byte_size_ = 1;
        for (auto dim : output_dims_) {
            output_byte_size_ *= dim;
        }
        output_byte_size_ *= sizeof(float);

        APP_INFO("ONNX Runtime model loaded successfully from: {}", model_path);
        {
            std::string dims_str;
            for (size_t i = 0; i < input_dims_.size(); ++i) {
                if (i > 0) {
                    dims_str += ", ";
                }
                dims_str += std::to_string(input_dims_[i]);
            }
            APP_INFO("Input dims: [{}]", dims_str);
        }
        {
            std::string dims_str;
            for (size_t i = 0; i < output_dims_.size(); ++i) {
                if (i > 0) {
                    dims_str += ", ";
                }
                dims_str += std::to_string(output_dims_[i]);
            }
            APP_INFO("Output dims: [{}]", dims_str);
        }
        APP_INFO("Input size: {} bytes, output size: {} bytes", input_byte_size_,
                 output_byte_size_);

        return true;
    } catch (const Ort::Exception & e) {
        APP_ERROR("ONNX Runtime error: {}", e.what());
        return false;
    }
}

bool OnnxRuntimeBackend::runInference(void * input_data, void * output_data) {
    if (!session_) {
        APP_ERROR("ONNX Runtime session not initialized");
        return false;
    }

    try {
        // 计算输入元素数量
        size_t input_count = 1;
        for (auto dim : input_dims_) {
            input_count *= dim;
        }

        // 创建输入 Tensor
        auto input_tensor =
            Ort::Value::CreateTensor<float>(*memory_info_, static_cast<float *>(input_data),
                                            input_count, input_dims_.data(), input_dims_.size());

        const char * input_names[]  = { input_name_.c_str() };
        const char * output_names[] = { output_name_.c_str() };

        auto output_tensors = session_.Run(Ort::RunOptions{ nullptr }, input_names, &input_tensor,
                                           1, output_names, 1);

        if (output_tensors.empty() || !output_tensors.front().IsTensor()) {
            APP_ERROR("ONNX Runtime inference produced invalid output");
            return false;
        }

        // 复制输出数据到用户提供的缓冲区
        auto &  output_tensor = output_tensors.front();
        float * src           = output_tensor.GetTensorMutableData<float>();
        size_t  output_count  = output_tensor.GetTensorTypeAndShapeInfo().GetElementCount();
        std::memcpy(output_data, src, output_count * sizeof(float));

        return true;
    } catch (const Ort::Exception & e) {
        APP_ERROR("ONNX Runtime inference failed: {}", e.what());
        return false;
    }
}

bool OnnxRuntimeBackend::runInferenceAsync(void * input_data,
                                           void * output_data,
                                           cudaStream_t /*stream*/) {
    // ONNX Runtime CPU 后端不支持异步推理，回退到同步推理
    return runInference(input_data, output_data);
}

std::vector<int> OnnxRuntimeBackend::getInputDims() const {
    return std::vector<int>(input_dims_.begin(), input_dims_.end());
}

std::vector<int> OnnxRuntimeBackend::getOutputDims() const {
    return std::vector<int>(output_dims_.begin(), output_dims_.end());
}

size_t OnnxRuntimeBackend::getInputByteSize() const {
    return input_byte_size_;
}

size_t OnnxRuntimeBackend::getOutputByteSize() const {
    return output_byte_size_;
}
