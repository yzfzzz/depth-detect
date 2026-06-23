#include "onnxruntime_backend.h"

#include "logger_manager.h"

#include <cstring>
#include <fstream>

OnnxRuntimeBackend::OnnxRuntimeBackend() : session_(nullptr), input_byte_size_(0) {}

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
        num_outputs_ = session_.GetOutputCount();
        output_tensor_.resize(num_outputs_);
        for (size_t i = 0; i < num_outputs_; i++) {
            char * raw_output_name = session_.GetOutputName(i, allocator);
            output_tensor_[i].name = raw_output_name;
            allocator.Free(raw_output_name);
            auto output_type_info  = session_.GetOutputTypeInfo(i);
            output_tensor_[i].dims = output_type_info.GetTensorTypeAndShapeInfo().GetShape();
            for (auto & dim : output_tensor_[i].dims) {
                if (dim <= 0) {
                    dim = 1;
                }
            }
        }

        // 计算输入/输出数据大小
        input_byte_size_ = 1;
        for (auto dim : input_dims_) {
            input_byte_size_ *= dim;
        }
        input_byte_size_ *= sizeof(float);
        for (int i = 0; i < output_tensor_.size(); i++) {
            size_t byte_size = 1;
            for (auto dim : output_tensor_[i].dims) {
                byte_size *= dim;
            }
            output_tensor_[i].byte_size = byte_size * sizeof(float);
        }

        APP_INFO("ONNX Runtime model loaded successfully from: {}", model_path);
        auto print_dims = [](const std::vector<int64_t> & dims) {
            std::string dims_str;
            for (size_t i = 0; i < dims.size(); ++i) {
                if (i > 0) {
                    dims_str += ", ";
                }
                dims_str += std::to_string(dims[i]);
            }
            return dims_str;
        };

        APP_INFO("Input dims: [{}], byte size: {} bytes", print_dims(input_dims_),
                 input_byte_size_);
        for (const OutputTensorInfo & t : output_tensor_) {
            APP_INFO("Output dims: [{}], byte size: {} bytes", print_dims(t.dims), t.byte_size);
        }

        return true;
    } catch (const Ort::Exception & e) {
        APP_ERROR("ONNX Runtime error: {}", e.what());
        return false;
    }
}

bool OnnxRuntimeBackend::runInference(void * input_data, std::vector<void *> output_data) {
    if (!session_) {
        APP_ERROR("ONNX Runtime session not initialized");
        return false;
    }
    if (output_data.size() != output_tensor_.size()) {
        APP_ERROR("output_data size is not equal to output_tensor_.size()");
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

        const char *              input_names[] = { input_name_.c_str() };
        std::vector<const char *> output_names;
        for (int i = 0; i < output_tensor_.size(); i++) {
            output_names.push_back(output_tensor_[i].name.c_str());
        }

        auto output = session_.Run(Ort::RunOptions{ nullptr }, input_names, &input_tensor, 1,
                                   output_names.data(), output_names.size());

        if (output.empty() || !output.front().IsTensor()) {
            APP_ERROR("ONNX Runtime inference produced invalid output");
            return false;
        }

        // 复制输出数据到用户提供的缓冲区
        for (int i = 0; i < output_tensor_.size(); i++) {
            auto &  output_tensor = output[i];
            float * src           = output_tensor.GetTensorMutableData<float>();
            size_t  output_count  = output_tensor.GetTensorTypeAndShapeInfo().GetElementCount();
            std::memcpy(output_data[i], src, output_count * sizeof(float));
        }
        return true;
    } catch (const Ort::Exception & e) {
        APP_ERROR("ONNX Runtime inference failed: {}", e.what());
        return false;
    }
}

bool OnnxRuntimeBackend::runInference(void * input_data, void * output_data) {
    return runInference(input_data, std::vector<void *>{ output_data });
}

bool OnnxRuntimeBackend::runInferenceAsync(void * input_data,
                                           void * output_data,
                                           cudaStream_t /*stream*/) {
    // ONNX Runtime CPU 后端不支持异步推理，回退到同步推理
    APP_WARN(
        "ONNX Runtime CPU backend does not support async inference, falling back to sync "
        "inference");
    return runInference(input_data, output_data);
}

bool OnnxRuntimeBackend::runInferenceAsync(void *              input_data,
                                           std::vector<void *> output_data,
                                           cudaStream_t /*stream*/) {
    // ONNX Runtime CPU 后端不支持异步推理，回退到同步推理
    APP_WARN(
        "ONNX Runtime CPU backend does not support async inference, falling back to sync "
        "inference");
    return runInference(input_data, output_data);
}

std::vector<int> OnnxRuntimeBackend::getInputDims() const {
    return std::vector<int>(input_dims_.begin(), input_dims_.end());
}

std::vector<int64_t> OnnxRuntimeBackend::getOutputDims(int output_index) const {
    if (output_index < 0 || output_index >= static_cast<int>(output_tensor_.size())) {
        APP_ERROR("Invalid output index: {}", output_index);
        return std::vector<int64_t>();
    }
    return output_tensor_[output_index].dims;
}

size_t OnnxRuntimeBackend::getInputByteSize() const {
    return input_byte_size_;
}

size_t OnnxRuntimeBackend::getOutputByteSize(int output_index) const {
    if (output_index < 0 || output_index >= static_cast<int>(output_tensor_.size())) {
        APP_ERROR("Invalid output index: {}", output_index);
        return 0;
    }
    return output_tensor_[output_index].byte_size;
}
