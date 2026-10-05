#pragma once

#include <map>
#include <memory>
#include <string>
#include <vector>

class InferenceBackend;
// 后端类型名与 yaml 中模型路径条目 key 的对应关系：
//   tensorrt    -> "engine"
//   onnxruntime -> "onnx"
//   qnn         -> "qnn"
std::string backendPathKeyForBackend(const std::string & backend_type);

// 创建指定类型的后端实例（不加载模型，loadModel 由调用方负责，便于逐候选回退尝试）
// 类型未知或该后端未参与编译时返回 nullptr
std::unique_ptr<InferenceBackend> createBackendByType(const std::string & backend_type,
                                                      int                 gpu_id = 0);

// 依配置解析后端候选链（按优先级排列）：
//   preferred 为具体类型（"tensorrt"/"onnxruntime"/...）-> 只有一项，不做静默回退，
//     加载失败由调用方直接报错
//   preferred 为 "auto" -> 保持历史回退链：use_gpu && GPU 可用 && 配置了 engine 路径
//     -> tensorrt；配置了 onnx 路径 -> onnxruntime
std::vector<std::string> backendCandidateChain(
    const std::map<std::string, std::string> & model_paths,
    bool                                       use_gpu,
    const std::string &                        preferred = "auto");
