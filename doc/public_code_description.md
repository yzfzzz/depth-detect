# 公共代码说明

本文档用于沉淀项目内全局可用或可多次复用的公共组件、公共函数、公共变量和公共配置。后续开发、修复或重构过程中，如果新增或修改了符合公共内容判断规则的代码，请同步维护本文档。

## 维护原则

1. 只记录具有复用价值、需要其他开发人员了解使用方式或注意事项的公共内容。
2. 页面、模块或文件内部一次性逻辑、局部变量、局部状态、局部事件处理函数一般不写入本文档。
3. 修改已有公共内容时，优先更新原条目，不要重复新增相似说明。
4. 本文档不会自动回填历史公共代码；除非单独执行公共能力盘点任务，否则从后续开发中逐步沉淀。

## 公共组件

### Pipeline（推理流水线）

- 文件路径：`cpp/core/include/pipeline.h`、`cpp/core/src/pipeline.cpp`
- 用途：串联 YOLO 检测、深度估计、ByteTrack 跟踪、运动状态判定的端到端推理流水线。提供同步 `process()` 和 CPU/GPU 重叠 `processOverlap()` 两种推理模式。
- 使用约定：构造传入模型路径（支持 .onnx 或 .engine）、FrameMeta 和 use_gpu 标志。调用 `process()` 或 `processOverlap()` 逐帧推理，输出写入 InferOutputContext。
- 注意事项：
  - `processOverlap()` 利用 CUDA Stream 实现 YOLO 与 Depth 模型的并行推理，仅在 GPU 模式下生效。
  - `track_classes_` 控制需要跟踪的 COCO 类别索引（默认：人、自行车、汽车、摩托车、公交车、卡车）。
  - 深度图和可视化结果在帧间缓存，需要访问时从 InferOutputContext 获取。

### BaseModel（模型基类）

- 文件路径：`cpp/inference/infer_models/include/base_model.h`、`cpp/inference/infer_models/src/base_model.cpp`
- 用途：所有推理模型的抽象基类，封装 ONNX Runtime / TensorRT 后端初始化、预处理、推理、后处理的通用流程。
- 使用约定：子类实现 `init()`、预处理/推理/后处理等虚函数。通过 `InferBackend` 工厂选择后端。
- 注意事项：构造时通过 `use_gpu` 参数控制是否启用 GPU 推理；`normalize_` 标志控制输入归一化。

### DepthModel（深度估计模型）

- 文件路径：`cpp/inference/infer_models/include/depth_model.h`、`cpp/inference/infer_models/src/depth_model.cpp`
- 用途：基于 Lite-Mono-Tiny 的单目深度估计模型。支持 ONNX 和 TensorRT 两种后端。
- 使用约定：由 Pipeline 内部管理，不需要单独调用。
- 注意事项：GPU 路径下预处理、推理、后处理均在 GPU 侧执行；CPU 路径使用 OpenCV 处理。

### YoloDetectModel（目标检测模型）

- 文件路径：`cpp/inference/infer_models/include/yolo_detect_model.h`、`cpp/inference/infer_models/src/yolo_detect_model.cpp`
- 用途：YOLOv8 目标检测模型。支持 ONNX 和 TensorRT 后端，含 NMS 后处理。
- 使用约定：`nms_thresh` 和 `conf_thresh` 控制检测阈值；`topk` 控制最大输出检测数。
- 注意事项：检测输出格式为 `[x1, y1, x2, y2, conf, class_id]`。

### InferBackend（推理后端抽象）

- 文件路径：`cpp/inference/infer_backend/`
- 用途：统一 ONNX Runtime 和 TensorRT 两种推理后端的接口，支持根据模型文件扩展名自动选择后端（.onnx → ONNX，.engine → TensorRT）。
- 注意事项：TensorRT 后端兼容 TRT 8.x（`destroy()`）和 10.x（标准 C++ 析构），通过 `NV_TENSORRT_MAJOR` 宏区分。

### MotionStateEngine（运动状态引擎）

- 文件路径：`cpp/core/include/motion_state_engine.h`、`cpp/core/src/motion_state_engine.cpp`
- 用途：基于卡尔曼滤波的物体运动状态判定，估算每个跟踪目标的速度和加速度，判断靠近/远离/加速/减速/匀速等状态。
- 使用约定：由 Pipeline 内部管理，每帧调用 `update()` 更新状态。
- 注意事项：
  - 当前按视差逻辑处理（值变大 = 靠近），若使用绝对深度需反转方向判定。
  - MotionState 枚举定义在 `frame.h` 中。

### ByteTrack Tracker（多目标跟踪器）

- 文件路径：`cpp/bytetrack/`
- 用途：ByteTrack 多目标跟踪算法实现，含卡尔曼滤波状态估计和 LAPJV 线性分配。
- 使用约定：由 Pipeline 内部管理。
- 注意事项：LAPJV 为稠密矩阵实现，适用于检测数较多场景；低分检测框也参与匹配以提高召回。

### ConfigManager（配置管理器）

- 文件路径：`cpp/utils/include/config_manager.h`、`cpp/utils/src/config_manager.cpp`
- 用途：基于 YAML 的全局配置管理，加载模型路径、推理参数、视频源等配置。
- 使用约定：构造时传入 YAML 配置文件路径；通过 `get()` / `getNode()` 读取配置项。
- 注意事项：配合 `benchmark.yaml` 和 `config.yaml` 使用。

### IOManager（输入输出管理器）

- 文件路径：`cpp/core/include/io_manager.h`、`cpp/core/src/io_manager.cpp`
- 用途：统一管理视频/摄像头输入和可视化输出。支持多输出模式（video、none、display）。
- 使用约定：构造 `IOManager(output_mode, video_path)` → `Init()` → 循环 `readNextFrame()`。
- 注意事项：`readNextFrame()` 返回 false 时表示视频结束，需调用 `Init()` 重新打开以循环播放。

### LoggerManager（日志管理器）

- 文件路径：`cpp/utils/include/logger_manager.h`、`cpp/utils/src/logger_manager.cpp`
- 用途：基于 spdlog 的单例日志系统，同时输出控制台和文件日志。提供 `APP_INFO`、`APP_WARN`、`APP_ERROR`、`APP_DEBUG` 便捷宏。
- 使用约定：程序入口调用 `LoggerManager::getInstance()` 初始化；之后直接使用 `APP_*` 宏。
- 注意事项：单例模式，重复调用 `getInstance` 返回同一实例。

### DisplayManager / VisualManager（可视化管理器）

- 文件路径：`cpp/tools/include/visual_manager.h`、`cpp/tools/src/visual_manager.cpp`
- 用途：负责推理结果（检测框、跟踪 ID、运动状态、深度图）的 OpenCV 可视化渲染。
- 使用约定：由 IOManager 内部管理，不直接调用。

### ScopedTimer（作用域计时器）

- 文件路径：`cpp/tools/include/scope_timer.h`
- 用途：RAII 风格的计时器，构造时开始计时，析构时输出耗时。用于性能分析和调优。
- 使用约定：在需要测量的代码块前声明 `ScopedTimer timer("阶段名")`；离开作用域自动输出耗时。

### CUDA/TRT 内存管理（memory.h）

- 文件路径：`cpp/include/memory.h`
- 用途：定义 CUDA 显存（`CudaDeleter`）、固定内存（`PinnedCudaDeleter`）和 TRT 对象（`TrtDeleter`）的智能指针类型别名。
- 使用约定：使用 `unique_ptr_cuda<T>`、`unique_ptr_pinned_cuda<T>`、`TrtEnginePtr` 等别名自动管理资源生命周期。
- 注意事项：`TrtDeleter` 通过 `NV_TENSORRT_MAJOR` 宏兼容 TRT 8.x (`destroy()`) 和 10.x (`delete`)。

## 公共函数

### CHECK_CUDA 宏 (`public.h`)

- 文件路径：`cpp/include/public.h`
- 用途：包装 CUDA API 调用，自动检查返回值并在出错时输出错误信息和退出。
- 使用约定：所有 CUDA 调用必须使用此宏包裹，如 `CHECK_CUDA(cudaMalloc(...))`。
- 注意事项：仅供 GPU 代码路径使用；失败时会调用 `exit(1)` 终止程序。

### 推理模型工厂函数（`InferBackend`）

- 文件路径：`cpp/inference/infer_backend/`
- 用途：根据模型文件扩展名（.onnx / .engine）自动创建对应的推理后端实例。
- 使用约定：由 `BaseModel` 子类内部调用，不需要直接使用。

## 公共变量与常量

暂无记录。

## 公共配置

### benchmark.yaml

- 文件路径：`bin/benchmark.yaml`
- 用途：Benchmark 测试的统一配置文件，定义视频路径、各精度模型路径（INT8/FP16/FP32）、ONNX 模型路径等。
- 使用约定：所有 benchmark 测试文件（`test_trt_*.cpp`、`test_onnx_*.cpp`）共用此配置。
- 注意事项：`task` 节点下按任务名分组，新增 benchmark 时在此文件添加对应配置段。

### config.yaml

- 文件路径：`bin/config.yaml`
- 用途：正常业务推理的配置文件，定义模型路径、输入源、输出模式等参数。
- 使用约定：由 `main.cpp` 读取，`ConfigManager` 解析。

## 条目模板

```md
### 名称

- 文件路径：
- 用途：
- 使用约定：
- 注意事项：
- 示例：
```
