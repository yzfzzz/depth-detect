cmake_minimum_required(VERSION 3.10)
project(depth_detection VERSION 1.0.0 LANGUAGES CXX)

# ==================== Build Type ====================
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# ==================== Language Standards ====================
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# ==================== Global Flags ====================
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/bin)

set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -Wl,--gc-sections")
add_definitions(-DAPI_EXPORTS)

# ==================== Options ====================
option(ENABLE_TIMER "Enable function timing" ON)
if(ENABLE_TIMER)
    message(STATUS "Enabling function timing")
    add_definitions(-DENABLE_TIMER)
endif()

option(ENABLE_JESTON_MEM_MANAGED "Enable Jetson memory managed" OFF)
if(ENABLE_JESTON_MEM_MANAGED)
    message(STATUS "Enabling Jetson memory managed")
    add_definitions(-DENABLE_JESTON_MEM_MANAGED)
endif()

option(ENABLE_BENCHMARK "Build benchmark tests" ON)
option(ENABLE_PYBIND "Build python pybind module" ON)

# ==================== CUDA Detection ====================
# nvcc + cudart + 头文件三者齐备才启用 CUDA 语言与相关 target；
# 缺 CUDA 的环境自动退化为 CPU-only 构建，不编译
# op_kernel / TensorRT 后端。FORCE_NO_CUDA=ON 可在有 CUDA 的机器上强制关闭
option(FORCE_NO_CUDA "Force CPU-only build even if CUDA toolkit exists" OFF)
set(HAS_CUDA OFF)
if(NOT FORCE_NO_CUDA)
    file(GLOB _cuda_roots LIST_DIRECTORIES true
        /usr/local/cuda
        /usr/local/cuda-[0-9]*
        $ENV{CUDA_PATH}
    )
    find_program(CUDA_NVCC_EXECUTABLE
        NAMES nvcc
        HINTS ${_cuda_roots}
        PATH_SUFFIXES bin
    )
    find_path(CUDA_RUNTIME_INCLUDE_DIR
        NAMES cuda_runtime.h
        HINTS ${_cuda_roots}
        PATH_SUFFIXES include
    )
    find_library(CUDA_CUDART_LIBRARY
        NAMES cudart
        HINTS ${_cuda_roots}
        PATH_SUFFIXES lib lib64
    )
    if(CUDA_NVCC_EXECUTABLE AND CUDA_RUNTIME_INCLUDE_DIR AND CUDA_CUDART_LIBRARY)
        set(HAS_CUDA ON)
        enable_language(CUDA)
        set(CMAKE_CUDA_STANDARD 14)
        set(CMAKE_CUDA_STANDARD_REQUIRED ON)
        set(CMAKE_CUDA_EXTENSIONS OFF)
        # 全工程可见的编译期开关：源码据此裁剪 CUDA 相关代码路径
        add_definitions(-DHAS_CUDA)
        message(STATUS "CUDA toolkit found: nvcc=${CUDA_NVCC_EXECUTABLE}, "
                       "include=${CUDA_RUNTIME_INCLUDE_DIR}")
    else()
        message(STATUS "CUDA toolkit not found (nvcc/cudart/headers) — "
                       "building CPU-only: op_kernel and TensorRT backend are skipped")
    endif()
    unset(_cuda_roots)
else()
    message(STATUS "FORCE_NO_CUDA=ON — building CPU-only: op_kernel and TensorRT backend are skipped")
endif()

# ==================== Backend Options ====================
# SDK 级依赖只进入对应后端的编译单元；关掉开关即可在缺 SDK 的机器上构建
option(USE_TENSORRT "Build TensorRT backend" ON)
option(USE_ONNXRUNTIME "Build ONNX Runtime backend" ON)
# QNN 后端为预留扩展点：落地 qnn_backend.h/.cpp 前开启无效（见 backend_factory.cpp）
option(USE_QNN "Build QNN backend (not implemented yet, reserved)" OFF)

# TensorRT 依赖 CUDA：无 CUDA 环境自动关闭而不是配置失败
if(USE_TENSORRT AND NOT HAS_CUDA)
    message(STATUS "USE_TENSORRT=ON but CUDA not found — TensorRT backend disabled")
    set(USE_TENSORRT OFF)
endif()
if(USE_TENSORRT)
    add_definitions(-DUSE_TENSORRT)
endif()
if(USE_ONNXRUNTIME)
    add_definitions(-DUSE_ONNXRUNTIME)
endif()
if(USE_QNN)
    if(EXISTS "${CMAKE_SOURCE_DIR}/cpp/inference/infer_backend/include/qnn_backend.h")
        add_definitions(-DENABLE_QNN)
        # TODO(qnn): QNN SDK include/lib 检测与后端 target 链接在此补充
    else()
        message(WARNING
            "USE_QNN=ON but qnn_backend.h not found — QNN backend not implemented yet, ignoring")
    endif()
endif()

# ==================== Platform Detection ====================
set(TARGET_PLATFORM "AUTO" CACHE STRING "Build target platform")
set_property(CACHE TARGET_PLATFORM PROPERTY STRINGS AUTO NANO2GB TX2 X86_64)
set(TARGET_CUDA_ARCHS "" CACHE STRING "CUDA architectures. Examples: 53, 62, native, 75;86")

function(detect_jetson_platform out_platform)
    set(_platform X86_64)
    if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64")
        set(_jetson_hint "")
        if(EXISTS "/proc/device-tree/model")
            file(READ "/proc/device-tree/model" _model_raw)
            string(APPEND _jetson_hint " ${_model_raw}")
        endif()
        if(EXISTS "/etc/nv_tegra_release")
            file(READ "/etc/nv_tegra_release" _tegra_release_raw)
            string(APPEND _jetson_hint " ${_tegra_release_raw}")
        endif()
        string(TOLOWER "${_jetson_hint}" _jetson_hint_lc)
        if(_jetson_hint_lc MATCHES "quill|tx2|t186")
            set(_platform TX2)
        elseif(_jetson_hint_lc MATCHES "nano|p3448|t210")
            set(_platform NANO2GB)
        else()
            set(_platform TX2)
        endif()
    endif()
    set(${out_platform} "${_platform}" PARENT_SCOPE)
endfunction()

if(TARGET_PLATFORM STREQUAL "AUTO")
    detect_jetson_platform(DETECTED_PLATFORM)
    set(TARGET_PLATFORM "${DETECTED_PLATFORM}")
endif()

set(IS_JETSON OFF)
if(TARGET_PLATFORM STREQUAL "TX2" OR TARGET_PLATFORM STREQUAL "NANO2GB")
    set(IS_JETSON ON)
endif()

message(STATUS "TARGET_PLATFORM = ${TARGET_PLATFORM}")
message(STATUS "IS_JETSON = ${IS_JETSON}")

# ==================== CUDA Architecture ====================
if(HAS_CUDA)
    if(TARGET_CUDA_ARCHS STREQUAL "")
        if(TARGET_PLATFORM STREQUAL "NANO2GB")
            set(TARGET_CUDA_ARCHS 53)
        elseif(TARGET_PLATFORM STREQUAL "TX2")
            set(TARGET_CUDA_ARCHS 62)
        else()
            execute_process(
                COMMAND nvidia-smi --query-gpu=compute_cap --format=csv,noheader
                OUTPUT_VARIABLE GPU_CC
                OUTPUT_STRIP_TRAILING_WHITESPACE
                ERROR_QUIET
                RESULT_VARIABLE NVSMI_RESULT
            )
            if(NVSMI_RESULT EQUAL 0 AND GPU_CC)
                # Extract only first GPU (handle multi-GPU systems)
                string(REGEX MATCH "[0-9]+\\.[0-9]+" GPU_CC_FIRST "${GPU_CC}")
                string(REPLACE "." "" TARGET_CUDA_ARCHS "${GPU_CC_FIRST}")
                message(STATUS "Auto-detected GPU compute capability: sm_${TARGET_CUDA_ARCHS}")
            else()
                message(WARNING "nvidia-smi not available, defaulting to compute_90 (H100)")
                set(TARGET_CUDA_ARCHS 90)
            endif()
        endif()
    endif()

    # 设置 CMake 标准的 CUDA 架构变量（消除 CMP0104 警告）
    set(CMAKE_CUDA_ARCHITECTURES ${TARGET_CUDA_ARCHS})

    # 同时维护手工 CUDA flags（兼容旧写法）
    set(CUDA_ARCH_FLAGS "")
    foreach(ARCH ${TARGET_CUDA_ARCHS})
        list(APPEND CUDA_ARCH_FLAGS "-gencode=arch=compute_${ARCH},code=sm_${ARCH}")
    endforeach()
    message(STATUS "CUDA_ARCH_FLAGS = ${CUDA_ARCH_FLAGS}")
endif()

# ==================== Compiler Options ====================
if(IS_JETSON)
    add_compile_options(
        $<$<COMPILE_LANGUAGE:CXX>:-O3>
        $<$<COMPILE_LANGUAGE:CXX>:-march=native>
        $<$<COMPILE_LANGUAGE:CXX>:-mtune=cortex-a57>
    )
    if(TARGET_PLATFORM STREQUAL "NANO2GB")
        list(APPEND CUDA_ARCH_FLAGS "-maxrregcount=32")
    endif()
else()
    add_compile_options(
        $<$<COMPILE_LANGUAGE:CXX>:-O3>
    )
endif()

if(HAS_CUDA)
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} ${CUDA_ARCH_FLAGS} --use_fast_math")
endif()

# ==================== Dependencies ====================
find_package(OpenCV REQUIRED)
find_package(yaml-cpp REQUIRED)

# TensorRT（仅 USE_TENSORRT=ON 时必需；头文件/库只注入 TensorRT 后端编译单元）
set(TENSORRT_HINT_DIRS
    /usr/local/TensorRT
    /usr/lib/aarch64-linux-gnu
    /usr/lib/x86_64-linux-gnu
    /usr/lib
)
if(USE_TENSORRT)
    # 缓存变量存在时跳过重复探测
    if(NOT TENSORRT_INCLUDE_DIR)
        find_path(TENSORRT_INCLUDE_DIR NvInfer.h HINTS ${TENSORRT_HINT_DIRS} PATH_SUFFIXES include)
    endif()
    if(NOT TENSORRT_NVINFER_LIBRARY)
        find_library(TENSORRT_NVINFER_LIBRARY NAMES nvinfer HINTS ${TENSORRT_HINT_DIRS} PATH_SUFFIXES lib lib64)
    endif()
    if(NOT TENSORRT_NVONNXPARSER_LIBRARY)
        find_library(TENSORRT_NVONNXPARSER_LIBRARY NAMES nvonnxparser HINTS ${TENSORRT_HINT_DIRS} PATH_SUFFIXES lib lib64)
    endif()

    if(NOT TENSORRT_INCLUDE_DIR OR NOT TENSORRT_NVINFER_LIBRARY OR NOT TENSORRT_NVONNXPARSER_LIBRARY)
        message(FATAL_ERROR "TensorRT not found. Please check your TensorRT installation, "
                            "TENSORRT_HINT_DIRS, or build with -DUSE_TENSORRT=OFF.")
    endif()
endif()

# yaml-cpp target compatibility
if(TARGET yaml-cpp::yaml-cpp)
    set(YAML_CPP_TARGET yaml-cpp::yaml-cpp)
else()
    set(YAML_CPP_TARGET yaml-cpp)
endif()
list(APPEND YAML_CPP_TARGET dl)

# ==================== Helper Function ====================
set(PROJECT_CPP_DIR ${PROJECT_SOURCE_DIR}/cpp)
set(ONNXRUNTIME_ROOT ${PROJECT_SOURCE_DIR}/third_party/onnxruntime)

# 1€ 滤波库
set(ONE_EURO_FILTER_DIR ${PROJECT_SOURCE_DIR}/third_party/OneEuroFilter)
if(NOT EXISTS ${ONE_EURO_FILTER_DIR}/cpp/OneEuroFilter.cpp)
    message(FATAL_ERROR "please git submodule update --init --recursive third_party/OneEuroFilter")
endif()
add_library(one_euro_filter STATIC ${ONE_EURO_FILTER_DIR}/cpp/OneEuroFilter.cpp)
target_include_directories(one_euro_filter PUBLIC ${ONE_EURO_FILTER_DIR}/cpp)

add_subdirectory(${PROJECT_SOURCE_DIR}/third_party/spdlog)
find_path(EIGEN3_INCLUDE_DIR NAMES Eigen/Core PATHS /usr/include/eigen3)
if(NOT EIGEN3_INCLUDE_DIR)
    message(FATAL_ERROR "Eigen3 not found, please install: sudo apt install libeigen3-dev")
endif()

# NVTX3 detection — works on both Jetson and x86_64
set(NVTX3_INCLUDE_DIR "")
if(IS_JETSON)
    file(GLOB _nsight_dirs LIST_DIRECTORIES true "/opt/nvidia/nsight-systems/*/target-linux-tegra-armv8/nvtx/include")
    if(_nsight_dirs)
        list(GET _nsight_dirs 0 _first_nsight_dir)
        set(NVTX3_INCLUDE_DIR "${_first_nsight_dir}")
        message(STATUS "Found NVTX3 in Nsight Systems: ${NVTX3_INCLUDE_DIR}")
    endif()
endif()

if(NOT NVTX3_INCLUDE_DIR)
    # Step 1: 先尝试 CMAKE 已知的 CUDA 路径 + PATH_SUFFIXES
    find_path(NVTX3_INCLUDE_DIR NAMES nvtx3/nvtx3.hpp
        HINTS ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
        PATH_SUFFIXES
            targets/x86_64-linux/include
            targets/aarch64-linux/include
        NO_DEFAULT_PATH
    )
endif()

if(NOT NVTX3_INCLUDE_DIR)
    # Step 2: 用 file(GLOB) 搜索版本化安装路径（find_path 不支持通配符）
    file(GLOB _cuda_roots LIST_DIRECTORIES true
        /usr/local/cuda
        /usr/local/cuda-1[0-9]*
    )
    foreach(_cuda_root ${_cuda_roots})
        find_path(_nvtx3_try NAMES nvtx3/nvtx3.hpp
            HINTS ${_cuda_root}
            PATH_SUFFIXES
                targets/x86_64-linux/include
                targets/aarch64-linux/include
                include
            NO_DEFAULT_PATH
        )
        if(_nvtx3_try)
            set(NVTX3_INCLUDE_DIR "${_nvtx3_try}")
            break()
        endif()
    endforeach()
    unset(_cuda_roots)
    unset(_nvtx3_try)
endif()

if(NVTX3_INCLUDE_DIR)
    message(STATUS "NVTX3 include: ${NVTX3_INCLUDE_DIR}")
    add_definitions(-DHAS_NVTX3)
else()
    message(WARNING "nvtx3/nvtx3.hpp not found — NVTX profiling disabled")
endif()


function(configure_common_includes target)
    # 公共 include 只含 OpenCV 等应用级依赖 + 接口层头目录；
    # SDK 级头目录（TensorRT/ONNX Runtime/QNN）只注入各自后端的编译单元，
    # 防止 SDK 头文件经公共路径扩散到模型层与上层代码
    target_include_directories(${target} PRIVATE
        ${PROJECT_CPP_DIR}/include
        ${PROJECT_CPP_DIR}/inference/infer_backend/include
        ${OpenCV_INCLUDE_DIRS}
        ${EIGEN3_INCLUDE_DIR}
        ${NVTX3_INCLUDE_DIR}
    )
    if(HAS_CUDA)
        target_include_directories(${target} PRIVATE
            ${CUDA_RUNTIME_INCLUDE_DIR}
            ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
        )
    endif()
endfunction()

# ==================== Libraries ====================
file(GLOB_RECURSE UTILS_SRCS ${PROJECT_CPP_DIR}/utils/src/*.cpp)
add_library(utils SHARED ${UTILS_SRCS})
configure_common_includes(utils)
target_include_directories(utils PUBLIC ${PROJECT_CPP_DIR}/utils/include)
target_link_libraries(utils PUBLIC spdlog::spdlog ${YAML_CPP_TARGET})

# bytetrack
file(GLOB_RECURSE BYTETRACK_SRCS ${PROJECT_CPP_DIR}/bytetrack/src/*.cpp)
add_library(bytetrack SHARED ${BYTETRACK_SRCS})
configure_common_includes(bytetrack)
target_include_directories(bytetrack PUBLIC ${PROJECT_CPP_DIR}/bytetrack/include)
target_link_libraries(bytetrack PRIVATE utils ${OpenCV_LIBS})

# op_kernel（CUDA kernel 库，仅 HAS_CUDA 构建参与编译）
if(HAS_CUDA)
    file(GLOB_RECURSE OP_KERNEL_SRCS ${PROJECT_CPP_DIR}/op_kernel/src/*.cu)
    add_library(op_kernel SHARED ${OP_KERNEL_SRCS})
    configure_common_includes(op_kernel)
    target_include_directories(op_kernel PUBLIC ${PROJECT_CPP_DIR}/op_kernel/include)
    set_target_properties(op_kernel PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
    if(IS_JETSON)
        target_compile_definitions(op_kernel PRIVATE ENABLE_FP16=1)
    endif()
    target_link_libraries(op_kernel PRIVATE
        ${CUDA_CUDART_LIBRARY}
        ${OpenCV_LIBS}
    )
endif()

# ==================== Inference Backends ====================
# 依赖方向：infer_models -> infer_backend_core(接口+工厂) -> 各后端(可选)。
# SDK 头目录与库只出现在对应后端 target 上
link_directories(${ONNXRUNTIME_ROOT}/lib)

# 接口层（无任何 SDK 依赖；logger_manager.h/config_manager.h 来自 utils 头文件）
add_library(infer_backend_core STATIC ${PROJECT_CPP_DIR}/inference/infer_backend/src/inference_backend.cpp)
configure_common_includes(infer_backend_core)
target_include_directories(infer_backend_core PUBLIC ${PROJECT_CPP_DIR}/inference/infer_backend/include)
target_include_directories(infer_backend_core PRIVATE ${PROJECT_CPP_DIR}/utils/include)
target_link_libraries(infer_backend_core PUBLIC spdlog::spdlog ${YAML_CPP_TARGET})

# TensorRT 后端（可选；USE_TENSORRT 已在检测阶段与 HAS_CUDA 联动）
if(USE_TENSORRT)
    add_library(backend_tensorrt STATIC
        ${PROJECT_CPP_DIR}/inference/infer_backend/src/tensorrt/tensorrt_backend.cpp)
    configure_common_includes(backend_tensorrt)
    target_include_directories(backend_tensorrt PRIVATE ${TENSORRT_INCLUDE_DIR} ${PROJECT_CPP_DIR}/utils/include)
    target_link_libraries(backend_tensorrt PUBLIC infer_backend_core
        ${TENSORRT_NVINFER_LIBRARY} ${CUDA_CUDART_LIBRARY})
endif()

# ONNX Runtime 后端（可选）
if(USE_ONNXRUNTIME)
    add_library(backend_onnxruntime STATIC
        ${PROJECT_CPP_DIR}/inference/infer_backend/src/onnx/onnxruntime_backend.cpp)
    configure_common_includes(backend_onnxruntime)
    target_include_directories(backend_onnxruntime PRIVATE ${ONNXRUNTIME_ROOT}/include ${PROJECT_CPP_DIR}/utils/include)
    target_link_libraries(backend_onnxruntime PUBLIC infer_backend_core onnxruntime)
endif()

# 后端工厂
add_library(backend_factory STATIC ${PROJECT_CPP_DIR}/inference/infer_backend/src/backend_factory.cpp)
configure_common_includes(backend_factory)
target_include_directories(backend_factory PRIVATE ${PROJECT_CPP_DIR}/utils/include ${ONNXRUNTIME_ROOT}/include)
target_link_libraries(backend_factory PUBLIC infer_backend_core)
if(USE_TENSORRT)
    target_include_directories(backend_factory PRIVATE ${TENSORRT_INCLUDE_DIR})
    target_link_libraries(backend_factory PRIVATE backend_tensorrt)
endif()
if(USE_ONNXRUNTIME)
    target_link_libraries(backend_factory PRIVATE backend_onnxruntime)
endif()

# 模型层
file(GLOB_RECURSE INFERENCE_MODEL_SRCS ${PROJECT_CPP_DIR}/inference/infer_models/src/*.cpp)
add_library(inference SHARED ${INFERENCE_MODEL_SRCS})
configure_common_includes(inference)
target_include_directories(inference PUBLIC
    ${PROJECT_CPP_DIR}/inference/infer_models/include
    ${PROJECT_CPP_DIR}/inference/infer_backend/include
)
set_target_properties(inference PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set(INFERENCE_LINK_LIBS
    backend_factory
    infer_backend_core
    utils
    ${OpenCV_LIBS}
)
if(HAS_CUDA)
    list(APPEND INFERENCE_LINK_LIBS op_kernel ${CUDA_CUDART_LIBRARY})
endif()
target_link_libraries(inference PRIVATE ${INFERENCE_LINK_LIBS})

# core
add_subdirectory(third_party/JsonSenderTest)
file(GLOB_RECURSE CORE_SRCS ${PROJECT_CPP_DIR}/core/src/*.cpp)
add_library(core SHARED ${CORE_SRCS})
configure_common_includes(core)
target_include_directories(core PUBLIC ${PROJECT_CPP_DIR}/core/include ${PROJECT_CPP_DIR}/bytetrack/include)
target_include_directories(core PRIVATE ${PROJECT_CPP_DIR}/tools/include)
target_link_libraries(core PUBLIC  inference utils one_euro_filter ${OpenCV_LIBS} JsonSender JsonSenderTcp)

# tools
file(GLOB_RECURSE TOOL_SRCS ${PROJECT_CPP_DIR}/tools/src/*.cpp)
add_library(tools SHARED ${TOOL_SRCS})
configure_common_includes(tools)
target_include_directories(tools PUBLIC ${PROJECT_CPP_DIR}/tools/include ${PROJECT_CPP_DIR}/core/include ${PROJECT_CPP_DIR}/bytetrack/include)
target_link_libraries(tools PRIVATE utils ${OpenCV_LIBS} core)

# ==================== Main Executable ====================
add_executable(main ${PROJECT_CPP_DIR}/main.cpp)
configure_common_includes(main)
set(MAIN_LINK_LIBS core bytetrack tools utils)
if(HAS_CUDA)
    list(APPEND MAIN_LINK_LIBS ${CUDA_CUDART_LIBRARY})
endif()
target_link_libraries(main PRIVATE ${MAIN_LINK_LIBS})

# ==================== Google Benchmark ====================
# 基准测试全部依赖 CUDA/TensorRT 流水线，CPU-only 构建不参与
if(ENABLE_BENCHMARK AND HAS_CUDA)
    message(STATUS "Building benchmark tests")
    find_package(benchmark REQUIRED)
    
    add_executable(test_trt_pipeline ${PROJECT_SOURCE_DIR}/benchmark/test_trt_pipeline.cpp)
    configure_common_includes(test_trt_pipeline)
    target_link_libraries(test_trt_pipeline PRIVATE 
        core bytetrack tools utils ${CUDA_CUDART_LIBRARY} benchmark::benchmark
    )

    add_executable(test_onnx_pipeline ${PROJECT_SOURCE_DIR}/benchmark/test_onnx_pipeline.cpp)
    configure_common_includes(test_onnx_pipeline)
    target_link_libraries(test_onnx_pipeline PRIVATE 
        core bytetrack tools utils ${CUDA_CUDART_LIBRARY} benchmark::benchmark
    )

    add_executable(test_trt_quant_error ${PROJECT_SOURCE_DIR}/benchmark/test_trt_quant_error.cpp)
    configure_common_includes(test_trt_quant_error)
    target_link_libraries(test_trt_quant_error PRIVATE 
        core bytetrack tools utils ${CUDA_CUDART_LIBRARY} benchmark::benchmark
    )
    
    add_executable(test_trt_quant_latency ${PROJECT_SOURCE_DIR}/benchmark/test_trt_quant_latency.cpp)
    configure_common_includes(test_trt_quant_latency)
    target_link_libraries(test_trt_quant_latency PRIVATE
        core bytetrack tools utils ${CUDA_CUDART_LIBRARY} benchmark::benchmark
    )


    add_executable(test_schedule_strategy ${PROJECT_SOURCE_DIR}/benchmark/test_schedule_strategy.cpp)
    configure_common_includes(test_schedule_strategy)
    target_link_libraries(test_schedule_strategy PRIVATE
        core bytetrack tools utils ${CUDA_CUDART_LIBRARY} benchmark::benchmark
    )
else()
    message(STATUS "Benchmarks disabled (ENABLE_BENCHMARK=OFF or no CUDA)")
endif()
