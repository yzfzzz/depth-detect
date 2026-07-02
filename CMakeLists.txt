cmake_minimum_required(VERSION 3.10)
project(depth_detection VERSION 1.0.0 LANGUAGES CXX CUDA)

# ==================== Build Type ====================
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# ==================== Language Standards ====================
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_EXTENSIONS OFF)

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

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} ${CUDA_ARCH_FLAGS} --use_fast_math")

# ==================== Dependencies ====================
find_package(OpenCV REQUIRED)
find_package(yaml-cpp REQUIRED)

# TensorRT
set(TENSORRT_HINT_DIRS 
    /usr/local/TensorRT 
    /usr/lib/aarch64-linux-gnu 
    /usr/lib/x86_64-linux-gnu 
    /usr/lib
)
find_path(TENSORRT_INCLUDE_DIR NvInfer.h HINTS ${TENSORRT_HINT_DIRS} PATH_SUFFIXES include)
find_library(TENSORRT_NVINFER_LIBRARY NAMES nvinfer HINTS ${TENSORRT_HINT_DIRS} PATH_SUFFIXES lib lib64)
find_library(TENSORRT_NVONNXPARSER_LIBRARY NAMES nvonnxparser HINTS ${TENSORRT_HINT_DIRS} PATH_SUFFIXES lib lib64)

find_library(CUDA_CUDART_LIBRARY NAMES cudart 
    HINTS ${CUDA_TOOLKIT_ROOT_DIR} 
    /usr/local/cuda
    PATH_SUFFIXES lib lib64
)
if(NOT CUDA_CUDART_LIBRARY)
    message(FATAL_ERROR "libcudart not found! Ensure CUDA is installed and CUDA_TOOLKIT_ROOT_DIR is set.")
endif()

if(NOT TENSORRT_INCLUDE_DIR OR NOT TENSORRT_NVINFER_LIBRARY OR NOT TENSORRT_NVONNXPARSER_LIBRARY)
    message(FATAL_ERROR "TensorRT not found. Please check your TensorRT installation or TENSORRT_HINT_DIRS.")
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
    target_include_directories(${target} PRIVATE 
        ${PROJECT_CPP_DIR}/include 
        ${OpenCV_INCLUDE_DIRS} 
        ${TENSORRT_INCLUDE_DIR}
        ${EIGEN3_INCLUDE_DIR}
        ${CUDA_INCLUDE_DIRS}
        ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
        ${NVTX3_INCLUDE_DIR}
        ${ONNXRUNTIME_ROOT}/include
    )
endfunction()

# ==================== Libraries ====================
file(GLOB_RECURSE UTILS_SRCS ${PROJECT_CPP_DIR}/utils/src/*.cpp)
add_library(utils SHARED ${UTILS_SRCS})
configure_common_includes(utils)
target_include_directories(utils PUBLIC ${PROJECT_CPP_DIR}/utils/include)
target_link_libraries(utils PUBLIC spdlog::spdlog ${YAML_CPP_TARGET})
target_link_libraries(utils PRIVATE ${TENSORRT_NVINFER_LIBRARY})

# bytetrack
file(GLOB_RECURSE BYTETRACK_SRCS ${PROJECT_CPP_DIR}/bytetrack/src/*.cpp)
add_library(bytetrack SHARED ${BYTETRACK_SRCS})
configure_common_includes(bytetrack)
target_include_directories(bytetrack PUBLIC ${PROJECT_CPP_DIR}/bytetrack/include)
target_link_libraries(bytetrack PRIVATE utils ${OpenCV_LIBS})

# op_kernel
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
    ${TENSORRT_NVINFER_LIBRARY} 
    ${TENSORRT_NVONNXPARSER_LIBRARY} 
    ${OpenCV_LIBS}
)

# inference model
link_directories(${ONNXRUNTIME_ROOT}/lib)
file(GLOB_RECURSE INFERENCE_SRCS 
    ${PROJECT_CPP_DIR}/inference/infer_models/src/*.cpp 
    ${PROJECT_CPP_DIR}/inference/infer_backend/src/*.cpp 
)
add_library(inference SHARED ${INFERENCE_SRCS})
configure_common_includes(inference)
target_include_directories(inference PUBLIC 
    ${PROJECT_CPP_DIR}/inference/infer_models/include 
    ${PROJECT_CPP_DIR}/inference/infer_backend/include
)
set_target_properties(inference PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_link_libraries(inference PRIVATE 
    op_kernel 
    utils
    ${CUDA_CUDART_LIBRARY}   
    ${TENSORRT_NVINFER_LIBRARY} 
    ${OpenCV_LIBS}
    onnxruntime
)

# core
add_subdirectory(third_party/JsonSenderTest)
file(GLOB_RECURSE CORE_SRCS ${PROJECT_CPP_DIR}/core/src/*.cpp)
add_library(core SHARED ${CORE_SRCS})
configure_common_includes(core)
target_include_directories(core PUBLIC ${PROJECT_CPP_DIR}/core/include ${PROJECT_CPP_DIR}/bytetrack/include)
target_link_libraries(core PUBLIC  inference utils ${OpenCV_LIBS} JsonSender)

# tools
file(GLOB_RECURSE TOOL_SRCS ${PROJECT_CPP_DIR}/tools/src/*.cpp)
add_library(tools SHARED ${TOOL_SRCS})
configure_common_includes(tools)
target_include_directories(tools PUBLIC ${PROJECT_CPP_DIR}/tools/include ${PROJECT_CPP_DIR}/core/include ${PROJECT_CPP_DIR}/bytetrack/include)
target_link_libraries(tools PRIVATE utils ${OpenCV_LIBS} core)

# ==================== Main Executable ====================
add_executable(main ${PROJECT_CPP_DIR}/main.cpp)
configure_common_includes(main)
target_link_libraries(main PRIVATE core bytetrack tools utils ${CUDA_CUDART_LIBRARY})

# ==================== Google Benchmark ====================
if(ENABLE_BENCHMARK)
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
else()
    message(STATUS "ENABLE_BENCHMARK=OFF")
endif()
