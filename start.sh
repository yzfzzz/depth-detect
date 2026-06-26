#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ONNX_VERSION="1.10.0"
ONNX_DIR="${SCRIPT_DIR}/third_party/onnxruntime"

# ---------- 1. 初始化 git submodule ----------
echo "[1/7] Initializing git submodules..."
git config --global --add safe.directory "$(pwd)" || true
git submodule update --init --recursive

# ---------- 2. 下载 ONNX Runtime ----------
# 如果头文件和库已存在则跳过
# 检测系统架构
ARCH="$(uname -m)"
case "${ARCH}" in
    x86_64)   ARCH="x64"   ;;
    aarch64)  ARCH="aarch64" ;;
    arm64)    ARCH="aarch64" ;;
    *) echo "Error: unsupported architecture '${ARCH}'"; exit 1 ;;
esac
if [[ -d "${ONNX_DIR}/include" ]] && [[ -f "${ONNX_DIR}/lib/libonnxruntime.so" ]]; then
    echo "[2/7] ONNX Runtime ${ONNX_VERSION} already exists, skipping download."
else
    echo "[2/7] Downloading ONNX Runtime ${ONNX_VERSION}..."

    TARBALL="onnxruntime-linux-${ARCH}-${ONNX_VERSION}.tgz"
    URL="https://github.com/microsoft/onnxruntime/releases/download/v${ONNX_VERSION}/${TARBALL}"

    # 下载
    wget -q --show-progress "${URL}" -O "/tmp/${TARBALL}"

    # 解压到临时目录
    TMP_DIR="$(mktemp -d)"
    tar -xzf "/tmp/${TARBALL}" -C "${TMP_DIR}"

    # 移动到 target 目录
    rm -rf "${ONNX_DIR}"
    mv "${TMP_DIR}/${TARBALL%.tgz}" "${ONNX_DIR}"

    # 清理
    rm -f "/tmp/${TARBALL}"
    rm -rf "${TMP_DIR}"

    echo "  -> ONNX Runtime ${ONNX_VERSION} installed to ${ONNX_DIR}"
fi

# ---------- 3. 验证 ----------
echo "[3/7] Verifying..."
REQUIRED_FILES=(
    "${ONNX_DIR}/include/onnxruntime_c_api.h"
    "${ONNX_DIR}/include/onnxruntime_cxx_api.h"
    "${ONNX_DIR}/lib/libonnxruntime.so"
)
ALL_OK=true
for f in "${REQUIRED_FILES[@]}"; do
    if [[ ! -f "$f" ]]; then
        echo "  [MISSING] $f"
        ALL_OK=false
    fi
done

if ${ALL_OK}; then
    echo "  All dependencies are ready."
else
    echo "  Some files are missing. Please check manually."
    exit 1
fi

# ---------- 检测 GPU ----------
echo "[4/7] Checking for GPU..."
if ! nvidia-smi > /dev/null 2>&1; then
    echo "  [SKIP] No GPU detected, skipping TensorRT engine export."
    exit 0
fi
echo "  [OK] GPU detected, proceeding with TensorRT engine export."

# ---------- 导出 TensorRT Engine ----------
echo "[4/7] Exporting TensorRT engines..."

export_model() {
    local onnx_path="$1"
    local precision_flag="$2"       # e.g. "--int8" / "--fp16" / ""
    local calib_video="$3"          # e.g. "../data/1shu_east_0514.mp4" / ""
    local preprocess_mode="$4"      # "depth" / "yolo"

    # 推断 engine 文件名（与 export_trt_engine.py 内部逻辑对齐）
    local model_dir
    model_dir="$(basename "$(dirname "$onnx_path")")"
    local onnx_base
    onnx_base="$(basename "${onnx_path%.onnx}")"
    local mode
    if [[ "$precision_flag" == *int8* ]]; then
        mode="int8"
    elif [[ "$precision_flag" == *fp16* ]]; then
        mode="fp16"
    else
        mode="fp32"
    fi
    local engine_glob="engine/${model_dir}/${onnx_base}_${mode}_*.engine"

    # 如果已存在则跳过
    if compgen -G "$engine_glob" > /dev/null 2>&1; then
        echo "  [SKIP] ${onnx_base} (${mode}) already exists → ${engine_glob}"
        return 0
    fi

    echo "  [BUILD] ${onnx_base} (${mode}) ..."
    local cmd=(python export_trt_engine.py --onnx "$onnx_path")
    [[ -n "$precision_flag" ]] && cmd+=($precision_flag)
    [[ -n "$calib_video" ]]     && cmd+=(--calibVideo "$calib_video")
    [[ -n "$preprocess_mode" ]] && cmd+=(--preprocess "$preprocess_mode")

    if ! "${cmd[@]}"; then
        echo "  [FAIL] ${onnx_base} (${mode}) failed!" >&2
        return 1
    fi
    echo "  [DONE] ${onnx_base} (${mode})"
}

cd model && pip install -r requirements.txt

CALIB_VIDEO="../data/1shu_east_0514.mp4"

# ---- lite-mono-tiny ----
if [ "$ARCH" ]; then
    # 检查架构，如果是 aarch64 则不导出 INT8
    if [ "$ARCH" = "aarch64" ]; then
        echo "  [INFO] aarch64 architecture detected, skipping INT8 export"
        export_model ./onnx/lite-mono-tiny/lite-mono-tiny_192x640_op11.onnx \
            "--fp16" "" "depth"
        export_model ./onnx/lite-mono-tiny/lite-mono-tiny_192x640_op11.onnx \
            "" "" "depth"
    else
        export_model ./onnx/lite-mono-tiny/lite-mono-tiny_192x640_op11.onnx \
            "--int8" "$CALIB_VIDEO" "depth"
        export_model ./onnx/lite-mono-tiny/lite-mono-tiny_192x640_op11.onnx \
            "--fp16" "" "depth"
        export_model ./onnx/lite-mono-tiny/lite-mono-tiny_192x640_op11.onnx \
            "" "" "depth"
    fi
fi

# ---- yolo26n ----
if [ "$ARCH" ]; then
    # 检查架构，如果是 aarch64 则不导出 INT8
    if [ "$ARCH" = "aarch64" ]; then
        echo "  [INFO] aarch64 architecture detected, skipping INT8 export"
        export_model ./onnx/yolo26n/yolo26n_640_op11.onnx \
            "--fp16" "" "yolo"
        export_model ./onnx/yolo26n/yolo26n_640_op11.onnx \
            "" "" "yolo"
    else
        export_model ./onnx/yolo26n/yolo26n_640_op11.onnx \
            "--int8" "$CALIB_VIDEO" "yolo"
        export_model ./onnx/yolo26n/yolo26n_640_op11.onnx \
            "--fp16" "" "yolo"
        export_model ./onnx/yolo26n/yolo26n_640_op11.onnx \
            "" "" "yolo"
    fi
fi

echo "  All engines built."
ls -lh ./engine
cd ../

# ---------- 5. 更新配置文件 ----------
echo "[5/7] Updating config.yaml & benchmark.yaml with engine paths..."
if ! python3 model/update_config.py --project-root "$SCRIPT_DIR"; then
    echo "  [FAIL] update_config.py, please check update_config.py script." >&2
    exit 1
fi

# ---------- 6. 安装依赖 ----------
echo "[6/7] Installing dependencies..."
apt-get update && apt-get install -y --no-install-recommends \
    libyaml-cpp-dev \
    libeigen3-dev \
    libopencv-dev \
    pybind11-dev \
    clangd \
    clang-format \
    build-essential \
    gcc \
    g++ \
    make \
    cmake \
    gdb \
    git \
    libbenchmark-dev \
    libbenchmark-tools \
    libgl1 \
    libglib2.0-0 \
    libxcb1 \
    libgl1 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# ---------- 7. 编译运行 ----------
echo "[7/7] Checking for build directory..."
if [ ! -d "build" ]; then
    echo "  [INFO] Creating build directory..."
    mkdir build
fi

cd build && cmake -DCMAKE_BUILD_TYPE=Release .. && make -j4 && cd ../
echo "[7/7] Running main..."
echo "  [INFO] You can now run the main program with: ./bin/main <video_path> config.yaml"
cd ./bin/ && ./main ../data/1shu_east_0514.mp4 config.yaml

echo "✅ All done. 😀 Give me a star on GitHub if you like it: https://github.com/yzfzzz/depth-detect"