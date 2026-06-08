set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ONNX_VERSION="1.10.0"
ONNX_DIR="${SCRIPT_DIR}/third_party/onnxruntime"

# ---------- 1. 初始化 git submodule ----------
echo "[1/3] Initializing git submodules..."
git config --global --add safe.directory "$(pwd)" || true
git submodule update --init --recursive

# ---------- 2. 下载 ONNX Runtime ----------
# 如果头文件和库已存在则跳过
if [[ -d "${ONNX_DIR}/include" ]] && [[ -f "${ONNX_DIR}/lib/libonnxruntime.so" ]]; then
    echo "[2/3] ONNX Runtime ${ONNX_VERSION} already exists, skipping download."
else
    echo "[2/3] Downloading ONNX Runtime ${ONNX_VERSION}..."

    # 检测系统架构
    ARCH="$(uname -m)"
    case "${ARCH}" in
        x86_64)   ONNX_ARCH="x64"   ;;
        aarch64)  ONNX_ARCH="aarch64" ;;
        arm64)    ONNX_ARCH="aarch64" ;;
        *) echo "Error: unsupported architecture '${ARCH}'"; exit 1 ;;
    esac

    TARBALL="onnxruntime-linux-${ONNX_ARCH}-${ONNX_VERSION}.tgz"
    URL="https://github.com/microsoft/onnxruntime/releases/download/v${ONNX_VERSION}/${TARBALL}"

    # 下载
    curl -fsSL "${URL}" -o "/tmp/${TARBALL}"

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
echo "[3/3] Verifying..."
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