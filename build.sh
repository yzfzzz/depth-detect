#!/usr/bin/env bash
set -euo pipefail

# ============================================================
#  build.sh — Auto-detect host CUDA version & select matching TensorRT image
#
#  Usage:
#    ./build.sh                          # auto-detect CUDA version
#    ./build.sh --base-image <image>     # manually specify base image
#    ./build.sh --dry-run                # show selected image without building
#    ./build.sh --help                   # show help
#
#  Environment variables:
#    BASE_IMAGE    manually specify base image (overrides auto-detection)
#    CUDA_VERSION  manually specify CUDA version, e.g. "12.6"
#    IMAGE_NAME    target image name, default: depth-detect:latest
# ============================================================

IMAGE_NAME="${IMAGE_NAME:-depth-detect:latest}"
BASE_IMAGE="${BASE_IMAGE:-}"
CUDA_VERSION="${CUDA_VERSION:-}"
DRY_RUN=false

# --------------------------------------------------
# Parse command-line arguments
# --------------------------------------------------
while [[ $# -gt 0 ]]; do
    case "$1" in
        --base-image)
            BASE_IMAGE="$2"
            shift 2
            ;;
        --cuda-version)
            CUDA_VERSION="$2"
            shift 2
            ;;
        --image-name)
            IMAGE_NAME="$2"
            shift 2
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --help|-h)
            head -24 "$0" | tail -16
            exit 0
            ;;
        *)
            echo "Unknown argument: $1"
            echo "Use --help for usage"
            exit 1
            ;;
    esac
done

# --------------------------------------------------
# 1. Detect CUDA version
# --------------------------------------------------
detect_cuda_version() {
    local ver=""

    # Method 1: nvidia-smi (most reliable, reflects driver-supported CUDA)
    if command -v nvidia-smi &>/dev/null; then
        ver=$(nvidia-smi 2>/dev/null | grep -oP 'CUDA Version:\s*\K[0-9]+\.[0-9]+' | head -1) || true
        if [[ -n "$ver" ]]; then
            echo "  ✓ nvidia-smi detected CUDA $ver" >&2
            echo "$ver"
            return 0
        fi
    fi

    # Method 2: nvcc
    if command -v nvcc &>/dev/null; then
        ver=$(nvcc --version 2>/dev/null | grep -oP 'release\s+\K[0-9]+\.[0-9]+' | head -1) || true
        if [[ -n "$ver" ]]; then
            echo "  ✓ nvcc detected CUDA $ver" >&2
            echo "$ver"
            return 0
        fi
    fi

    # Method 3: /usr/local/cuda/version.json
    if [[ -f /usr/local/cuda/version.json ]]; then
        ver=$(grep -oP '"cuda"\s*:\s*"[0-9]+\.[0-9]+"' /usr/local/cuda/version.json 2>/dev/null | grep -oP '[0-9]+\.[0-9]+' | head -1) || true
        if [[ -n "$ver" ]]; then
            echo "  ✓ /usr/local/cuda/version.json detected CUDA $ver" >&2
            echo "$ver"
            return 0
        fi
    fi

    # Method 4: /usr/local/cuda/version.txt
    if [[ -f /usr/local/cuda/version.txt ]]; then
        ver=$(grep -oP '[0-9]+\.[0-9]+' /usr/local/cuda/version.txt 2>/dev/null | head -1) || true
        if [[ -n "$ver" ]]; then
            echo "  ✓ /usr/local/cuda/version.txt detected CUDA $ver" >&2
            echo "$ver"
            return 0
        fi
    fi

    return 1
}

# --------------------------------------------------
# 2. CUDA version → TensorRT NGC image mapping
# --------------------------------------------------
# NVIDIA TensorRT NGC container version mapping:
#   https://catalog.ngc.nvidia.com/orgs/nvidia/containers/tensorrt
# Priority: exact match → closest lower version fallback
declare -A CUDA_TO_TRT=(
    # CUDA 12.8
    ["12.8"]="nvcr.io/nvidia/tensorrt:25.04-py3"
    # CUDA 12.7
    ["12.7"]="nvcr.io/nvidia/tensorrt:25.02-py3"
    # CUDA 12.6
    ["12.6"]="nvcr.io/nvidia/tensorrt:24.12-py3"
    # CUDA 12.5
    ["12.5"]="nvcr.io/nvidia/tensorrt:24.10-py3"
    # CUDA 12.4
    ["12.4"]="nvcr.io/nvidia/tensorrt:24.06-py3"
    # CUDA 12.3
    ["12.3"]="nvcr.io/nvidia/tensorrt:24.03-py3"
    # CUDA 12.2
    ["12.2"]="nvcr.io/nvidia/tensorrt:24.01-py3"
    # CUDA 12.1
    ["12.1"]="nvcr.io/nvidia/tensorrt:23.12-py3"
    # CUDA 12.0
    ["12.0"]="nvcr.io/nvidia/tensorrt:23.10-py3"
    # CUDA 11.8
    ["11.8"]="nvcr.io/nvidia/tensorrt:23.08-py3"
)

lookup_trt_image() {
    local cuda_ver="$1"

    # Exact match
    if [[ -n "${CUDA_TO_TRT[$cuda_ver]:-}" ]]; then
        echo "${CUDA_TO_TRT[$cuda_ver]}"
        return 0
    fi

    # Try major.minor only match
    local major_minor="${cuda_ver%.*}"
    # Find the highest version matching major.minor
    local best=""
    for cu_ver in "${!CUDA_TO_TRT[@]}"; do
        if [[ "$cu_ver" == "$cuda_ver" ]]; then
            echo "${CUDA_TO_TRT[$cu_ver]}"
            return 0
        fi
    done

    # Partial match: find latest image ≤ current version
    local best_cu=""
    for cu_ver in "${!CUDA_TO_TRT[@]}"; do
        if [[ "$cu_ver" < "$cuda_ver" || "$cu_ver" == "$cuda_ver" ]]; then
            if [[ -z "$best_cu" ]] || [[ "$cu_ver" > "$best_cu" ]]; then
                best_cu="$cu_ver"
            fi
        fi
    done

    if [[ -n "$best_cu" ]]; then
        echo "${CUDA_TO_TRT[$best_cu]}"
        return 0
    fi

    return 1
}

# --------------------------------------------------
# 3. Main
# --------------------------------------------------
main() {
    echo "============================================"
    echo "  depth-detect — Smart Docker Build Script"
    echo "============================================"
    echo ""

    # Determine CUDA version
    if [[ -z "$CUDA_VERSION" ]]; then
        echo "[*] Detecting host CUDA version..."
        if CUDA_VERSION=$(detect_cuda_version); then
            echo "  → Detected CUDA version: ${CUDA_VERSION}"
        else
            echo "  ✗ Could not detect CUDA version (verify nvidia-smi / nvcc is available)"
            echo "  → Falling back to default: nvcr.io/nvidia/tensorrt:25.04-py3"
            CUDA_VERSION="12.8"
        fi
    else
        echo "[*] Using manually specified CUDA version: ${CUDA_VERSION}"
    fi

    # Determine base image
    if [[ -n "$BASE_IMAGE" ]]; then
        echo "[*] Using manually specified base image: ${BASE_IMAGE}"
    else
        echo "[*] Matching TensorRT image..."
        BASE_IMAGE=$(lookup_trt_image "$CUDA_VERSION")
        if [[ -z "$BASE_IMAGE" ]]; then
            echo "  ✗ No TensorRT image found matching CUDA ${CUDA_VERSION}"
            echo "  → Falling back to default: nvcr.io/nvidia/tensorrt:25.04-py3"
            BASE_IMAGE="nvcr.io/nvidia/tensorrt:25.04-py3"
        fi
        echo "  → Matched image: ${BASE_IMAGE}"
    fi

    echo ""
    echo "  Target image: ${IMAGE_NAME}"
    echo "  Base image:   ${BASE_IMAGE}"
    echo ""

    if $DRY_RUN; then
        echo "[dry-run] Skipping docker build"
        exit 0
    fi

    # Build
    echo "[*] Building Docker image..."
    docker build \
        --build-arg BASE_IMAGE="${BASE_IMAGE}" \
        -t "${IMAGE_NAME}" \
        -f "$(dirname "$0")/Dockerfile" \
        "$(dirname "$0")"

    echo ""
    echo "============================================"
    echo "  ✓ Image built successfully: ${IMAGE_NAME}"
    echo "============================================"
    echo ""
    echo "Run container:"
    echo "  docker run --gpus all -it --restart=unless-stopped --name depth_detect  -v ./:/home/work/depth-detect  -e DISPLAY=host.docker.internal:0.0  ${IMAGE_NAME}"
 ${IMAGE_NAME}"
}

main
