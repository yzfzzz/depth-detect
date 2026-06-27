# 基础镜像 — 由 build.sh 自动检测 CUDA 版本后传入，也可手动指定：
#   docker build --build-arg BASE_IMAGE=nvcr.io/nvidia/tensorrt:25.04-py3 .
ARG BASE_IMAGE=nvcr.io/nvidia/tensorrt:25.04-py3
FROM ${BASE_IMAGE}

# 设置时区为上海
ENV TZ=Asia/Shanghai
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# 避免安装交互提示
ENV DEBIAN_FRONTEND=noninteractive

# 安装系统依赖并清理缓存
RUN apt-get update && apt-get install -y --no-install-recommends \
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
    # Ultralytics 依赖的系统库（常用于 OpenCV 显示和视频编解码）
    libgl1 \
    libglib2.0-0 \
    libxcb1 \
    libgl1 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# 全局设置 pip 使用清华大学镜像源加速
ENV PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple

# 安装 python 依赖
RUN pip install --no-cache-dir ultralytics onnxsim timm carla

# 设置工作目录
WORKDIR /workspace

CMD ["/bin/bash"]
