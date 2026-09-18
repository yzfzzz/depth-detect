#!/usr/bin/env bash
# =============================================================================
#  start.sh — depth-detect 一键脚本
#             子模块 -> ONNX Runtime -> 读配置 -> engine 导出
#             -> 刷新 config 路径 -> 系统依赖 -> 编译 -> 运行
#
#  用法:
#    ./start.sh                           # 全流程（bin/config.yaml + 默认视频）
#    ./start.sh --video data/2_car.mp4    # 指定视频文件
#    ./start.sh --video 0                 # 指定 USB 相机索引（纯数字 = 相机）
#    ./start.sh --no-run                  # 只构建不运行（Docker / CI 友好）
#    ./start.sh --headless                # 强制无 GUI 运行（is_display=false）
#    ./start.sh --int8                    # 额外导出 INT8 engine（仅 x86，耗时）
#    ./start.sh --export-all              # 导出 model/onnx 下所有模型
#    ./start.sh --skip-export             # 跳过 engine 导出
#    ./start.sh --help                    # 查看全部参数
#
#  设计要点:
#    1. 模型清单不再硬编码：由 scripts/read_config.py 解析 bin/config.yaml 中
#       type=engine|light_engine 的条目，按「engine 所在目录名」去
#       model/onnx/<同名目录>/ 找对应 ONNX 再导出。
#       -> config 换模型（yolo26s / yolo26n / yolo26n-depth / lite-mono-tiny ...）
#          本脚本零改动。
#    2. 无 GPU 或 prefer.use_gpu=false 时**不中断**：跳过 engine 导出，
#       自动降级为 config 里 type=onnx 的 CPU 推理路径，依赖安装 / 编译 / 运行照常。
#    3. 引擎路径刷新由 scripts/update_config.py 完成：模型身份取自条目自身目录名
#       （绝不被扫描到的其它模型顶替），并写回相对 bin/ 的相对路径。
#    4. 不安装 Python 依赖：假定运行环境（Docker 镜像等）已具备
#       pyyaml / numpy / opencv / pycuda 等。
#    5. 本脚本**运行时输出全部为英文 ASCII**（仅注释保留中文），
#       避免在缺少中文字体 / 编码支持的系统上出现乱码。
#    6. 配套脚本放在父仓库的 scripts/ 下，**不放 model/ 子模块**：
#       model/ 是 submodule，放进去的改动会被 submodule 更新重置掉；
#       同时第 1 步的子模块更新**永不使用 --force**，任何情况下都不会丢弃
#       子模块内已跟踪文件的本地改动。
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

trap 'rc=$?; echo "" >&2; echo "  [FAIL] start.sh aborted at line ${LINENO} (exit code ${rc})" >&2' ERR

# Git Bash / Cygwin 下 bash 的 /d/xxx 形式路径无法交给原生 Python / 可执行文件，
# 统一在此转换；Linux 无 cygpath 时原样返回。
native_path() {
    if command -v cygpath >/dev/null 2>&1; then
        cygpath -w -a "$1"
    else
        printf '%s' "$1"
    fi
}

# -----------------------------------------------------------------------------
# 默认参数（均可由命令行覆盖）
# -----------------------------------------------------------------------------
CONFIG_REL="bin/config.yaml"
VIDEO=""                        # 空 = 自动挑选 data/ 下第一个视频
RUN_TASK=true                   # --no-run 置 false
EXPORT_MODE="auto"              # auto / force / off
ENABLE_INT8=false
EXPORT_ALL=false
FORCE_HEADLESS=false
SKIP_SUBMODULE=false
SKIP_ONNXRUNTIME=false
SKIP_SYSDEPS=false
SKIP_BUILD=false

ONNX_VERSION="1.10.0"
CALIB_VIDEO_REL="data/1shu_east_0514.mp4"

# 步骤计数器（保证编号连续，与是否有步骤被跳过无关）
STEP=0
TOTAL=8
step() {
    STEP=$((STEP + 1))
    echo ""
    echo "=============== [${STEP}/${TOTAL}] $* ==============="
}

usage() {
    cat <<'EOF'
Usage:
  ./start.sh [options]

Pipeline (8 steps):
  1) Initialize git submodules      2) Prepare ONNX Runtime
  3) Read config and preflight      4) Export TensorRT engines
  5) Refresh model paths in config  6) Check system dependencies
  7) Build project                  8) Run bin/main

Options:
  --config <path>      config file path (default: bin/config.yaml)
  --video  <path|idx>  video file or camera index for the run (digits = camera, e.g. 0)
  --no-run             build only, do not run (Docker / CI friendly)
  --headless           force GUI-less run (generates a temp is_display=false config)
  --int8               also export INT8 engines (x86 only, calibration is slow)
  --export-all         export every model under model/onnx
  --export             force engine export (ignore GPU / use_gpu switch)
  --skip-export        skip engine export
  --skip-submodule     skip git submodule sync
  --skip-onnxruntime   skip ONNX Runtime download/check
  --skip-deps          skip system dependency check
  --skip-build         skip compilation
  -h, --help           show this help

Notes:
  * Which engines to export is decided by the type=engine / light_engine entries in
    bin/config.yaml; the matching ONNX is looked up in model/onnx/<same model dir>/
    (op11 preferred).
  * Without an NVIDIA GPU, or with prefer.use_gpu=false, the script does NOT abort:
    it falls back to the type=onnx models (CPU backend) and still installs
    dependencies, builds and runs.
  * Python dependencies are assumed to be present already (Docker image etc.).
EOF
    exit 0
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --config)
            [[ $# -ge 2 ]] || { echo "--config requires a value" >&2; exit 1; }
            CONFIG_REL="$2"; shift 2 ;;
        --video)
            [[ $# -ge 2 ]] || { echo "--video requires a value" >&2; exit 1; }
            VIDEO="$2";      shift 2 ;;
        --no-run)          RUN_TASK=false;  shift ;;
        --headless)        FORCE_HEADLESS=true; shift ;;
        --int8)            ENABLE_INT8=true; shift ;;
        --export-all)      EXPORT_ALL=true;  shift ;;
        --skip-export)     EXPORT_MODE="off"; shift ;;
        --export)          EXPORT_MODE="force"; shift ;;
        --skip-submodule)  SKIP_SUBMODULE=true; shift ;;
        --skip-onnxruntime) SKIP_ONNXRUNTIME=true; shift ;;
        --skip-deps)       SKIP_SYSDEPS=true; shift ;;
        --skip-build)      SKIP_BUILD=true;   shift ;;
        --help|-h)         usage ;;
        *) echo "Unknown argument: $1 (use --help for usage)" >&2; exit 1 ;;
    esac
done

CONFIG_ABS="${SCRIPT_DIR}/${CONFIG_REL}"
[[ -f "$CONFIG_ABS" ]] || { echo "Config file not found: $CONFIG_ABS" >&2; exit 1; }

# Python 解释器（容器内 python3 / 部分环境只有 python）
PY="$(command -v python3 || command -v python || true)"
[[ -n "$PY" ]] || { echo "python3/python not found, cannot continue" >&2; exit 1; }

case "$(uname -m)" in
    x86_64) ARCH="x64" ;;
    aarch64 | arm64) ARCH="aarch64" ;;
    *) echo "Unsupported architecture: $(uname -m)" >&2; exit 1 ;;
esac

echo "============================================================"
echo "  depth-detect bootstrap"
echo "  Project root : ${SCRIPT_DIR}"
echo "  Config       : ${CONFIG_REL}"
echo "  Architecture : $(uname -m) (${ARCH})"
echo "============================================================"

# =============================================================================
# 1. 初始化 git submodule
# =============================================================================
step "Initializing git submodules"
if ${SKIP_SUBMODULE}; then
    echo "  [SKIP] --skip-submodule"
else
    # 将仓库根目录及其所有子模块的 gitdir 都加入安全列表
    for d in "" "/model" "/third_party/spdlog" "/third_party/onnxruntime" \
             "/third_party/JsonSenderTest" "/third_party/OneEuroFilter"; do
        git config --global --add safe.directory "${SCRIPT_DIR}${d}" || true
    done
    git submodule sync --recursive
    # 绝不使用 --force：子模块里「已跟踪文件」的本地改动会被它直接丢弃，
    # 且父仓库自身的脚本一旦误放进子模块就会被反复重置。
    # 非强制更新在存在本地改动时只会报错退出，因此先更新、失败再给出可执行的提示。
    if ! git submodule update --init --recursive; then
        echo "  [WARN] submodule update did not complete; keeping the current checkouts" >&2
        # 注意：git submodule foreach 的脚本**不能以可能返回非 0 的语句结尾**，
        # 否则 git 会判定子命令失败并以 128 中止整个 foreach（在第一个干净子模块处就断掉）。
        # 因此这里用 if/then/fi 包裹，保证脚本始终以 0 退出。
        DIRTY_SUBMODULES="$(git submodule foreach --quiet \
            'if git status --porcelain --untracked-files=no | grep -q .; then echo "    $name"; fi' \
            2>/dev/null || true)"
        if [[ -n "$DIRTY_SUBMODULES" ]]; then
            echo "         submodules with local modifications (they block the update):" >&2
            printf '%s\n' "$DIRTY_SUBMODULES" >&2
            echo "         -> commit or stash inside them to let the update proceed" >&2
        fi
    fi
fi

# =============================================================================
# 2. ONNX Runtime（CPU 降级路径的必需依赖）
# =============================================================================
step "Preparing ONNX Runtime ${ONNX_VERSION}"
ONNX_DIR="${SCRIPT_DIR}/third_party/onnxruntime"
if ${SKIP_ONNXRUNTIME}; then
    echo "  [SKIP] --skip-onnxruntime"
elif [[ -f "${ONNX_DIR}/include/onnxruntime_c_api.h" && \
        -f "${ONNX_DIR}/lib/libonnxruntime.so" ]]; then
    echo "  [OK] already present, skipping download"
else
    TARBALL="onnxruntime-linux-${ARCH}-${ONNX_VERSION}.tgz"
    URL_GITHUB="https://github.com/microsoft/onnxruntime/releases/download/v${ONNX_VERSION}/${TARBALL}"
    URL_MIRROR="https://mirror.ghproxy.com/https://github.com/microsoft/onnxruntime/releases/download/v${ONNX_VERSION}/${TARBALL}"

    echo "  Downloading ${TARBALL}"
    if ! wget -q --show-progress "${URL_GITHUB}" -O "/tmp/${TARBALL}" 2>/dev/null; then
        echo "  -> direct GitHub download failed, trying mirror"
        wget -q --show-progress "${URL_MIRROR}" -O "/tmp/${TARBALL}"
    fi

    TMP_DIR="$(mktemp -d)"
    tar -xzf "/tmp/${TARBALL}" -C "${TMP_DIR}"
    rm -rf "${ONNX_DIR}"
    mv "${TMP_DIR}/${TARBALL%.tgz}" "${ONNX_DIR}"
    rm -f "/tmp/${TARBALL}"
    rm -rf "${TMP_DIR}"
    echo "  [OK] installed to ${ONNX_DIR}"
fi

if ! ${SKIP_ONNXRUNTIME}; then
    MISSING_ONNX=()
    for f in "${ONNX_DIR}/include/onnxruntime_c_api.h" \
             "${ONNX_DIR}/include/onnxruntime_cxx_api.h" \
             "${ONNX_DIR}/lib/libonnxruntime.so"; do
        [[ -f "$f" ]] || MISSING_ONNX+=("$f")
    done
    if [[ ${#MISSING_ONNX[@]} -gt 0 ]]; then
        echo "  [FAIL] required files are missing:" >&2
        printf '         %s\n' "${MISSING_ONNX[@]}" >&2
        exit 1
    fi
fi

# =============================================================================
# 3. 读取 bin/config.yaml，预检 + 推导待导出模型清单
#    （解析逻辑在 scripts/read_config.py，退出码 3 = PyYAML 不可用需降级）
# =============================================================================
step "Reading config and running preflight checks"
CONFIG_ABS="$(native_path "$CONFIG_ABS")"
SCRIPT_DIR_NATIVE="$(native_path "$SCRIPT_DIR")"
PY_READ_CONFIG="$(native_path "${SCRIPT_DIR}/scripts/read_config.py")"
PY_UPDATE_CONFIG="$(native_path "${SCRIPT_DIR}/scripts/update_config.py")"

CFG_RC=0
CFG_DUMP="$("${PY}" "${PY_READ_CONFIG}" --config "${CONFIG_ABS}" \
            --project-root "${SCRIPT_DIR_NATIVE}")" || CFG_RC=$?

CFG_YAML_OK=true
EXPORT_LIST=""
if [[ "${CFG_RC}" == "3" ]]; then
    # read_config.py 对这种情况保持静默（由调用方给出提示）
    echo "  [WARN] PyYAML is not available, skipping config preflight and engine export"
    echo "         -> at runtime the type=onnx models (CPU backend) will be used"
    CFG_YAML_OK=false
    CFG_USE_GPU=false
    CFG_IS_DISPLAY=true
    CFG_MODEL_TYPE="unknown"
    CFG_SIMULATE_DELAY=false
    CFG_SEND_TCP=false
    CFG_TCP_TARGET="-"
elif [[ "${CFG_RC}" != "0" ]]; then
    echo "  [FAIL] read_config.py exited with code ${CFG_RC}" >&2
    exit 1
else
    eval "$(printf '%s\n' "$CFG_DUMP" | grep '^CFG_')"
    EXPORT_LIST="$(printf '%s\n' "$CFG_DUMP" | grep '^EXPORT_MODEL=' | cut -d= -f2- || true)"

    echo "  prefer.use_gpu             : ${CFG_USE_GPU}"
    echo "  display_manager.is_display : ${CFG_IS_DISPLAY}"
    echo "  depth.model_type           : ${CFG_MODEL_TYPE}"
    echo "  io_manager.simulate_delay  : ${CFG_SIMULATE_DELAY}"
    echo "  io_manager.send_tcp        : ${CFG_SEND_TCP} -> ${CFG_TCP_TARGET}"

    if [[ -n "$EXPORT_LIST" ]]; then
        echo "  Models needing an engine (type=engine/light_engine in config):"
        while IFS=$'\t' read -r _m _p _o; do
            [[ -n "$_m" ]] || continue
            echo "    - ${_m} (preprocess=${_p}) onnx=${_o:-NOT FOUND}"
        done <<< "$EXPORT_LIST"
    fi
fi

# =============================================================================
# 4. 导出 TensorRT engine（无 GPU / CPU 模式自动降级跳过）
# =============================================================================
step "Exporting TensorRT engines"

GPU_OK=false
if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi >/dev/null 2>&1; then
    GPU_OK=true
fi

DO_EXPORT=true
SKIP_REASON=""
if [[ "$EXPORT_MODE" == "off" ]]; then
    DO_EXPORT=false; SKIP_REASON="--skip-export"
elif [[ "$EXPORT_MODE" == "force" || "$EXPORT_ALL" == "true" ]]; then
    : # 显式要求导出，忽略 GPU / use_gpu 开关
elif [[ "$CFG_YAML_OK" != "true" ]]; then
    DO_EXPORT=false; SKIP_REASON="PyYAML is not available (config cannot be parsed)"
elif [[ "$CFG_USE_GPU" != "true" ]]; then
    DO_EXPORT=false; SKIP_REASON="prefer.use_gpu=false -> using the CPU / ONNX backend"
elif [[ "$GPU_OK" != "true" ]]; then
    DO_EXPORT=false; SKIP_REASON="NVIDIA GPU or driver not available -> falling back to type=onnx (CPU)"
fi

if ${DO_EXPORT}; then
    # 精度集合：默认 fp16 + fp32（config 的 engine 条目即 fp16）；
    # INT8 需 --int8 显式开启（校准耗时，且 aarch64 不支持）
    PRECISIONS=("fp16" "fp32")
    if ${ENABLE_INT8} && [[ "$ARCH" != "aarch64" ]]; then
        PRECISIONS=("int8" "fp16" "fp32")
    fi

    CALIB_VIDEO="$(native_path "${SCRIPT_DIR}/${CALIB_VIDEO_REL}")"

    # export_engine <model_dir> <preprocess> <onnx_path> <precision>
    export_engine() {
        local model_dir="$1" preprocess="$2" onnx_path="$3" precision="$4"
        local onnx_base
        onnx_base="$(basename "${onnx_path%.onnx}")"
        # 与 export_trt_engine.py 的默认命名规则对齐：engine/<模型目录>/<onnx基名>_<精度>_trtX.Y.engine
        if compgen -G "engine/${model_dir}/${onnx_base}_${precision}_*.engine" >/dev/null 2>&1; then
            echo "  [SKIP] ${model_dir} (${precision}) already exists"
            return 0
        fi

        local cmd=("${PY}" export_trt_engine.py --onnx "$onnx_path" --preprocess "$preprocess")
        case "$precision" in
            int8)
                [[ -f "$CALIB_VIDEO" ]] || {
                    echo "  [SKIP] ${model_dir} (int8): calibration video not found: ${CALIB_VIDEO}" >&2
                    return 0
                }
                cmd+=(--int8 --calibVideo "$CALIB_VIDEO")
                ;;
            fp16) cmd+=(--fp16) ;;
            fp32) ;;
        esac

        echo "  [BUILD] ${model_dir} (${precision}, preprocess=${preprocess}) ..."
        if ! "${cmd[@]}"; then
            echo "  [FAIL] ${model_dir} (${precision}) export failed" >&2
            return 1
        fi
        echo "  [DONE] ${model_dir} (${precision})"
    }

    export_one_model() {
        local model_dir="$1" preprocess="$2" onnx_path="$3"
        [[ -n "$onnx_path" && -f "$onnx_path" ]] || {
            echo "  [SKIP] ${model_dir}: no matching ONNX (expected model/onnx/${model_dir}/*op11*.onnx)" >&2
            return 0
        }
        local p
        for p in "${PRECISIONS[@]}"; do
            export_engine "$model_dir" "$preprocess" "$onnx_path" "$p"
        done
    }

    if [[ "$EXPORT_ALL" == "true" ]]; then
        echo "  --export-all: exporting every model under model/onnx (op11 preferred)"
        for onnx in "${SCRIPT_DIR}"/model/onnx/*/*op11*.onnx; do
            [[ -e "$onnx" ]] || continue
            export_one_model "$(basename "$(dirname "$onnx")")" \
                "$(basename "$(dirname "$onnx")" | grep -q '^yolo' && echo yolo || echo depth)" \
                "$(native_path "$onnx")"
        done
    elif [[ -n "${EXPORT_LIST:-}" ]]; then
        while IFS=$'\t' read -r model_dir preprocess onnx_path; do
            [[ -n "$model_dir" ]] || continue
            export_one_model "$model_dir" "$preprocess" "$onnx_path"
        done <<< "$EXPORT_LIST"
    else
        echo "  [WARN] no type=engine/light_engine entry found in config, skipping export" >&2
    fi
else
    echo "  [SKIP] ${SKIP_REASON}"
    echo "         -> at runtime the type=onnx models in config are used (CPU backend), same features, slower"
fi

# =============================================================================
# 5. 刷新 config.yaml 中的 engine / onnx 路径
# =============================================================================
step "Refreshing model paths in config"
if ! "${PY}" "${PY_UPDATE_CONFIG}" \
        --config "${CONFIG_ABS}" --project-root "${SCRIPT_DIR_NATIVE}"; then
    echo "  [FAIL] update_config.py failed" >&2
    exit 1
fi

# =============================================================================
# 6. 系统依赖
# =============================================================================
step "Checking system dependencies"
if ${SKIP_SYSDEPS}; then
    echo "  [SKIP] --skip-deps"
elif ! command -v dpkg >/dev/null 2>&1; then
    echo "  [SKIP] not a Debian-based distro, skipping apt dependency check"
else
    REQUIRED_PKGS=(
        build-essential gcc g++ make cmake gdb git
        libyaml-cpp-dev libeigen3-dev libopencv-dev pybind11-dev
        clangd clang-format
        libbenchmark-dev libbenchmark-tools
        libgl1 libglib2.0-0 libxcb1
        wget ca-certificates
    )

    MISSING=()
    for pkg in "${REQUIRED_PKGS[@]}"; do
        if dpkg -s "$pkg" &>/dev/null; then
            echo "  [ OK ] $pkg"
        else
            echo "  [MISS] $pkg"
            MISSING+=("$pkg")
        fi
    done

    if [[ ${#MISSING[@]} -gt 0 ]]; then
        echo "  Installing ${#MISSING[@]} missing package(s) ..."
        apt-get update
        apt-get install -y --no-install-recommends "${MISSING[@]}"
        apt-get clean
        rm -rf /var/lib/apt/lists/*
    else
        echo "  All dependencies present, skipping apt"
    fi
fi

# =============================================================================
# 7. 编译
# =============================================================================
step "Building project"
if ${SKIP_BUILD}; then
    echo "  [SKIP] --skip-build"
else
    mkdir -p "${SCRIPT_DIR}/build"
    cd "${SCRIPT_DIR}/build"
    cmake -DCMAKE_BUILD_TYPE=Release ..
    make -j"$(nproc)"
    cd "${SCRIPT_DIR}"
    echo "  [OK] build finished"
fi

# =============================================================================
# 8. 运行
# =============================================================================
# 生成 headless 配置副本（is_display: false），只改这一行，保留缩进与行尾注释
write_headless_config() {
    "${PY}" "${PY_READ_CONFIG}" --config "${CONFIG_ABS}" \
        --write-headless "$(native_path "${SCRIPT_DIR}/bin/.config.headless.yaml")" >/dev/null
}

step "Running main"
if ! ${RUN_TASK}; then
    echo "  [SKIP] --no-run"
    echo ""
    echo "  Manual run:"
    echo "    cd bin && ./main ../data/1shu_east_0514.mp4 config.yaml"
else
    # ---- 选择视频源（纯数字 = USB 相机索引）----
    if [[ -z "$VIDEO" ]]; then
        if [[ -f "${SCRIPT_DIR}/${CALIB_VIDEO_REL}" ]]; then
            VIDEO="${SCRIPT_DIR}/${CALIB_VIDEO_REL}"
        else
            VIDEO="$(find "${SCRIPT_DIR}/data" -maxdepth 1 -name '*.mp4' 2>/dev/null | sort | head -1 || true)"
        fi
    elif [[ ! "$VIDEO" =~ ^[0-9]+$ && ! "$VIDEO" =~ ^([A-Za-z]:[\\/]|/) ]]; then
        VIDEO="${SCRIPT_DIR}/${VIDEO}"   # 相对路径按项目根补全
    fi

    if [[ -z "$VIDEO" ]]; then
        echo "  [FAIL] no usable video source (no mp4 under data/), pass one with --video" >&2
        exit 1
    fi
    echo "  Video source: ${VIDEO}"

    # ---- 运行用配置文件（headless 时生成临时副本，不动原 config）----
    RUN_CONFIG="config.yaml"   # 相对 bin/
    if ${FORCE_HEADLESS}; then
        write_headless_config
        RUN_CONFIG=".config.headless.yaml"
        echo "  --headless: generated ${RUN_CONFIG} (is_display=false)"
    elif [[ "${CFG_IS_DISPLAY:-true}" == "true" && -z "${DISPLAY:-}" ]]; then
        echo "  [WARN] DISPLAY is not set but is_display=true in config, switching to headless" >&2
        write_headless_config
        RUN_CONFIG=".config.headless.yaml"
    fi

    # ---- 运行（cwd=bin，config 内相对路径以 bin/ 为基准）----
    RUN_VIDEO="$VIDEO"
    [[ "$VIDEO" =~ ^[0-9]+$ ]] || RUN_VIDEO="$(native_path "$VIDEO")"
    cd "${SCRIPT_DIR}/bin"
    ./main "$RUN_VIDEO" "$RUN_CONFIG"
    cd "${SCRIPT_DIR}"

    echo ""
    echo "  [OK] run finished, output is in bin/out_dir"
fi

echo ""
echo "============================================================"
echo "  Done."
echo "  Run: cd bin && ./main <video path | camera index> config.yaml"
echo "  Docs: https://github.com/yzfzzz/depth-detect"
echo "============================================================"
