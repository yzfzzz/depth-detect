#!/usr/bin/env python3
"""
配置文件统一管理脚本

子命令:
    gen       生成 bin/ 下三份运行配置（config.yaml / config_jetson.yaml / config_ci.yaml），
              生成时检测模型路径并只做报告：[OK] 本机存在 / [MISS] 本机不存在（保留模板值，
              如 config_jetson 的 trt8.2 属于 Jetson 目标环境意图，不在开发机上改写）。
              需要把路径刷成真实文件时，在目标机器上执行 refresh 子命令
    read      读取 config，输出 KEY=VALUE 运行时信息供 start.sh 解析（含 --write-headless）
    refresh   就地刷新 config 中的 engine/onnx 模型路径（身份保持：条目只在自己
              当前的模型目录内挑文件，绝不跨目录漂移）

用法:
    python3 scripts/config_tool.py gen                       # 生成全部三份 + 路径检测
    python3 scripts/config_tool.py gen --variant config_ci   # 只生成指定一份
    python3 scripts/config_tool.py read                      # 预检 + 模型导出清单
    python3 scripts/config_tool.py refresh --dry-run         # 只打印将要做的改动

read 输出格式（KEY=VALUE，一行一条，供 start.sh eval）:
    CFG_USE_GPU=true|false
    CFG_IS_DISPLAY=true|false
    CFG_MODEL_TYPE=<lite_mono|yolo_depth|...>
    CFG_DEPTH_ENABLED=true|false
    CFG_SIMULATE_DELAY=true|false
    CFG_SEND_TCP=true|false
    CFG_TCP_TARGET=<ip>:<port>
    EXPORT_MODEL=<模型目录>\t<preprocess>\t<onnx 绝对路径>

退出码（read 沿用原约定，start.sh 依赖）:
    gen/refresh: 0 成功, 1 参数/IO 错误, 2 config 文件不存在
    read:        0 成功, 1 其他错误, 2 config 不存在, 3 PyYAML 不可用（调用方降级）
"""

import argparse
import os
import re
import sys

# ============================================================================
# 模型路径检测 / 刷新（gen 与 refresh 共用；身份保持语义来自原 update_config.py）
# ============================================================================

ENGINE_TYPES = ("engine", "light_engine")
ONNX_TYPES = ("onnx", "light_onnx")
PRECISION_ORDER = ("fp16", "fp32", "int8")

TYPE_RE = re.compile(r"^\s*-\s*type:\s*([A-Za-z_][A-Za-z0-9_]*)\s*(?:#.*)?$")
PATH_RE = re.compile(r'^(\s*path:\s*)"([^"]*)"(.*)$')


def detect_precision(name):
    """从文件名推断精度；engine 文件名约定为 ..._<精度>_trtX.Y.engine。"""
    for precision in PRECISION_ORDER:
        if precision in name:
            return precision
    return None


def detect_op(name):
    """从文件名推断 onnx opset 标记，如 op11。"""
    match = re.search(r"(op\d+)", name)
    return match.group(1) if match else None


def list_model_files(directory, suffix):
    if not os.path.isdir(directory):
        return []
    return sorted(
        name for name in os.listdir(directory)
        if name.endswith(suffix) and os.path.isfile(os.path.join(directory, name))
    )


def detect_trt_tag(name):
    """从文件名推断 TensorRT 版本标记，如 trt10.9 / trt8.2。"""
    match = re.search(r"(trt\d+\.\d+)", name)
    return match.group(1) if match else None


def pick_engine(directory, current_name):
    """
    在 directory 下挑一个 .engine，优先级从高到低：
      1) 同精度 + 同 TRT 版本标记（身份完全一致，避免自己改自己的配置）
      2) 同 TRT 版本标记（版本比精度更能标识 engine 的可用性）
      3) 同精度
      4) 按 fp16 > fp32 > int8 取最优
    """
    candidates = list_model_files(directory, ".engine")
    if not candidates:
        return None

    current_precision = detect_precision(current_name)
    current_trt = detect_trt_tag(current_name)
    if current_precision and current_trt:
        exact = [name for name in candidates
                 if detect_precision(name) == current_precision
                 and detect_trt_tag(name) == current_trt]
        if exact:
            return exact[0]
    if current_trt:
        same_trt = [name for name in candidates if detect_trt_tag(name) == current_trt]
        if same_trt:
            return same_trt[0]
    if current_precision:
        same = [name for name in candidates if detect_precision(name) == current_precision]
        if same:
            return same[0]

    for precision in PRECISION_ORDER:
        hit = [name for name in candidates if detect_precision(name) == precision]
        if hit:
            return hit[0]
    return candidates[0]


def pick_onnx(directory, current_name):
    """
    在 directory 下挑一个 .onnx：优先 op11（与导出引擎时的选择保持一致），
    其次沿用当前条目的 opset，最后退回排序第一个。
    """
    candidates = list_model_files(directory, ".onnx")
    if not candidates:
        return None

    op11 = [name for name in candidates if "op11" in name]
    if op11:
        return op11[0]

    current_op = detect_op(current_name)
    if current_op:
        same = [name for name in candidates if detect_op(name) == current_op]
        if same:
            return same[0]
    return candidates[0]


def refresh_model_paths(lines, config_dir, project_root, fix=True):
    """
    扫描 config 行，检测每个 model path 条目。

    fix=True（refresh 子命令）：把路径改写为真实存在的文件；
    fix=False（gen 子命令）：只报告，不改写——config_jetson 的 trt8.2 之类的
    「目标环境意图」不由当前机器的文件系统裁决，改写只发生在目标机器的 refresh。

    返回 (new_lines, report_lines, changed_count)。
    报告标记: [OK] 路径已存在 / [FIX] 自动修正为真实文件 / [MISS] 文件不存在
    （保持原值）/ [KEEP] 目录缺失或无法解析，保持原值 / [WARN] path 行格式问题。
    """
    config_dir = os.path.abspath(config_dir)
    engine_root = os.path.join(project_root, "model", "engine")
    onnx_root = os.path.join(project_root, "model", "onnx")

    pending_type = None
    report = []
    changed = 0

    for index, line in enumerate(lines):
        match_type = TYPE_RE.match(line)
        if match_type:
            pending_type = match_type.group(1)
            continue

        match_path = PATH_RE.match(line)
        if not match_path:
            # 有 type 但 path 行写得不合规范（例如没有引号）时给个提示，避免静默漏掉
            if pending_type and re.match(r"^\s*path:", line):
                report.append("  [WARN] line %d: cannot parse path line, left as-is"
                              % (index + 1))
            continue
        if not pending_type:
            continue

        entry_type, pending_type = pending_type, None
        indent, old_value, tail = match_path.group(1), match_path.group(2), match_path.group(3)

        if entry_type not in ENGINE_TYPES + ONNX_TYPES:
            continue
        if not old_value.strip():
            report.append("  [KEEP] %-12s (empty path)" % entry_type)
            continue

        model_dir = os.path.basename(os.path.dirname(old_value))
        if not model_dir:
            report.append("  [WARN] line %d: no model dir in \"%s\", left as-is"
                          % (index + 1, old_value))
            continue

        if entry_type in ENGINE_TYPES:
            root, kind = engine_root, "engine"
            new_name = pick_engine(os.path.join(engine_root, model_dir),
                                   os.path.basename(old_value))
        else:
            root, kind = onnx_root, "onnx"
            new_name = pick_onnx(os.path.join(onnx_root, model_dir),
                                 os.path.basename(old_value))

        if not new_name:
            report.append("  [KEEP] %-12s %-18s no %s under model/%s/%s/"
                          % (entry_type, model_dir, kind, kind, model_dir))
            continue

        new_value = os.path.relpath(os.path.join(root, model_dir, new_name),
                                    config_dir).replace(os.sep, "/")

        if new_value == old_value:
            report.append("  [ OK ] %-12s %-18s %s" % (entry_type, model_dir, old_value))
            continue

        if not fix:
            report.append("  [MISS] %-12s %-18s %s\n         (%s file not found on this "
                          "machine, template path kept)"
                          % (entry_type, model_dir, old_value, kind))
            continue

        new_line = '%s"%s"%s' % (indent, new_value, tail)
        if not new_line.endswith("\n"):
            new_line += "\n"
        lines[index] = new_line
        changed += 1
        report.append("  [FIX ] %-12s %-18s %s\n         -> %s"
                      % (entry_type, model_dir, old_value, new_value))

    return lines, report, changed


def project_model_roots_exist(project_root):
    engine_root = os.path.join(project_root, "model", "engine")
    onnx_root = os.path.join(project_root, "model", "onnx")
    return os.path.isdir(engine_root) or os.path.isdir(onnx_root)

# ============================================================================
# gen：三份配置的模板与参数（保留注释，不依赖 PyYAML）
# ============================================================================

TEMPLATE = """\
# 本文件由 scripts/config_tool.py gen 自动生成，环境差异见脚本内 VARIANTS
display_manager:
  is_display: {is_display}    # 是否启用本机的 GUI 交互窗口
  enable_control_panel: {enable_control_panel}   # 是否启用运行时参数控制面板（纯滑动条，拖动实时生效，需 is_display 开启）

prefer:
  use_gpu: {use_gpu}      # 是否使用 GPU 加速推理，若 false 则强制使用 CPU 后端
{backend_hint}  overlap: {overlap}      # 使用TensorRT作为后端时，若为True，则使用流水线重叠推理，若 false 则强制使用串行推理
  stagger_infer: {stagger_infer} # 错峰推理总开关：按各自间隔调度检测/深度推理，两路同帧时走重叠推理（开启后优先于 overlap）

yolo:
  detect_interval: 1   # 检测推理间隔：1=每帧，2=隔帧，3=隔2帧（错峰模式生效）
  yolo_model_path:
    - type: engine
      path: "../model/engine/yolo26s/yolo26s_640_op11_fp16_{trt_ver}.engine"   # 主模型（非碰撞帧使用）
    - type: onnx
      path: "../model/onnx/yolo26n/yolo26n_640_op11.onnx"
    - type: light_engine
      path: "../model/engine/yolo26n/yolo26n_640_op11_fp16_{trt_ver}.engine"   # 轻量模型：错峰碰撞帧重叠推理专用；删除则复用主模型
{light_onnx_entry}
  yolo_nms_thresh: 0.6    # YOLO NMS阈值 (下采样截断遮挡框)
  yolo_conf_thresh: 0.4

depth:
  enabled : true
  depth_interval: 3       # 深度推理间隔：1=每帧，2=隔帧，3=隔2帧（语义同 yolo.detect_interval，错峰模式生效）
  model_type: "yolo_depth"   # 选项：lite_mono 或 yolo_depth
  depth_model_path:
    - type: engine
      path: "../model/engine/yolo26n-depth/yolo26n-depth_640_op11_fp16_{trt_ver}.engine"
    - type: onnx
      path: "../model/onnx/yolo26n-depth/yolo26n-depth_640_op11.onnx"

motion_state_engine:      # 运动/危险判定引擎：只保留 approach（快速靠近）一路，见 cpp/core/include/motion_state_engine.h
  approach:
    enabled: true          # 是否启用快速靠近检测（关闭后不动任何既有行为）
    filter: "kalman"     # area/raw_depth 因果滤波: none / one_euro(1€) / kalman(一维匀速)
    detect_warmup: 5      # 框高（检测）通道基线攒帧数
    detect_recent_w: 10   # 框高通道近期趋势窗口长度（前后半窗各 recent_w/2）
    depth_warmup: 3       # 深度（模型）通道基线攒帧数
    depth_recent_w: 6     # 深度通道近期趋势窗口长度
    thr_depth: 0.15        # 相对基线的深度降幅阈值
    thr_height: 0.15         # 相对基线的框高增幅阈值
    score_thr: 0.45        # 进入分数线
    confirm: 2             # 进入需连续达标帧数
    exit_score_thr: 0.2    # 退出分数线
    exit_confirm: 2        # 退出需连续证据帧数

danger_alert:
  is_filter_small_objects: true   # 是否过滤小目标（快速靠近的目标若面积过小则不上报）
  min_object_area: 100              # 小目标过滤阈值


camera:
  width: 1280    # 相机采集宽度（USB 相机索引模式时生效）
  height: 720    # 相机采集高度
  fps: 30        # 相机采集帧率


io_manager:
  out_dir: "out_dir"      # 落盘文件夹名称
  save_track_log: {save_track_log}       # 是否保存每个 track 的类别/原始深度/面积/帧数等到 CSV
  save_mode: images        # 保存输出: video / images / both / none
  save_buffer_gb: 1.0     # 异步落盘缓冲区上限（GB）；写满时丢新帧并计数告警（不阻塞主流程），退出时自动排空不丢数据
  send_tcp: true          # 是否发送TCP数据包
  send_tcp_ip: "192.168.43.1"  # 发送TCP数据包的IP地址
  send_tcp_port: 5005     # 发送TCP数据包的端口号
  tcp_reconnect: true     # 断联自动重连；false = 保持旧行为（失败即停发，不重试）
  tcp_check_interval_s: 5 # 看门狗探活/重连周期（秒）
  tcp_connect_timeout_s: 3 # 单次非阻塞 connect 的超时上限（秒）
  simulate_delay: {simulate_delay}    # 是否按视频自身帧率模拟实时节奏（快了等待、慢了跳帧；仅视频文件源有效，跑满算力压测时改 false）
  simulate_fps: 12        # 模拟的现场帧率（仅 simulate_delay 开启时生效）；0 = 跟随视频文件自身的 fps

logger:
  save_file: true         # 是否保存日志文件到logs文件夹
  console_output: true    # 是否在终端显示日志
  log_level: info          # 日志级别: trace / debug / info / warn / err / critical

bytetrack:
  track_high_thresh: 0.50
  track_low_thresh: 0.1
  new_track_thresh: 0.60
  track_buffer: 30
  match_thresh: 0.80
"""

LIGHT_ONNX_ENTRY = (
    "    - type: light_onnx\n"
    '      path: "../model/onnx/yolo26n/yolo26n_640_op11.onnx"   '
    "# 轻量模型：错峰碰撞帧重叠推理专用；删除则复用主模型\n"
)
BACKEND_HINT = (
    "  # backend: auto    # 后端选择：auto(缺省，TensorRT 优先 + ONNX 兜底) / "
    "tensorrt / onnxruntime（显式指定失败不回退）\n"
)

VARIANTS = {
    # 开发/桌面环境：GUI + TensorRT（TRT 10.9 engine）+ 错峰推理
    "config": dict(
        is_display="true",
        enable_control_panel="true",
        use_gpu="true",
        backend_hint=BACKEND_HINT,
        overlap="true",
        stagger_infer="true",
        trt_ver="trt10.9",
        light_onnx=True,
        save_track_log="true",
        simulate_delay="true",
    ),
    # Jetson 部署环境：headless + TRT 8.2 engine（JetPack 自带 TensorRT 8.x）
    "config_jetson": dict(
        is_display="false",
        enable_control_panel="true",
        use_gpu="true",
        backend_hint="",
        overlap="true",
        stagger_infer="true",
        trt_ver="trt8.2",
        light_onnx=True,
        save_track_log="false",
        simulate_delay="false",
    ),
    # CPU-only CI 环境：ONNX Runtime 兜底，无 GPU、无轻量模型重叠链路
    "config_ci": dict(
        is_display="false",
        enable_control_panel="true",
        use_gpu="false",
        backend_hint="",
        overlap="true",
        stagger_infer="false",
        trt_ver="trt10.9",
        light_onnx=False,
        save_track_log="false",
        simulate_delay="false",
    ),
}


def render(variant):
    text = TEMPLATE
    if variant["light_onnx"]:
        text = text.replace("{light_onnx_entry}", LIGHT_ONNX_ENTRY)
    else:
        text = text.replace("{light_onnx_entry}", "")
    return text.format(**variant)


def cmd_gen(args):
    project_root = os.path.abspath(args.project_root) if args.project_root else os.path.abspath(
        os.path.join(os.path.dirname(__file__), ".."))
    out_dir = os.path.abspath(args.out_dir) if args.out_dir else os.path.join(
        project_root, "bin")
    try:
        os.makedirs(out_dir, exist_ok=True)
    except OSError as e:
        print("ERROR: cannot create output dir %s: %s" % (out_dir, e), file=sys.stderr)
        return 1

    check_paths = project_model_roots_exist(project_root)
    if not check_paths:
        print("  [WARN] neither model/engine/ nor model/onnx/ exists — "
              "model paths are generated as-is without existence check")

    names = [args.variant] if args.variant else sorted(VARIANTS)
    for name in names:
        lines = render(VARIANTS[name]).splitlines(keepends=True)

        report = []
        if check_paths:
            # fix=False：gen 只检测并报告，不改写——模板里的 trt8.2（Jetson 意图）等
            # 目标环境路径不会被本机的文件系统覆盖；改写是目标机器上 refresh 的事
            _, report, _ = refresh_model_paths(lines, out_dir, project_root, fix=False)

        dst = os.path.join(out_dir, "%s.yaml" % name)
        with open(dst, "w", encoding="utf-8", newline="") as f:
            f.writelines(lines)
        print("wrote %s" % dst)
        for item in report:
            print(item)
    return 0

# ============================================================================
# read：预检 + 运行时信息导出（原 read_config.py，start.sh 依赖其输出与退出码）
# ============================================================================


def load_yaml(path):
    """读取 YAML；PyYAML 缺失时返回 None（由调用方决定降级）。"""
    try:
        import yaml
    except ImportError:
        return None
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def cfg_get(cfg, *keys, default=None):
    """按嵌套 key 取值，任一层缺失则返回 default。"""
    cur = cfg
    for key in keys:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def resolve_path(bin_dir, path):
    """把 config 里的相对路径（以 bin/ 为基准）解析为绝对路径。"""
    text = str(path or "").strip()
    if not text:
        return ""
    return text if os.path.isabs(text) else os.path.normpath(os.path.join(bin_dir, text))


def find_onnx(project_root, model_dir):
    """在 model/onnx/<模型目录>/ 下挑 ONNX，优先 op11；找不到返回 ""。"""
    onnx_dir = os.path.join(project_root, "model", "onnx", model_dir)
    candidates = list_model_files(onnx_dir, ".onnx")
    if not candidates:
        return ""
    preferred = [name for name in candidates if "op11" in name]
    return os.path.join(onnx_dir, (preferred or candidates)[0])


def collect_export_models(cfg, bin_dir, project_root):
    """
    推导需要导出 engine 的模型清单。
    返回 [(模型目录, preprocess 模式, onnx 绝对路径), ...]，已去重且保持 config 顺序。
    """
    models = []
    seen = set()

    if not cfg_get(cfg, "depth", "enabled", default=True):
        # 深度关闭时只处理 yolo
        sections = (("yolo", "yolo_model_path"),)
    else:
        sections = (("yolo", "yolo_model_path"), ("depth", "depth_model_path"))

    for section, key in sections:
        for item in (cfg_get(cfg, section, key) or []):
            if not isinstance(item, dict):
                continue
            if str(item.get("type", "")) not in ("engine", "light_engine"):
                continue

            model_dir = os.path.basename(
                os.path.dirname(resolve_path(bin_dir, item.get("path", ""))))
            if not model_dir or model_dir in seen:
                continue
            seen.add(model_dir)

            # preprocess 模式：yolo* 走 letterbox + RGB/255，其余走 depth 归一化
            preprocess = "yolo" if model_dir.startswith("yolo") else "depth"
            models.append((model_dir, preprocess, find_onnx(project_root, model_dir)))

    return models


def write_headless_config(src, dst):
    """生成 headless 副本：只替换 is_display 行，保留缩进与行尾注释。"""
    out = []
    with open(src, "r", encoding="utf-8") as f:
        for line in f:
            stripped = line.lstrip()
            if stripped.startswith("is_display:"):
                indent = line[: len(line) - len(stripped)]
                comment = ""
                if "#" in stripped:
                    comment = "  #" + stripped.split("#", 1)[1].rstrip()
                line = "%sis_display: false%s\n" % (indent, comment)
            out.append(line)
    with open(dst, "w", encoding="utf-8") as f:
        f.writelines(out)


def cmd_read(args):
    project_root = os.path.abspath(args.project_root) if args.project_root else os.path.abspath(
        os.path.join(os.path.dirname(__file__), ".."))
    config_path = os.path.abspath(args.config) if args.config else os.path.join(
        project_root, "bin", "config.yaml")

    if not os.path.isfile(config_path):
        print("ERROR: config.yaml not found: %s" % config_path, file=sys.stderr)
        return 2

    # ---- 模式 1: 生成 headless 副本（纯文本行替换，不需要 PyYAML）----
    if args.write_headless:
        dst = os.path.abspath(args.write_headless)
        parent = os.path.dirname(dst)
        if parent:
            os.makedirs(parent, exist_ok=True)
        write_headless_config(config_path, dst)
        print("wrote %s" % dst)
        return 0

    # ---- 模式 2: 预检 + 模型清单 ----
    # PyYAML 缺失属于「可预期的降级态」，由调用方（start.sh）给出提示，
    # 这里保持静默，只以退出码 3 告知（避免同一句提示重复打印）。
    cfg = load_yaml(config_path)
    if cfg is None:
        return 3

    bin_dir = os.path.dirname(config_path)

    print("CFG_USE_GPU=%s" % ("true" if cfg_get(cfg, "prefer", "use_gpu", default=True) else "false"))
    print("CFG_IS_DISPLAY=%s" % ("true" if cfg_get(
        cfg, "display_manager", "is_display", default=True) else "false"))
    print("CFG_MODEL_TYPE=%s" % cfg_get(cfg, "depth", "model_type", default="lite_mono"))
    print("CFG_DEPTH_ENABLED=%s" % ("true" if cfg_get(
        cfg, "depth", "enabled", default=True) else "false"))
    print("CFG_SIMULATE_DELAY=%s" % ("true" if cfg_get(
        cfg, "io_manager", "simulate_delay", default=False) else "false"))
    print("CFG_SEND_TCP=%s" % ("true" if cfg_get(
        cfg, "io_manager", "send_tcp", default=False) else "false"))
    print("CFG_TCP_TARGET=%s:%s" % (
        cfg_get(cfg, "io_manager", "send_tcp_ip", default="-"),
        cfg_get(cfg, "io_manager", "send_tcp_port", default="-")))

    for model_dir, preprocess, onnx_path in collect_export_models(cfg, bin_dir, project_root):
        print("EXPORT_MODEL=%s\t%s\t%s" % (model_dir, preprocess, onnx_path))

    return 0

# ============================================================================
# refresh：就地刷新 config 中的模型路径（原 update_config.py）
# ============================================================================


def cmd_refresh(args):
    project_root = os.path.abspath(args.project_root) if args.project_root else os.path.abspath(
        os.path.join(os.path.dirname(__file__), ".."))
    config_path = os.path.abspath(args.config) if args.config else os.path.join(
        project_root, "bin", "config.yaml")

    if not os.path.isfile(config_path):
        print("ERROR: config not found: %s" % config_path, file=sys.stderr)
        return 2

    if not project_model_roots_exist(project_root):
        print("  [WARN] neither model/engine/ nor model/onnx/ exists, config left untouched")
        return 0

    with open(config_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    lines, report, changed = refresh_model_paths(
        lines, os.path.dirname(config_path), project_root)

    for item in report:
        print(item)

    if changed and not args.dry_run:
        with open(config_path, "w", encoding="utf-8", newline="") as f:
            f.writelines(lines)

    print("  [%s] %d change(s)%s"
          % ("DRY-RUN" if args.dry_run else "DONE", changed,
             " (nothing written)" if args.dry_run else ""))
    return 0

# ============================================================================
# CLI
# ============================================================================


def main():
    parser = argparse.ArgumentParser(
        prog="config_tool.py",
        description="Generate / inspect / refresh the bin/config*.yaml files")
    sub = parser.add_subparsers(dest="command", required=True)

    p_gen = sub.add_parser("gen", help="generate config.yaml / config_jetson.yaml / config_ci.yaml "
                                       "with model-path existence check")
    p_gen.add_argument("--out-dir", default=None,
                       help="output directory (default: <project-root>/bin)")
    p_gen.add_argument("--variant", choices=sorted(VARIANTS), default=None,
                       help="generate a single variant instead of all three")
    p_gen.add_argument("--project-root", default=None,
                       help="project root (default: parent of this script's directory)")

    p_read = sub.add_parser("read", help="print KEY=VALUE runtime info for start.sh")
    p_read.add_argument("--config", default=None,
                        help="config path (default: <project-root>/bin/config.yaml)")
    p_read.add_argument("--project-root", default=None,
                        help="project root (default: parent of this script's directory)")
    p_read.add_argument("--write-headless", default=None, metavar="DST",
                        help="write a copy with display_manager.is_display=false to DST, then exit")

    p_refresh = sub.add_parser("refresh", help="refresh model paths in a config in place")
    p_refresh.add_argument("--config", default=None,
                           help="config path (default: <project-root>/bin/config.yaml)")
    p_refresh.add_argument("--project-root", default=None,
                           help="project root (default: parent of this script's directory)")
    p_refresh.add_argument("--dry-run", action="store_true",
                           help="print the planned changes without writing anything")

    args = parser.parse_args()
    return {"gen": cmd_gen, "read": cmd_read, "refresh": cmd_refresh}[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
