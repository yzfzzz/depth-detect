#!/usr/bin/env python3
"""
读取 bin/config.yaml，向 start.sh 提供运行时信息。

用法:
    # 1) 预检 + 推导待导出 engine 的模型清单（stdout 输出，供 shell 解析）
    python3 scripts/read_config.py --config bin/config.yaml --project-root .

    # 2) 生成 headless 副本（只把 display_manager.is_display 改成 false，
    #    保留缩进与行尾注释，不动原文件）
    python3 scripts/read_config.py --config bin/config.yaml --write-headless bin/.config.headless.yaml

输出格式（KEY=VALUE，一行一条）:
    CFG_USE_GPU=true|false
    CFG_IS_DISPLAY=true|false
    CFG_MODEL_TYPE=<lite_mono|yolo_depth|...>
    CFG_DEPTH_ENABLED=true|false
    CFG_SIMULATE_DELAY=true|false
    CFG_SEND_TCP=true|false
    CFG_TCP_TARGET=<ip>:<port>
    EXPORT_MODEL=<模型目录>\t<preprocess>\t<onnx 绝对路径>

模型清单的推导规则:
    只取 config 中 type=engine / light_engine 的条目（type=onnx 是 CPU 兜底，无需导出）。
    模型身份 = 该条目的 path 所在的父目录名（例如 ../model/engine/yolo26n-depth/xxx.engine
    → 模型目录 yolo26n-depth），再据此到 model/onnx/<同名目录>/ 找 ONNX（优先 op11）。
    preprocess 规则：目录名以 yolo 开头 → yolo（letterbox + RGB/255），
    其余（lite-mono 等）→ depth。

退出码:
    0  成功
    1  其他错误
    2  config 文件不存在
    3  PyYAML 不可用（调用方应降级为 ONNX/CPU 路径）
"""

import argparse
import os
import sys


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
    if not os.path.isdir(onnx_dir):
        return ""
    cands = sorted(name for name in os.listdir(onnx_dir) if name.endswith(".onnx"))
    if not cands:
        return ""
    preferred = [name for name in cands if "op11" in name]
    return os.path.join(onnx_dir, (preferred or cands)[0])


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

            model_dir = os.path.basename(os.path.dirname(resolve_path(bin_dir, item.get("path", ""))))
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


def main():
    parser = argparse.ArgumentParser(description="Read config.yaml and export runtime info")
    parser.add_argument("--config", default=None,
                        help="config.yaml path (default: <project-root>/bin/config.yaml)")
    parser.add_argument("--project-root", default=None,
                        help="project root (default: parent of this script's directory)")
    parser.add_argument("--write-headless", default=None, metavar="DST",
                        help="write a copy with display_manager.is_display=false to DST, then exit")
    args = parser.parse_args()

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


if __name__ == "__main__":
    sys.exit(main())
