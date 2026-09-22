#!/usr/bin/env python3
"""
刷新 bin/config.yaml 中的模型路径，使其与 model/engine/<模型>/ 和
model/onnx/<模型>/ 下真实存在的文件一致。

与上游 model/update_config.py 的区别（本脚本是父仓库自有的，不要放回 model/ 子模块，
否则 start.sh 第 1 步的 `git submodule update --force` 会把改动重置掉）：

  1. 模型身份不做任何猜测。每个条目的模型目录直接取自它**自己当前 path** 的父目录名，
     并且只用同一目录下的文件来刷新它。因此 depth_model_path 里写的是 yolo26n-depth，
     就永远只会被刷成 yolo26n-depth，不会被悄悄换成扫描到的第一个深度模型（lite-mono-tiny）。
  2. engine / light_engine / onnx / light_onnx 四种类型全部处理
     （上游版本完全不认 light_* 条目，导致 light 路径永久失修）。
  3. 写回的路径一律相对 config 文件所在目录（bin/），保持 config.yaml 跨机器/容器可移植，
     不会再写成 /workspace/... 这类绝对路径。
  4. 只替换 `path:` 行引号内的字符串，缩进与行尾注释逐字节保留。
  5. 目录里找不到可用文件时保持原值（[KEEP]），绝不写坏 config。

用法:
    python3 scripts/update_config.py                       # 就地刷新
    python3 scripts/update_config.py --dry-run             # 只打印将要做的改动
    python3 scripts/update_config.py --config bin/config.yaml --project-root .

退出码:
    0  成功（含"无需改动"）
    1  参数/路径错误
    2  config 文件不存在
"""

import argparse
import os
import re
import sys

ENGINE_TYPES = ("engine", "light_engine")
ONNX_TYPES = ("onnx", "light_onnx")
PRECISION_ORDER = ("fp16", "fp32", "int8")

TYPE_RE = re.compile(r"^\s*-\s*type:\s*([A-Za-z_][A-Za-z0-9_]*)\s*(?:#.*)?$")
PATH_RE = re.compile(r'^(\s*path:\s*)"([^"]*)"(.*)$')


def detect_precision(name):
    """从文件名推断精度；engine 文件名约定为 ..._<精度>_trtX.Y.engine。"""
    for precision in ("int8", "fp16", "fp32"):
        if precision in name:
            return precision
    return None


def detect_op(name):
    """从文件名推断 onnx opset 标记，如 op11。"""
    match = re.search(r"(op\d+)", name)
    return match.group(1) if match else None


def pick_engine(directory, current_name):
    """
    在 directory 下挑一个 .engine：
      1) 优先沿用当前条目已有的精度（身份不变，避免自己改自己的配置）
      2) 否则按 fp16 > fp32 > int8 取最优
    """
    if not os.path.isdir(directory):
        return None
    candidates = sorted(
        name for name in os.listdir(directory)
        if name.endswith(".engine") and os.path.isfile(os.path.join(directory, name))
    )
    if not candidates:
        return None

    current_precision = detect_precision(current_name)
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
    在 directory 下挑一个 .onnx：优先 op11（与 read_config.py 的导出选择保持一致），
    其次沿用当前条目的 opset，最后退回排序第一个。
    """
    if not os.path.isdir(directory):
        return None
    candidates = sorted(
        name for name in os.listdir(directory)
        if name.endswith(".onnx") and os.path.isfile(os.path.join(directory, name))
    )
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


def refresh(config_path, project_root, dry_run):
    config_dir = os.path.dirname(os.path.abspath(config_path))
    engine_root = os.path.join(project_root, "model", "engine")
    onnx_root = os.path.join(project_root, "model", "onnx")

    with open(config_path, "r", encoding="utf-8") as handle:
        lines = handle.readlines()

    pending_type = None
    actions = []
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
                actions.append("  [WARN] line %d: cannot parse path line, left as-is" % (index + 1))
            continue
        if not pending_type:
            continue

        entry_type, pending_type = pending_type, None
        indent, old_value, tail = match_path.group(1), match_path.group(2), match_path.group(3)

        if entry_type not in ENGINE_TYPES + ONNX_TYPES:
            continue
        if not old_value.strip():
            actions.append("  [KEEP] %-12s (empty path)" % entry_type)
            continue

        model_dir = os.path.basename(os.path.dirname(old_value))
        if not model_dir:
            actions.append("  [WARN] line %d: no model dir in \"%s\", left as-is"
                           % (index + 1, old_value))
            continue

        if entry_type in ENGINE_TYPES:
            root, new_name = engine_root, pick_engine(
                os.path.join(engine_root, model_dir), os.path.basename(old_value))
            kind = "engine"
        else:
            root, new_name = onnx_root, pick_onnx(
                os.path.join(onnx_root, model_dir), os.path.basename(old_value))
            kind = "onnx"

        if not new_name:
            actions.append("  [KEEP] %-12s %-18s no %s under model/%s/%s/"
                           % (entry_type, model_dir, kind, kind, model_dir))
            continue

        new_abs = os.path.join(root, model_dir, new_name)
        new_value = os.path.relpath(new_abs, config_dir).replace(os.sep, "/")

        if new_value == old_value:
            actions.append("  [ OK ] %-12s %-18s %s" % (entry_type, model_dir, old_value))
            continue

        new_line = '%s"%s"%s' % (indent, new_value, tail)
        if not new_line.endswith("\n"):
            new_line += "\n"
        lines[index] = new_line
        changed += 1
        actions.append("  [UPD ] %-12s %-18s %s\n         -> %s"
                       % (entry_type, model_dir, old_value, new_value))

    for action in actions:
        print(action)

    if changed and not dry_run:
        with open(config_path, "w", encoding="utf-8", newline="") as handle:
            handle.writelines(lines)

    print("  [%s] %d change(s)%s"
          % ("DRY-RUN" if dry_run else "DONE", changed,
             " (nothing written)" if dry_run else ""))
    return 0


def main():
    parser = argparse.ArgumentParser(
        description="Refresh model paths in bin/config.yaml (identity preserving)")
    parser.add_argument("--config", default=None,
                        help="config.yaml path (default: <project-root>/bin/config.yaml)")
    parser.add_argument("--project-root", default=None,
                        help="project root (default: parent of this script's directory)")
    parser.add_argument("--dry-run", action="store_true",
                        help="print the planned changes without writing anything")
    args = parser.parse_args()

    project_root = os.path.abspath(args.project_root) if args.project_root else os.path.abspath(
        os.path.join(os.path.dirname(__file__), ".."))
    config_path = os.path.abspath(args.config) if args.config else os.path.join(
        project_root, "bin", "config.yaml")

    if not os.path.isfile(config_path):
        print("ERROR: config.yaml not found: %s" % config_path, file=sys.stderr)
        return 2

    engine_root = os.path.join(project_root, "model", "engine")
    onnx_root = os.path.join(project_root, "model", "onnx")
    if not os.path.isdir(engine_root) and not os.path.isdir(onnx_root):
        print("  [WARN] neither %s nor %s exists, config left untouched"
              % (engine_root, onnx_root))
        return 0

    return refresh(config_path, project_root, args.dry_run)


if __name__ == "__main__":
    sys.exit(main())
