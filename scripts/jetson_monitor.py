#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Jetson(TX2) 资源监视器：采内存 / 功耗 / 温度，一直采到目标进程退出。

目标进程两种来源、输出三档，可任意组合：

  # 1) 附着：监视已在运行的进程（PID，或唯一匹配的进程名），采到它退出
  python3 scripts/jetson_monitor.py --pid 12345 --plot runs/mem_temp.png
  python3 scripts/jetson_monitor.py --pid main  --plot runs/mem_temp.png

  # 2) 拉起：自行 fork 被测程序，采集窗口与它的生命周期严格对齐
  python3 scripts/jetson_monitor.py --plot runs/mem_temp.png \\
      --exec "cd bin && ./main ../data/dd/905-3-1.mp4 config.yaml"

  # 3) 仅需原始数据、不画图（供 pandas / gnuplot / Excel 二次加工）
  python3 scripts/jetson_monitor.py --json runs/906-2-1.json --exec "..."

输出：终端统计（各指标均值 / 峰值 / 峰现时刻 / 最小值）始终打印；--plot 写 PNG，
--json 写原始序列；两者都不传时仅输出终端统计。
"""
import argparse
import difflib
import glob
import json
import os
import re
import shlex
import shutil
import signal
import subprocess
import sys
import time
from collections import OrderedDict

VERSION = "1.0"
PALETTE = ("#d62728", "#ff7f0e", "#1f77b4", "#9467bd", "#2ca02c", "#8c564b")
SHELL_NAMES = ("sh", "dash", "bash", "ash", "zsh", "ksh", "busybox")  # 仅作 wrapper，不计入负载
POWER_GLOBS = (  # 开销低：单层通配优先，命中即返回
    "/sys/bus/i2c/devices/*-004*/iio_device/in_power*_input",
    "/sys/bus/i2c/devices/*-004*/iio:device*/in_power*_input",
)
POWER_GLOBS_DEEP = (  # 兜底：`**` 会遍历整个 sysfs（Jetson 上需数秒），以上均未命中时才用
    "/sys/devices/**/i2c-*/*-004*/iio*/in_power*_input",
)
TEMP_MIN_MC = -200000  # 低于此值视为该温区无效

# 仅对扩展名在白名单内的参数做存在性体检：无法识别的一律按普通参数处理
ARG_FILE_EXTS = (".yaml", ".yml", ".json", ".mp4", ".avi", ".mkv", ".mov", ".jpg", ".jpeg",
                 ".png", ".engine", ".onnx", ".plan", ".bin", ".cfg", ".ini", ".list", ".txt")

# 解析 --exec 时遇到这些词即放弃静态判断：其后多为 shell 展开结果，无法检查
SHELL_WORDS = ("cd", "exec", "eval", "env", "nohup", "setsid", "time", "nice", "ionice",
               "sudo", "taskset", "stdbuf", "watch", "xargs", "sh", "bash", "python",
               "python3", "tee")
STOP = False   # 收到信号置位，采集循环据此收尾
_MONITOR = None  # 当前 Monitor，供信号处理器在强制退出前一并结束工作负载


# ------------------------------------------------------------------ 基础读取
def slurp(path):
    """读文本节点；读取失败（不存在 / 无权限）统一返回 None。"""
    try:
        with open(path) as fh:
            return fh.read().strip()
    except (IOError, OSError):
        return None


def slurp_int(path):
    """读整数节点，容忍 '1881600 kHz' 这类带单位写法。"""
    m = re.search(r"-?\d+", slurp(path) or "")
    return int(m.group()) if m else None


def kv_int(path, keys=None):
    """把 'Key: 1234 kB' 形式的文件解析成 {key: 1234}；keys 非空时仅收白名单内的键。"""
    out = {}
    for line in (slurp(path) or "").splitlines():
        key, sep, rest = line.partition(":")
        key = key.strip()
        if not sep or (keys and key not in keys) or not rest.split():
            continue
        try:
            out[key] = int(rest.split()[0])
        except ValueError:
            pass
    return out


def _collect_rails(root, pats):
    """按给定通配模式收集 INA3221 通道 -> {导轨名: 节点路径}。"""
    found = {}
    for pat in pats:
        for path in sorted(glob.glob(root + pat, recursive=True)):
            base = os.path.basename(path)
            parent = os.path.dirname(path)
            name = slurp(os.path.join(parent,
                                      base.replace("in_power", "rail_name").replace("_input", "")))
            if not name:  # 无 rail_name 时命名为「芯片_通道」，避免不同芯片的同名通道互相覆盖
                name = "%s_%s" % (os.path.basename(os.path.dirname(parent)), base)
            found.setdefault(name, path)
    return found


def power_rails(root, warn=None):
    """INA3221 通道 -> [(导轨名, 节点路径)]，节点单位 µW。

    单层通配优先，未命中再递归扫 /sys/devices：`**` 会遍历整个 sysfs，Jetson 上需数秒。
    """
    found = _collect_rails(root, POWER_GLOBS)
    if found:
        return sorted(found.items())
    if warn:
        warn("[INFO] 单层通配未命中功耗节点，改为递归扫 /sys/devices（可能需数秒）...\n")
    return sorted(_collect_rails(root, POWER_GLOBS_DEEP).items())


def thermal_zones(root):
    """温区 -> [(类型名, temp 节点路径)]，节点单位 m℃。"""
    return [(slurp(os.path.join(z, "type")) or os.path.basename(z), os.path.join(z, "temp"))
            for z in sorted(glob.glob(root + "/sys/devices/virtual/thermal/thermal_zone*"))]


def find_pid(root, name):
    """在 /proc 中按名字查进程，返回全部命中的 pid（去重由调用方负责）。

    三个匹配口径依次放宽：comm、argv[0] 的 basename、cmdline 任意子串。
    """
    me, hits = os.getpid(), []
    for ent in sorted(os.listdir(root + "/proc")):
        if not ent.isdigit() or int(ent) == me:
            continue
        comm = slurp("%s/proc/%s/comm" % (root, ent)) or ""
        argv = [a for a in (slurp("%s/proc/%s/cmdline" % (root, ent)) or "").split("\0") if a]
        if name == comm or (argv and os.path.basename(argv[0]) == name) \
                or any(name in a for a in argv):
            hits.append(int(ent))
    return hits


def proc_stat(root, pid):
    """一次读 /proc/<pid>/stat -> (状态, ppid, 进程组, utime + stime 滴答)；进程不存在时返回 None。

    comm 可能含空格与括号，必须从最后一个 ')' 之后切分。
    """
    txt = slurp("%s/proc/%d/stat" % (root, pid))
    if not txt:
        return None
    fields = txt[txt.rfind(")") + 1:].split()
    try:
        return fields[0], int(fields[1]), int(fields[2]), int(fields[11]) + int(fields[12])
    except (IndexError, ValueError):
        return None


def proc_state(root, pid):
    """只取状态字符（R/S/D/Z...），读取失败返回 None。"""
    st = proc_stat(root, pid)
    return st[0] if st else None


def alive(root, pid):
    return proc_state(root, pid) not in (None, "Z")


def group_members(root, pgid, skip=()):
    """同一进程组里仍存活（非僵尸）的 (pid, comm)，按 pid 升序；skip 中的 pid 跳过。"""
    if not os.path.isdir(root + "/proc"):
        return []
    out = []
    for ent in sorted(os.listdir(root + "/proc")):
        if not ent.isdigit():
            continue
        pid = int(ent)
        if pid in skip:
            continue
        st = proc_stat(root, pid)
        if st is None or st[2] != pgid or st[0] == "Z":
            continue
        out.append((pid, (slurp("%s/proc/%d/comm" % (root, pid)) or "").strip()))
    return out


def clk_tck():
    """每秒的时钟滴答数（SC_CLK_TCK），取不到时用 Linux 默认值 100。"""
    try:
        return float(os.sysconf("SC_CLK_TCK"))
    except (AttributeError, ValueError, OSError):
        return 100.0


# ------------------------------------------------------------------ JSON 落盘
def iso_time(wall):
    """本地时间的 ISO8601，能取到时区偏移则一并输出。"""
    lt = time.localtime(wall)
    return time.strftime("%Y-%m-%dT%H:%M:%S", lt) + time.strftime("%z", lt)


def json_array(vals, indent=0, per_line=16):
    """数组每行 per_line 个折行：单行过长不便阅读，逐值一行又会产生数千行。"""
    if not vals:
        return "[]"
    pad = " " * indent
    rows = [", ".join(json.dumps(v, ensure_ascii=False) for v in vals[i:i + per_line])
            for i in range(0, len(vals), per_line)]
    return "[\n" + ",\n".join(pad + r for r in rows) + "\n" + pad[:-2] + "]"


def json_dump(obj, indent=0):
    if isinstance(obj, dict):
        if not obj:
            return "{}"
        body = ",\n".join("%s%s: %s" % (" " * (indent + 2), json.dumps(k, ensure_ascii=False),
                                        json_dump(v, indent + 2)) for k, v in obj.items())
        return "{\n%s\n%s}" % (body, " " * indent)
    if isinstance(obj, (list, tuple)):
        return json_array(list(obj), indent=indent + 2)
    return json.dumps(obj, ensure_ascii=False)


# ------------------------------------------------------------------ --exec 体检
def split_command(cmd, base=None):
    """拆 --exec 命令 -> (词元, 程序起始下标, 生效工作目录)。

    识别开头的 `cd DIR && ...` / `cd DIR; ...`：程序在哪个目录下执行，决定其相对路径能否找到。
    """
    try:
        toks = shlex.split(cmd)
    except ValueError:  # 引号未闭合等异常输入退回简单拆分，静态体检不阻断执行
        toks = cmd.split()
    base = os.path.abspath(base or os.getcwd())
    idx = 0
    if toks[:1] == ["cd"]:
        sep = next((i for i, t in enumerate(toks) if t in ("&&", ";")), -1)
        if 0 < sep and len(toks) > sep + 1 and len(toks) > 2:
            base = os.path.normpath(os.path.join(base, os.path.expanduser(toks[1])))
            idx = sep + 1
    return toks, idx, base


def suggest_missing_space(tok, base):
    """「./main../data/dd/x.mp4」这类缺少空格的粘连：在 `..` 处切分，前半段为真实文件时给出提示。"""
    for i in range(2, len(tok) - 1):
        if tok[i:i + 2] != "..":
            continue
        head = tok[:i]
        if head and os.path.isfile(os.path.join(base, head)):
            return "%s %s" % (head, tok[i:])
    return None


def preflight(cmd, base=None):
    """启动前静态体检 --exec 命令 -> (问题列表, 程序是否解析成功)。

    相对路径按 monitor 当前目录解析，写错会使目标立即退出、采样窗口近乎为空，故提前拦截并给出相近文件名。
    """
    issues = []
    toks, idx, cwd = split_command(cmd, base)
    if not os.path.isdir(cwd):  # 常见于已在 bin/ 下仍写 cd bin，或目录名拼写错误
        return ["cd 的目标目录不存在：%s" % cwd], False
    prog, args = None, []
    while idx < len(toks):
        tok = toks[idx]
        if tok in ("&&", ";", "||", "|"):
            idx += 1
            continue
        if re.match(r"^[A-Za-z_][A-Za-z0-9_]*=", tok):  # VAR=VAL 前缀
            idx += 1
            continue
        if tok in SHELL_WORDS or not os.path.basename(tok):
            return [], True  # 交由 shell 展开的部分无法静态检查，直接放行
        prog, args = tok, toks[idx + 1:]
        break
    if prog is None:
        return ["--exec 未解析出任何命令"], False

    if "/" in prog:  # 带路径：一律相对 cwd 解析（与 sh 行为一致）
        path = os.path.normpath(os.path.join(cwd, os.path.expanduser(prog)))
        if not os.path.exists(path):
            issues.append("程序 %s 不存在（按工作目录 %s 解析为 %s）" % (prog, cwd, path))
            fix = suggest_missing_space(prog, cwd)
            if fix:
                issues.append("疑似缺少空格：改为 \"%s\"" % fix)
            return issues, False
        if not os.access(path, os.X_OK):
            issues.append("程序 %s 无执行权限：chmod +x %s" % (prog, path))
            return issues, False
    elif not shutil.which(prog):
        issues.append("命令 %s 不在 PATH 中，也不是当前目录下的文件" % prog)
        return issues, False

    for a in args:  # 参数中形似路径的一并检查是否存在（配置文件名拼写错误最常见）
        if a.startswith("-") or os.path.splitext(a)[1].lower() not in ARG_FILE_EXTS:
            continue
        t = os.path.normpath(os.path.join(cwd, os.path.expanduser(a)))
        if os.path.exists(t):
            continue
        near = difflib.get_close_matches(os.path.basename(a), os.listdir(
            os.path.dirname(t) if os.path.isdir(os.path.dirname(t)) else cwd), n=1, cutoff=0.5)
        issues.append("参数文件 %s 不存在（按 %s 解析）%s"
                      % (a, cwd, "；是否要用 %s ？" % near[0] if near else ""))
    return issues, True


# ------------------------------------------------------------------ 采集与统计
class Stats(object):
    """一条指标的时间序列。"""

    __slots__ = ("unit", "t", "v")

    def __init__(self, unit):
        self.unit, self.t, self.v = unit, [], []

    def add(self, t, value):
        """追加一个采样点：t 是采样起始后的秒数，value 一律存 float。"""
        self.t.append(t)
        self.v.append(float(value))

    # 统计量按需现算（一条序列仅数千点，无需维护增量状态）

    @property
    def mean(self):
        return sum(self.v) / len(self.v)

    @property
    def peak(self):
        return max(self.v)

    @property
    def peak_at(self):
        return self.t[self.v.index(self.peak)]

    @property
    def low(self):
        return min(self.v)


class Monitor(object):
    """采集循环：探测硬件节点 -> 解析目标进程 -> 按节拍采样 -> 收尾 -> 输出报告与产物。"""

    def __init__(self, args):
        """只做参数落地；真正的节点探测与进程解析都在 run() 里。"""
        self.a = args
        self.root = args.root or ""
        self.stats = {}          # key -> Stats，统计与落盘都按 key 排序，输出稳定
        self.rails = []
        self.zones = []
        self.pid = None          # 被监视的目标进程（拉起模式下为真正的负载，非 wrapper）
        self.child = None        # 拉起模式的 wrapper（shell=True 时是 sh）
        self.child_pgid = None   # wrapper 的进程组：setsid 后 == child.pid，供收尾与存活判定使用
        self.members = None      # 进程组成员缓存（扫 /proc 开销不低，每 1s 刷新一次）
        self.members_at = 0.0
        self.seen_member = False  # 是否在组中见过负载：未见过时不可用「组空」作为退出依据
        self.reason = None       # 收尾原因，进终端报告与 JSON
        self.killed_by_us = False  # 工作负载由本程序主动结束（此时不应报「秒退」）
        self.json_path = args.json  # --json 的落盘路径（None = 不落盘）
        self.start_wall = self.end_wall = time.time()
        self.live = bool(args.live if args.live is not None else sys.stdout.isatty())
        self.prev_ticks = None   # (时刻, 目标进程 cpu ticks)
        self.prev_stat = None    # (total, idle)，--extras 用

    # --- 单次采样 ---

    def add(self, key, unit, value, t):
        """记账入口：value 为 None 表示本次未采到，直接丢弃不留空位。"""
        if value is not None:
            self.stats.setdefault(key, Stats(unit)).add(t, value)

    def sample(self, t):
        """一次采样：内存 / 功耗 / 温度 / 目标进程 RSS+CPU，再叠加 --extras 与实时状态行。"""
        mi = kv_int(self.root + "/proc/meminfo")
        total, avail = mi.get("MemTotal"), mi.get("MemAvailable", mi.get("MemFree"))
        if total and avail is not None:
            self.add("mem_used_mb", "MB", (total - avail) / 1024.0, t)

        rails = {}
        for name, path in self.rails:
            uw = slurp_int(path)
            if uw is not None:
                rails[name] = uw / 1000.0
                self.add("power_" + name, "mW", uw / 1000.0, t)
        if rails:  # 存在 VDD_IN 这类总输入通道时只取它，否则退化为三路求和
            main = [v for n, v in rails.items() if n.upper() in ("VDD_IN", "VIN", "VDD_IN_SYS")]
            self.add("power_total_mw", "mW", main[0] if main else sum(rails.values()), t)

        for name, path in self.zones:
            mc = slurp_int(path)
            if mc is not None and mc > TEMP_MIN_MC:
                self.add("temp_" + name, "C", mc / 1000.0, t)

        if self.pid is not None:
            status = kv_int("%s/proc/%d/status" % (self.root, self.pid), ("VmRSS", "VmHWM"))
            if "VmRSS" in status:
                self.add("proc_rss_mb", "MB", status["VmRSS"] / 1024.0, t)
            if "VmHWM" in status:
                self.add("proc_hwm_mb", "MB", status["VmHWM"] / 1024.0, t)
            ticks = proc_stat(self.root, self.pid)
            ticks = ticks[3] if ticks else None
            if ticks is not None:
                if self.prev_ticks is not None:  # CPU 占用只能靠两次 ticks 差分
                    dt = t - self.prev_ticks[0]
                    if dt > 0:
                        self.add("proc_cpu_pct", "%",
                                 100.0 * (ticks - self.prev_ticks[1]) / (dt * clk_tck()), t)
                self.prev_ticks = (t, ticks)

        if self.a.extras:
            self.sample_extras(t)
        if self.live:
            self.status_line(t)

    def sample_extras(self, t):
        """CPU 总占用（/proc/stat 差分）与 GPU 占用（gpu.0/load，TX2 上是千分比）。"""
        lines = (slurp(self.root + "/proc/stat") or "").splitlines()[:1]
        try:
            vals = [int(x) for x in lines[0].split()[1:]] if lines else []
        except ValueError:
            vals = []
        if len(vals) >= 5:
            total, idle = sum(vals), vals[3] + vals[4]
            if self.prev_stat is not None:
                d_total, d_idle = total - self.prev_stat[0], idle - self.prev_stat[1]
                if d_total > 0:
                    self.add("cpu_pct", "%", 100.0 * (d_total - d_idle) / d_total, t)
            self.prev_stat = (total, idle)
        gpu = slurp_int(self.root + "/sys/devices/gpu.0/load")
        if gpu is not None:
            self.add("gpu_pct", "%", gpu / 10.0 if gpu > 100 else float(gpu), t)

    def status_line(self, t):
        """原地刷新一行实时状态（RAM / 功耗 / 最热温区 / RSS），仅在终端下启用。"""
        mem = self.stats.get("mem_used_mb")
        power = self.stats.get("power_total_mw")
        rss = self.stats.get("proc_rss_mb")
        temps = [(k[len("temp_"):], s) for k, s in self.stats.items() if k.startswith("temp_")]
        hot = max(((n, s.v[-1]) for n, s in temps), key=lambda kv: kv[1]) if temps else None
        info = []
        if mem:
            info.append("RAM %.0fMB" % mem.v[-1])
        if power:
            info.append("PWR %.2fW" % (power.v[-1] / 1000.0))
        if hot:
            info.append("%s %.1fC" % hot)
        if rss:
            info.append("RSS %.0fMB" % rss.v[-1])
        sys.stdout.write("\r[%6.1fs] %-58s" % (t, "  ".join(info)))
        sys.stdout.flush()

    # --- 主循环 ---

    def run(self):
        """全流程：探测节点 -> 启动/解析目标进程 -> 按节拍采样 -> 收尾。"""
        sys.stderr.write("[INFO] jetson_monitor %s | 工作目录: %s | 采样间隔 %.2fs\n"
                         % (VERSION, os.getcwd(), self.a.interval))
        self.rails = power_rails(self.root, sys.stderr.write)
        self.zones = thermal_zones(self.root)
        for got, hint in (
                (self.rails, "未发现 INA3221 功耗节点（部分 L4T 上仅 root 可读），请尝试 sudo 重跑"),
                (self.zones, "未发现 thermal_zone*，温度项缺失"),
                (slurp(self.root + "/proc/meminfo"), "无法读取 /proc/meminfo")):
            if not got:
                sys.stderr.write("[WARN] %s\n" % hint)
        sys.stderr.write("[INFO] 功耗通道: %s | 温区: %s\n"
                         % (", ".join(n for n, _ in self.rails) or "none",
                            ", ".join(n for n, _ in self.zones) or "none"))

        if STOP:  # 探测阶段被 Ctrl+C 打断：不再拉起工作负载
            sys.stderr.write("[INFO] 已收到停止信号，跳过拉起工作负载\n")
            self.start_wall = self.end_wall = time.time()
            return

        a = self.a
        if a.exec_cmd:
            issues, ok = preflight(a.exec_cmd)
            for msg in issues:
                sys.stderr.write("[WARN] %s\n" % msg)
            if issues and not ok and not a.no_check:
                raise SystemExit("[ERROR] --exec 体检不通过，已中止；确认仍要执行请加 --no-check\n"
                                 "        当前目录: %s\n        命令: %s"
                                 % (os.getcwd(), a.exec_cmd))
            sys.stderr.write("[INFO] 拉起工作负载: %s\n" % a.exec_cmd)
            # 独立会话：终端 Ctrl+C 无法送达，代价是收尾需整组处理（见 stop_workload）
            self.child = subprocess.Popen(a.exec_cmd, shell=True, start_new_session=True)
            self.pid = self.child.pid
            if os.name == "posix":
                self.child_pgid = self.child.pid   # setsid 之后 pgid == pid
                self.pick_workload()
            sys.stderr.write("[INFO] wrapper pid=%d，采样对象 pid=%d，采到它退出为止\n"
                             % (self.child.pid, self.pid))
        else:
            self.pid = self.resolve(a.pid)
            sys.stderr.write("[INFO] 目标进程 pid=%d，采到它退出为止\n" % self.pid)

        self.start_wall = time.time()
        t0 = time.monotonic()
        next_tick = t0
        while not STOP:  # 绝对时刻定时，避免单次采样超时导致后续采样持续偏移
            t = time.monotonic() - t0
            self.sample(t)
            if self.finished(t) or STOP:
                break
            next_tick += a.interval
            delay = next_tick - time.monotonic()
            if delay > 0:
                self.pause(delay)  # 内部阻塞以 interval 为上限，收到信号后最多再等一个间隔
            else:
                next_tick = time.monotonic()  # 采样慢于 interval：重置定时基准，避免忙等
        self.end_wall = time.time()
        if self.live:
            sys.stdout.write("\n")
        self.stop_workload()
        self.hint_if_dead_on_arrival()

    def hint_if_dead_on_arrival(self):
        """目标未正常运行即退出：区分「命令/路径错误」与「程序自身报错」。

        探测阶段最长数秒，判据必须同时卡帧数与时长，只看任一项都会误判。
        """
        if self.killed_by_us or self.child is None or not self.child.returncode:
            return
        window = self.end_wall - self.start_wall
        if self.samples > 3 or window > 5.0:  # 采样量或时长足够，说明它确实正常运行过
            return
        code = self.child.returncode
        if code in (126, 127):
            tip = "shell 未能找到命令（%s）" % {126: "无执行权限", 127: "路径不存在"}[code]
        else:
            tip = "程序自身退出，请查看其报错（模型/配置/视频路径最常见）"
        sys.stderr.write(
            "[WARN] 目标 %.1fs 内只采到 %d 帧就以 code %s 退出：%s\n"
            "[WARN]   相对路径按 monitor 当前目录 %s 解析；建议先手工执行一次该命令\n"
            % (window, self.samples, code, tip, os.getcwd()))

    def pause(self, delay):
        """睡满采样间隔；拉起模式下同时等待 wrapper，其退出即提前唤醒。

        仅拉起模式有 child 可等；附着模式必须走 sleep 分支，否则退化为忙等。
        """
        if self.child is None:
            time.sleep(delay)
            return
        try:
            self.child.wait(timeout=delay)
        except subprocess.TimeoutExpired:
            pass

    # --- 拉起模式：识别负载、判定存活、收尾 ---

    def group_scan(self, force=False):
        """wrapper 所在进程组中存活的成员（不含 wrapper 与自身），1s 缓存一次。"""
        if self.child_pgid is None:
            return []
        now = time.monotonic()
        if force or self.members is None or now - self.members_at >= 1.0:
            self.members = group_members(self.root, self.child_pgid,
                                         skip=(os.getpid(), self.child.pid))
            self.members_at = now
            if self.members:
                self.seen_member = True
        return self.members

    def pick_workload(self):
        """把采样对象从 wrapper shell 切换为真正的负载进程。

        sh 只是守着复合命令的 wrapper，对其采样的 VmRSS/CPU 均取自 sh；最多等 1s，超时则回退。
        """
        deadline = time.monotonic() + 1.0
        while True:
            real = [m for m in self.group_scan(force=True) if m[1] not in SHELL_NAMES]
            if real:
                self.set_pid(real[0][0], real[0][1])
                return
            if time.monotonic() >= deadline:
                return
            time.sleep(0.05)

    def set_pid(self, pid, comm=""):
        """切换采样对象；更换进程必须清除 CPU 差分的历史点，否则首个采样值偏高。"""
        if pid == self.pid:
            return
        self.pid = pid
        self.prev_ticks = None  # CPU 差分重新起算
        sys.stderr.write("[INFO] 采样对象切到 pid=%d%s\n" % (pid, " (%s)" % comm if comm else ""))

    def stop_workload(self):
        """收尾时确保工作负载一同退出：它在独立会话中，终端信号无法送达，只能整组处理。

        先 SIGTERM 等待 2s，未退出再 SIGKILL。
        """
        if self.child is None or self.child.poll() is not None:
            return
        self.killed_by_us = True
        for hard, wait in ((False, 2.0), (True, 1.0)):
            self.signal_group(hard)
            try:
                self.child.wait(timeout=wait)
                sys.stderr.write("[INFO] 工作负载已收尾（%s）\n"
                                 % ("SIGKILL" if hard else "SIGTERM"))
                return
            except subprocess.TimeoutExpired:
                pass
        if self.child.poll() is None:
            sys.stderr.write("[WARN] 工作负载 pid=%d 仍未退出，请手动处理\n" % self.child.pid)

    def kill_now(self):
        """强制退出（第二次 Ctrl+C）专用：立刻 SIGKILL 整组。

        工作负载在独立会话中，monitor 退出后它无人接管，若不处理会留下继续占用 GPU 的孤儿进程。
        """
        if self.child is None or self.child.poll() is not None:
            return
        self.killed_by_us = True
        self.signal_group(hard=True)

    def signal_group(self, hard=False):
        """优先整组发信号（wrapper 的子孙一并处理）；取不到进程组时退化为仅对 wrapper 发信号。"""
        if self.child_pgid is not None:
            try:
                os.killpg(self.child_pgid, signal.SIGKILL if hard else signal.SIGTERM)
                return
            except OSError:
                pass
        try:
            if hard:
                self.child.kill()
            else:
                self.child.terminate()
        except OSError:
            pass

    def resolve(self, target):
        """附着模式：将 --pid 的 'PID 或进程名' 解析为唯一真实 pid，歧义或不存在时直接报错退出。"""
        if target.isdigit():
            if not os.path.isdir("%s/proc/%s" % (self.root, target)):
                raise SystemExit("[ERROR] pid %s 不存在或已退出" % target)
            return int(target)
        if not os.path.isdir(self.root + "/proc"):
            raise SystemExit("[ERROR] 按名字附着需要 /proc（仅 Linux/Jetson 可用）")
        hits = find_pid(self.root, target)
        if len(hits) != 1:
            raise SystemExit("[ERROR] '%s' 匹配到 %d 个进程 %s，请改用 --pid NUM"
                             % (target, len(hits), hits))
        sys.stderr.write("[INFO] '%s' -> pid %d\n" % (target, hits[0]))
        return hits[0]

    def finished(self, t):
        """收尾判定：达到时长上限 / 附着目标消失 / 拉起的工作负载结束。"""
        if self.a.duration and t >= self.a.duration:
            self.reason = "达到 --duration %.1fs" % self.a.duration
            return True
        if self.child is None:  # 附着模式
            if alive(self.root, self.pid):
                return False
            state = proc_state(self.root, self.pid)  # 已消失：再读一次区分「消失」与「僵尸」
            self.reason = "目标进程 pid %d 已退出%s" % (
                self.pid, "（僵尸态，父进程未回收）" if state == "Z" else "")
            return True
        code = self.child.poll()  # 拉起模式：wrapper 自身已退出
        if code is not None:
            self.reason = "子进程退出（code %s）" % code
            return True
        if self.seen_member and not self.group_scan():
            # wrapper 仍存活但负载已全部退出：复合命令 fork 出中间层或残留后台子孙时 wrapper
            # 不会退出，缺少这一条会一直采样下去。
            self.reason = "工作负载已退出（wrapper 仍在，已一并收尾）"
            return True
        return False

    # --- 报告 ---

    @property
    def samples(self):
        """已采帧数：以内存序列长度为准（每个采样点必写它）。"""
        return len(self.stats["mem_used_mb"].v) if "mem_used_mb" in self.stats else 0

    def board(self):
        """开发板型号（device-tree model），读不到时返回 'unknown board'。"""
        return (slurp(self.root + "/proc/device-tree/model") or "").replace("\x00", " ").strip() \
            or "unknown board"

    def report(self):
        """终端统计表：每个指标一行（均值/峰值/峰现时刻/最小）+ 功耗/内存/最热温区小结。"""
        window = self.end_wall - self.start_wall
        digits = {"mW": 1, "C": 1, "%": 1, "MB": 1}
        out = ["", "=" * 74,
               "Jetson Resource Monitor  |  %s -> %s  |  %.1fs  |  %d samples  |  %s"
               % (time.strftime("%H:%M:%S", time.localtime(self.start_wall)),
                  time.strftime("%H:%M:%S", time.localtime(self.end_wall)),
                  window, self.samples, self.reason or "手动停止"),
               "=" * 74,
               "%-24s %-4s %10s %10s %8s %10s" % ("metric", "unit", "mean", "peak", "peak_at", "min")]
        for key in sorted(self.stats):
            s = self.stats[key]
            d = digits.get(s.unit, 2)
            out.append("%-24s %-4s %10.*f %10.*f %7.1fs %10.*f"
                       % (key, s.unit, d, s.mean, d, s.peak, s.peak_at, d, s.low))
        out.append("=" * 74)
        power, mem = self.stats.get("power_total_mw"), self.stats.get("mem_used_mb")
        if power:
            out.append("整机平均功耗 %.2f W，峰值 %.2f W；按均值估算能耗约 %.1f mWh"
                       % (power.mean / 1000.0, power.peak / 1000.0,
                          power.mean / 1000.0 * window / 3.6))
        if mem:
            out.append("内存已用（含 GPU 共享）平均 %.0f MB，峰值 %.0f MB" % (mem.mean, mem.peak))
        temps = [(k[len("temp_"):], s) for k, s in self.stats.items() if k.startswith("temp_")]
        if temps:
            name, s = max(temps, key=lambda kv: kv[1].peak)
            out.append("最高温区 %s，峰值 %.1f℃" % (name, s.peak))
        return "\n".join(out)

    # --- 画图：内存 + 温度两块面板，只出 PNG ---

    def draw(self, path):
        """出内存 + 温度两块面板的 PNG；图内文字全英文，不依赖 CJK 字体。"""
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            sys.stderr.write('[WARN] 未安装 matplotlib，跳过出图。安装方式：'
                             'pip3 install "matplotlib==3.3.4"（Python 3.6）\n')
            return
        groups = []
        if "mem_used_mb" in self.stats:
            groups.append(("Memory used (MB)", [("Memory used (incl. GPU shared)",
                                                self.stats["mem_used_mb"])]))
        temps = [(k[len("temp_"):], self.stats[k]) for k in sorted(self.stats)
                 if k.startswith("temp_")]
        if temps:
            groups.append(("Temperature (\u00b0C)", temps))
        if not groups:
            sys.stderr.write("[WARN] 无内存/温度样本，跳过出图\n")
            return

        fonts = ["DejaVu Sans", "Liberation Sans", "Arial"]  # 图内文字全英文，不依赖 CJK 字体
        plt.rcParams["font.sans-serif"] = fonts
        plt.rcParams["axes.unicode_minus"] = False
        fig, axes = plt.subplots(len(groups), 1, figsize=(10, 3.4 * len(groups)),
                                 sharex=True, squeeze=False)
        for ax, (ylabel, items) in zip(axes[:, 0], groups):
            drawn = []
            for i, (label, s) in enumerate(items):
                color = PALETTE[i % len(PALETTE)]
                ax.plot(s.t, s.v, color=color, linewidth=1.6, label=label)
                ax.axhline(s.mean, color=color, linewidth=0.7, linestyle=":", alpha=0.7)
                ax.plot([s.peak_at], [s.peak], "o", color=color, markersize=4)
                drawn.append((color, s))
            ax.set_ylabel(ylabel)
            ax.grid(alpha=0.3)
            # 图例置于面板外，避免与曲线、峰值标注重叠
            ax.legend(loc="lower left", bbox_to_anchor=(0.0, 1.01), ncol=min(len(items), 3),
                      fontsize=9, frameon=False, borderaxespad=0.0)
            # 峰值标注：贴近右边界时翻到左侧，贴近顶部时移到下方，避免被裁剪或压线
            x_lo, x_hi = ax.get_xlim()
            y_lo, y_hi = ax.get_ylim()
            for color, s in drawn:
                high = s.peak > y_hi - (y_hi - y_lo) * 0.1
                left = s.peak_at > x_lo + (x_hi - x_lo) * 0.8
                ax.annotate("peak %.1f" % s.peak, (s.peak_at, s.peak), color=color, fontsize=9,
                            textcoords="offset points",
                            xytext=(-4 if left else 4, -16 if high else 5),
                            horizontalalignment="right" if left else "left")
        axes[-1][0].set_xlabel("Time (s since sampling start)")
        fig.suptitle("Jetson Resource Monitor: Memory & Temperature (dotted = mean)",
                     fontsize=11)
        # 图上保留一行元信息：板子/时间窗/采样参数/目标进程，图单独分发时也可自证来源
        meta = ("%s | %s -> %s (%.1fs, %d samples, interval %.2fs)"
                % (self.board(),
                   time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(self.start_wall)),
                   time.strftime("%H:%M:%S", time.localtime(self.end_wall)),
                   self.end_wall - self.start_wall, self.samples, self.a.interval))
        if self.a.exec_cmd:
            meta += "\nTarget (spawned): pid %s | %s" % (self.pid, self.a.exec_cmd)
        else:
            meta += "\nTarget (attached): pid %s" % self.pid
        fig.text(0.01, 0.005, meta, fontsize=8, color="#555555")
        fig.tight_layout(rect=(0, 0.045, 1, 0.965))
        fig.savefig(path, dpi=130)
        plt.close(fig)
        sys.stderr.write("[INFO] 曲线图已写入 %s（%.1f KB）\n"
                         % (os.path.abspath(path), os.path.getsize(path) / 1024.0))

    # --- 落盘：原始采样序列 + 元信息 ---

    def dump_json(self, path):
        """把原始采样序列写成 JSON（--json 用，不依赖 matplotlib）。

        每指标一条 {"unit","t","v"}（t = 采样起始后秒数），仅存原始值不预算统计，另附板子 / 目标 / 窗口 / reason / summary。
        """
        metrics, summary = OrderedDict(), OrderedDict()
        for key in sorted(self.stats):
            s = self.stats[key]
            metrics[key] = OrderedDict([("unit", s.unit),
                                        ("t", [round(x, 4) for x in s.t]),
                                        ("v", [round(x, 4) for x in s.v])])
            summary[key] = OrderedDict([("unit", s.unit), ("n", len(s.v)),
                                        ("mean", round(s.mean, 4)), ("peak", round(s.peak, 4)),
                                        ("peak_at", round(s.peak_at, 3)), ("min", round(s.low, 4))])
        target = OrderedDict([("mode", "spawn" if self.a.exec_cmd else "attach"),
                              ("pid", self.pid)])
        if self.a.exec_cmd:
            target["exec_cmd"] = self.a.exec_cmd
        doc = OrderedDict([
            ("monitor", OrderedDict([("name", "jetson_monitor.py"), ("version", VERSION)])),
            ("board", self.board()),
            ("target", target),
            ("window", OrderedDict([("start", iso_time(self.start_wall)),
                                    ("end", iso_time(self.end_wall)),
                                    ("seconds", round(self.end_wall - self.start_wall, 3)),
                                    ("interval", self.a.interval),
                                    ("samples", self.samples)])),
            ("reason", self.reason or "手动停止"),
            ("summary", summary),
            ("metrics", metrics),
        ])
        with open(path, "w") as fh:
            fh.write(json_dump(doc) + "\n")
        if not metrics:
            sys.stderr.write("[WARN] 未采到任何指标（采样窗口为空），JSON 中仅有元信息\n")
        sys.stderr.write("[INFO] 原始数据已写入 %s（%.1f KB：%d samples × %d 指标）\n"
                         % (os.path.abspath(path), os.path.getsize(path) / 1024.0,
                            self.samples, len(metrics)))

    def flush_json(self):
        """立即落盘已采数据：收尾 / 强制退出 / 异常三条路径均调用它。"""
        if not self.json_path:
            return
        try:
            self.dump_json(self.json_path)
        except (IOError, OSError, ValueError) as err:
            sys.stderr.write("[WARN] JSON 落盘失败：%s\n" % err)


# ------------------------------------------------------------------ 入口
def prepare_out(path, ext):
    """统一输出路径：补扩展名 + 自动建父目录（runs/ 不存在时无需先 mkdir）。"""
    if not path:
        return None
    base, old = os.path.splitext(path)
    if old.lower() != ext:
        path = base + ext
    parent = os.path.dirname(path)
    if parent and not os.path.isdir(parent):
        os.makedirs(parent)
    return path


def parse_args(argv=None):
    """定义命令行接口，并把 --plot/--json 的路径补全成「带扩展名 + 父目录已建好」。"""
    ap = argparse.ArgumentParser(
        description="Jetson(TX2) 资源监视器：采内存/功耗/温度到目标进程退出，出终端统计 + PNG/JSON",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="示例（在仓库根目录执行）:\n"
               "  %(prog)s --exec \"cd bin && ./main ../data/dd/906-2-1.mp4 config_jetson.yaml\" "
               "--plot runs/906-2-1.png\n"
               "  %(prog)s --pid main --plot runs/attach.png\n"
               "  %(prog)s --json runs/906-2-1.json --exec \"...\"   # 不画图，仅落原始数据\n"
               "已在 bin/ 下则无需再 cd：--exec \"./main ../data/dd/906-2-1.mp4 "
               "config_jetson.yaml\"\n"
               "运行时 Ctrl+C 软收尾（输出统计与图像后退出），再按一次强制退出并结束工作负载\n")
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--pid", metavar="PID|NAME",
                      help="附着模式：已运行进程的 pid，或进程名（如 main，需唯一匹配）")
    mode.add_argument("--exec", dest="exec_cmd", metavar="CMD",
                      help="拉起模式：将被测程序作为子进程运行（shell 语法）。相对路径按 monitor "
                           "当前目录解析，需换目录则写 cd DIR && ...；启动前会做一次静态体检")
    ap.add_argument("--no-check", dest="no_check", action="store_true",
                    help="跳过 --exec 的静态体检（程序/配置文件路径查不到也照常执行）")
    ap.add_argument("--plot", metavar="PATH", default=None,
                    help="结束时把内存/温度曲线写成 PNG（扩展名非 .png 时自动改为 .png）")
    ap.add_argument("--json", metavar="PATH", default=None,
                    help="不画图，将原始采样序列写成 JSON（含 window/reason/summary，"
                         "可供 pandas/gnuplot 使用；扩展名非 .json 时自动改为 .json）")
    ap.add_argument("--interval", type=float, default=0.2, help="采样间隔秒，默认 0.2（下限 0.02）")
    ap.add_argument("--duration", type=float, default=0.0,
                    help="采样时长上限秒，默认 0 = 持续采集至目标退出")
    ap.add_argument("--extras", action="store_true", help="额外采集 CPU 总占用与 GPU 占用")
    ap.add_argument("--live", dest="live", action="store_true", default=None,
                    help="强制开启实时状态行（默认：stdout 为终端时开启）")
    ap.add_argument("--quiet", dest="live", action="store_false", help="关闭实时状态行")
    ap.add_argument("--root", default=None, metavar="DIR",
                    help="将 /proc、/sys 挂到该前缀下读取（自测用，实机请勿传入）")
    ap.add_argument("--version", action="version", version="jetson_monitor.py %s" % VERSION)
    args = ap.parse_args(argv)
    args.interval = max(0.02, args.interval)
    for attr, ext in (("plot", ".png"), ("json", ".json")):
        setattr(args, attr, prepare_out(getattr(args, attr), ext))
    return args


def install_signals():
    """Ctrl+C / SIGTERM 只置标志位，交给主循环收尾；再按一次才 SIGKILL 工作负载后强退。

    处理器内自我重发会被反复重入、主线程无法回到采样循环；阻塞点均以 --interval 为上限。
    """
    def on_signal(signum, _frame):
        global STOP
        if STOP:
            sys.stderr.write("\n[INFO] 再次收到信号 %d，强制退出（先结束工作负载）\n" % signum)
            if _MONITOR is not None:
                _MONITOR.flush_json()  # 已采数据先行落盘
                _MONITOR.kill_now()
            signal.signal(signum, signal.SIG_DFL)
            os.kill(os.getpid(), signum)
            return
        STOP = True
        sys.stderr.write("\n[INFO] 收到信号 %d，正在收尾（再按一次 Ctrl+C 可强制退出）...\n" % signum)

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            signal.signal(sig, on_signal)
        except (ValueError, OSError, AttributeError):
            pass  # 非主线程 / 平台不支持


def main(argv=None):
    """入口：解析参数 -> 安装信号处理器 -> 执行采集 -> 输出报告与产物（PNG / JSON）。"""
    global _MONITOR
    args = parse_args(argv)
    install_signals()
    mon = Monitor(args)
    _MONITOR = mon  # 记录引用，强制退出时信号处理器才能一并结束工作负载
    try:
        mon.run()
    except KeyboardInterrupt:  # 安装信号处理器后一般不会进入此分支，保留作兜底
        mon.stop_workload()
        mon.flush_json()
    except BaseException:
        mon.stop_workload()  # 中途出错也不遗留后台工作负载
        mon.flush_json()     # 已采数据不丢弃
        raise
    print(mon.report())
    if args.plot:
        mon.draw(args.plot)
    if args.json:
        mon.flush_json()
    if not args.plot and not args.json:
        sys.stderr.write("[INFO] 未指定 --plot / --json，本次仅输出终端统计（不画图、不落盘）\n")
    elif args.json and not args.plot:
        sys.stderr.write("[INFO] 未指定 --plot，本次不画图，仅落 JSON 原始数据\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
