#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GPU 实验监控：写入精简 .txt 报告；完整快照写入 last_snapshot.json 供下次对照（不塞进报告）。
仅使用 Python 标准库。
"""

from __future__ import annotations

import argparse
import json
import os
import re
import socket
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


REPORT_PREFIX = "gpu_monitor_"
STATE_FILE = "last_snapshot.json"
DEFAULT_KEEP_TXT_REPORTS = 5
JSON_BEGIN = "---BEGIN_GPU_MONITOR_JSON---"
JSON_END = "---END_GPU_MONITOR_JSON---"

# 空闲判定：无 compute 进程且显存占用低于该值（MiB）时视为“空卡”（可按环境调整）
DEFAULT_IDLE_MEM_MIB = 128


@dataclass
class ProcInfo:
    pid: int
    ppid: int
    process_name: str
    used_gpu_memory_mib: int
    cmdline: str
    parent_cmdline: str = ""
    # 用于报告展示：向上解析后的实验父进程（常为 nohup/bash 下的 .sh）
    experiment_parent_pid: int = 0
    experiment_parent_cmdline: str = ""


@dataclass
class GpuState:
    index: int
    name: str = ""
    utilization_gpu: int = 0
    memory_used_mib: int = 0
    memory_total_mib: int = 0
    compute_apps: List[ProcInfo] = field(default_factory=list)


def _run(cmd: List[str], timeout: int = 60) -> Tuple[int, str, str]:
    try:
        p = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            env={**os.environ, "LC_ALL": "C"},
        )
        return p.returncode, p.stdout or "", p.stderr or ""
    except FileNotFoundError as e:
        return 127, "", str(e)
    except subprocess.TimeoutExpired as e:
        return -1, "", str(e)


def _parse_csv_lines(stdout: str) -> List[List[str]]:
    rows: List[List[str]] = []
    for line in stdout.strip().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        # 简单 CSV：字段内可能有逗号时 nvidia-smi 会加引号，这里做最小处理
        parts: List[str] = []
        cur = ""
        in_q = False
        for ch in line:
            if ch == '"':
                in_q = not in_q
            elif ch == "," and not in_q:
                parts.append(cur.strip())
                cur = ""
            else:
                cur += ch
        parts.append(cur.strip())
        rows.append(parts)
    return rows


def _gpu_count() -> int:
    code, out, _ = _run(["nvidia-smi", "--list-gpus"])
    if code != 0:
        return 0
    return len([ln for ln in out.splitlines() if ln.strip().startswith("GPU ")])


def _read_cmdline(pid: int) -> str:
    try:
        raw = Path(f"/proc/{pid}/cmdline").read_bytes()
        if not raw:
            return ""
        return raw.decode("utf-8", errors="replace").replace("\x00", " ").strip()
    except OSError:
        return ""


def _read_ppid(pid: int) -> int:
    try:
        for line in Path(f"/proc/{pid}/status").read_text(errors="replace").splitlines():
            if line.startswith("PPid:"):
                return int(line.split()[1])
    except (OSError, ValueError, IndexError):
        pass
    return -1


def _parent_cmdline(ppid: int) -> str:
    if ppid <= 0:
        return ""
    return _read_cmdline(ppid)


def _normalize_cmdline(cmd: str) -> str:
    if not cmd:
        return ""
    return re.sub(r"\s+", " ", cmd.strip())


def resolve_experiment_parent(compute_pid: int, max_depth: int = 16) -> Tuple[int, str]:
    """
    从 nvidia-smi 报告的 GPU 计算进程 PID 向上查找用于展示的实验父进程：
    优先匹配命令行中含 .sh 的进程；否则取第一个非 python 的祖先；
    再否则退回直接父进程。
    """
    best_pid, best_cmd = -1, ""
    visited: set[int] = set()
    cur = compute_pid
    for _ in range(max_depth):
        ppid = _read_ppid(cur)
        if ppid <= 1 or ppid in visited:
            break
        visited.add(ppid)
        cmd = _read_cmdline(ppid)
        cur = ppid
        if not cmd:
            continue
        norm = _normalize_cmdline(cmd)
        # 优先：脚本路径 *.sh
        if re.search(r"(?:^|[\s/])[^\s]+\.sh\b", cmd):
            return ppid, norm
        low = cmd.lstrip().lower()
        if re.match(r"python\d*(\s|$)", low):
            continue
        if best_pid < 0:
            best_pid, best_cmd = ppid, norm
    if best_pid > 0 and best_cmd:
        return best_pid, best_cmd
    ppid = _read_ppid(compute_pid)
    if ppid > 0:
        c = _normalize_cmdline(_read_cmdline(ppid))
        if c:
            return ppid, c
    return compute_pid, _normalize_cmdline(_read_cmdline(compute_pid))


def collect_gpu_states() -> Tuple[List[GpuState], str]:
    """返回各 GPU 状态及 nvidia-smi 原始诊断（若有）。"""
    diag: List[str] = []
    n = _gpu_count()
    if n == 0:
        # 兼容部分环境
        code, out, err = _run(
            [
                "nvidia-smi",
                "--query-gpu=index,name,utilization.gpu,memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ]
        )
        if code != 0:
            diag.append(f"nvidia-smi GPU 查询失败: {err or out}")
            return [], "\n".join(diag)
        rows = _parse_csv_lines(out)
        n = len(rows)

    uuid_to_index: Dict[str, int] = {}
    code, out, err = _run(
        ["nvidia-smi", "--query-gpu=index,gpu_uuid", "--format=csv,noheader"]
    )
    if code == 0 and out.strip():
        for row in _parse_csv_lines(out):
            if len(row) >= 2:
                try:
                    uuid_to_index[row[1].strip()] = int(row[0].strip())
                except ValueError:
                    continue
    else:
        diag.append(f"无法解析 gpu_uuid 映射: {err or out}")

    gpus: List[GpuState] = []
    code, out, err = _run(
        [
            "nvidia-smi",
            "--query-gpu=index,name,utilization.gpu,memory.used,memory.total",
            "--format=csv,noheader,nounits",
        ]
    )
    if code != 0:
        diag.append(f"nvidia-smi 基本查询失败: {err or out}")
        return [], "\n".join(diag)

    for row in _parse_csv_lines(out):
        if len(row) < 5:
            continue
        try:
            idx = int(row[0].strip())
        except ValueError:
            continue
        try:
            util = int(float(row[2].strip().replace(" %", "")))
        except ValueError:
            util = 0
        try:
            mem_used = int(float(row[3].strip()))
            mem_tot = int(float(row[4].strip()))
        except ValueError:
            mem_used, mem_tot = 0, 0
        gpus.append(
            GpuState(
                index=idx,
                name=row[1].strip(),
                utilization_gpu=util,
                memory_used_mib=mem_used,
                memory_total_mib=mem_tot,
            )
        )

    # compute apps（按 UUID）
    code, out, err = _run(
        [
            "nvidia-smi",
            "--query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory",
            "--format=csv,noheader",
        ]
    )
    apps_by_gpu: Dict[int, List[ProcInfo]] = {g.index: [] for g in gpus}
    if code == 0 and out.strip():
        for row in _parse_csv_lines(out):
            if len(row) < 4:
                continue
            gu, pid_s, pname, mem_s = [x.strip() for x in row[:4]]
            try:
                pid = int(pid_s)
            except ValueError:
                continue
            try:
                mem = int(float(mem_s.replace(" MiB", "").strip()))
            except ValueError:
                mem = 0
            gi = uuid_to_index.get(gu)
            if gi is None:
                # 回退：若只有一张卡，直接归到 0
                if len(gpus) == 1:
                    gi = gpus[0].index
                else:
                    continue
            ppid = _read_ppid(pid)
            cmd = _read_cmdline(pid)
            pcmd = _parent_cmdline(ppid) if ppid > 0 else ""
            exp_ppid, exp_pcmd = resolve_experiment_parent(pid)
            apps_by_gpu.setdefault(gi, []).append(
                ProcInfo(
                    pid=pid,
                    ppid=ppid,
                    process_name=pname,
                    used_gpu_memory_mib=mem,
                    cmdline=cmd,
                    parent_cmdline=pcmd,
                    experiment_parent_pid=exp_ppid,
                    experiment_parent_cmdline=exp_pcmd,
                )
            )
    else:
        diag.append(f"compute-apps 查询失败或为空: {err or out}")

    for g in gpus:
        g.compute_apps = apps_by_gpu.get(g.index, [])

    return gpus, "\n".join(diag)


def gpu_is_idle(g: GpuState, idle_mem_mib: int) -> bool:
    if g.compute_apps:
        return False
    return g.memory_used_mib <= idle_mem_mib


def extract_log_paths(cmdline: str) -> List[str]:
    if not cmdline:
        return []
    paths: List[str] = []
    for pat in (
        r"(?:^|[\s;])(?:>>|>|2>|&>)\s*([^\s&|;]+)",
        r"\btee\s+(?:-a\s+)?([^\s&|;]+)",
    ):
        for m in re.finditer(pat, cmdline):
            p = m.group(1).strip().strip('"').strip("'")
            if p and p not in paths:
                paths.append(p)
    return paths


def pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def analyze_log_tail(path: Path, tail_lines: int = 400) -> Dict[str, Any]:
    result: Dict[str, Any] = {
        "path": str(path),
        "exists": path.is_file(),
        "readable": False,
        "size_bytes": 0,
        "last_lines_sampled": 0,
        "signals": [],
        "summary": "",
    }
    if not result["exists"]:
        result["summary"] = "日志文件不存在（可能路径已变或未重定向到文件）。"
        return result
    try:
        result["size_bytes"] = path.stat().st_size
        # 只读尾部，避免大文件
        with path.open("rb") as f:
            f.seek(0, os.SEEK_END)
            size = f.tell()
            chunk = min(size, 256 * 1024)
            f.seek(max(0, size - chunk))
            data = f.read().decode("utf-8", errors="replace")
        lines = data.splitlines()
        tail = lines[-tail_lines:] if len(lines) > tail_lines else lines
        result["readable"] = True
        result["last_lines_sampled"] = len(tail)
        text = "\n".join(tail).lower()

        neg_patterns = [
            (r"traceback", "发现 Traceback"),
            (r"\berror\b", "出现 Error 关键字"),
            (r"runtimeerror", "RuntimeError"),
            (r"cuda out of memory", "CUDA OOM"),
            (r"out of memory", "OOM"),
            (r"killed\b", "进程可能被 OOM killer 终止（Killed）"),
            (r"segmentation fault", "段错误"),
            (r"nccl error", "NCCL 错误"),
            (r"errno", "系统 errno 相关错误"),
            (r"assertionerror", "AssertionError"),
            (r"nan\b", "出现 NaN（需结合上下文）"),
            (r"inf\b", "出现 Inf（需结合上下文）"),
        ]
        pos_patterns = [
            (r"training (?:complete|finished|done)", "训练完成类提示"),
            (r"experiment (?:complete|finished|done)", "实验完成"),
            (r"\ball (?:tasks|jobs) (?:done|completed)\b", "全部任务完成"),
            (r"saved (?:checkpoint|model)", "已保存检查点"),
            (r"finished successfully", "成功结束"),
            (r"\bdone\b\.\s*$", "行尾 done（弱信号）"),
        ]

        for rx, label in neg_patterns:
            if re.search(rx, text, re.IGNORECASE):
                result["signals"].append(f"[负面] {label}")
        for rx, label in pos_patterns:
            if re.search(rx, text, re.IGNORECASE):
                result["signals"].append(f"[正面] {label}")

        # 最后一行非空（完整写入报告，不截断）
        last_nonempty = next((ln for ln in reversed(tail) if ln.strip()), "")
        result["last_line"] = last_nonempty

        if any(s.startswith("[负面]") for s in result["signals"]):
            result["summary"] = "倾向：异常退出或运行中出错（请人工确认日志）。"
        elif any(s.startswith("[正面]") for s in result["signals"]):
            result["summary"] = "倾向：可能已正常跑完（请结合实验是否应有后续输出确认）。"
        else:
            result["summary"] = "倾向：不明确（无强信号）；请打开日志查看末尾上下文。"
    except OSError as e:
        result["summary"] = f"读取日志失败: {e}"
    return result


def summarize_gpu_line(g: GpuState) -> str:
    """单行摘要：每张卡一行；按实验父进程聚合，命令行完整输出。"""
    if not g.compute_apps:
        return f"无计算进程，显存 {g.memory_used_mib}/{g.memory_total_mib} MiB"
    by_parent: Dict[int, Tuple[int, str]] = {}
    mem_by_parent: Dict[int, int] = {}
    for a in g.compute_apps:
        ppid = a.experiment_parent_pid or a.ppid
        if ppid <= 0:
            ppid = a.pid
        pcmd = a.experiment_parent_cmdline or _normalize_cmdline(
            a.parent_cmdline
        )
        if ppid not in mem_by_parent:
            mem_by_parent[ppid] = 0
            by_parent[ppid] = (ppid, pcmd)
        mem_by_parent[ppid] += a.used_gpu_memory_mib
        if len(pcmd) > len(by_parent[ppid][1]):
            by_parent[ppid] = (ppid, pcmd)
    parts: List[str] = []
    for ppid in sorted(mem_by_parent.keys()):
        _, pcmd = by_parent[ppid]
        mem = mem_by_parent[ppid]
        parts.append(
            f"父进程 pid={ppid} ~{mem}MiB | {pcmd}"
        )
    return " || ".join(parts)


def state_to_json_dict(
    gpus: List[GpuState], hostname: str, ts: str, diag: str
) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "version": 1,
        "timestamp": ts,
        "hostname": hostname,
        "nvidia_smi_diag": diag,
        "gpus": [],
    }
    for g in gpus:
        d = {
            "index": g.index,
            "name": g.name,
            "utilization_gpu": g.utilization_gpu,
            "memory_used_mib": g.memory_used_mib,
            "memory_total_mib": g.memory_total_mib,
            "compute_apps": [],
        }
        for a in g.compute_apps:
            d["compute_apps"].append(
                {
                    "pid": a.pid,
                    "ppid": a.ppid,
                    "process_name": a.process_name,
                    "used_gpu_memory_mib": a.used_gpu_memory_mib,
                    "cmdline": a.cmdline,
                    "parent_cmdline": a.parent_cmdline,
                    "experiment_parent_pid": a.experiment_parent_pid,
                    "experiment_parent_cmdline": a.experiment_parent_cmdline,
                }
            )
        out["gpus"].append(d)
    return out


def parse_embedded_json_from_report(report_path: Path) -> Optional[Dict[str, Any]]:
    """兼容旧版：从 gpu_monitor_*.txt 尾部读取内嵌 JSON。"""
    try:
        text = report_path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    if JSON_BEGIN not in text or JSON_END not in text:
        return None
    try:
        blob = text.split(JSON_BEGIN, 1)[1].split(JSON_END, 1)[0].strip()
        return json.loads(blob)
    except (ValueError, json.JSONDecodeError, IndexError):
        return None


def prune_keep_txt_reports(out_dir: Path, keep: int) -> None:
    """只保留最近的 keep 个 gpu_monitor_*.txt，按修改时间从新到旧。"""
    if keep < 1:
        return
    txt_files = sorted(
        out_dir.glob(f"{REPORT_PREFIX}*.txt"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    for old in txt_files[keep:]:
        try:
            old.unlink()
        except OSError:
            pass


def load_previous_snapshot(out_dir: Path) -> Tuple[Optional[Dict[str, Any]], str]:
    """
    运行本次采集前先读「上一次」快照：优先 last_snapshot.json，
    否则回退到最新的旧版 .txt（含内嵌 JSON）。
    """
    snap = out_dir / STATE_FILE
    if snap.is_file():
        try:
            return json.loads(snap.read_text(encoding="utf-8")), STATE_FILE
        except json.JSONDecodeError:
            pass
    candidates = sorted(
        out_dir.glob(f"{REPORT_PREFIX}*.txt"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    for c in candidates:
        data = parse_embedded_json_from_report(c)
        if data:
            return data, f"{c.name}（旧版内嵌 JSON）"
    return None, ""


def format_report(
    gpus: List[GpuState],
    hostname: str,
    ts: str,
    diag: str,
    prev_source: str,
    prev_data: Optional[Dict[str, Any]],
    idle_mem_mib: int,
    tail_lines: int,
    expected_gpus: Optional[int],
) -> str:
    lines: List[str] = []
    lines.append(f"GPU 监控  {ts}  {hostname}")
    cv = os.environ.get("CUDA_VISIBLE_DEVICES")
    if cv is not None:
        lines.append(f"CUDA_VISIBLE_DEVICES={cv}")
    if diag:
        lines.append(f"nvidia-smi: {diag.strip()}")
    if not gpus:
        lines.append("【注意】未枚举到 GPU（驱动/权限/无卡）。")
        lines.append("机器快照见同目录 " + STATE_FILE + "（若有）。")
        return "\n".join(lines)
    if expected_gpus is not None and len(gpus) != expected_gpus:
        lines.append(
            f"【注意】当前 {len(gpus)} 张卡，期望 {expected_gpus} 张。"
        )

    prev_by_idx: Dict[int, Dict[str, Any]] = {}
    if prev_data and isinstance(prev_data.get("gpus"), list):
        for g in prev_data["gpus"]:
            try:
                prev_by_idx[int(g["index"])] = g
            except (TypeError, ValueError, KeyError):
                continue

    lines.append("")
    lines.append(
        "概览  U%=GPU利用率  显存=已用/总计(MiB)  摘要=父进程命令（完整，按父进程聚合显存）"
    )
    for g in sorted(gpus, key=lambda x: x.index):
        idle = gpu_is_idle(g, idle_mem_mib)
        tag = "空闲" if idle else "占用"
        summ = summarize_gpu_line(g)
        lines.append(
            f"[GPU{g.index}] {tag}  {g.utilization_gpu:>3}%  "
            f"{g.memory_used_mib}/{g.memory_total_mib}  |  {summ}"
        )

    idle_gpus = [g for g in sorted(gpus, key=lambda x: x.index) if gpu_is_idle(g, idle_mem_mib)]
    if not idle_gpus:
        lines.append("")
        lines.append("结论: 无空闲卡（按当前阈值）。详细状态见 " + STATE_FILE + "。")
        return "\n".join(lines)

    lines.append("")
    lines.append("—— 仅空闲卡：对照上次快照 ——")
    if prev_source:
        lines.append(f"上次快照来源: {prev_source}")
    else:
        lines.append("上次快照: 无（首次运行或尚无 " + STATE_FILE + " / 旧版内嵌 JSON）。")

    for g in idle_gpus:
        lines.append("")
        lines.append(f"【GPU {g.index}】空闲  显存≤{idle_mem_mib}MiB 且无 compute 进程")
        if not prev_data:
            lines.append("  无法对照（无上次快照）。")
            continue

        pg = prev_by_idx.get(g.index)
        if not pg:
            lines.append("  上次快照中无该卡。")
            continue

        prev_apps = pg.get("compute_apps") or []
        if not prev_apps:
            lines.append("  上次该卡也无计算进程（可能一直空）。")
            continue

        all_logs: List[str] = []
        for pa in prev_apps:
            child_pid = int(pa.get("pid", -1))
            exp_pid = int(
                pa.get("experiment_parent_pid", 0) or pa.get("ppid", 0) or 0
            )
            exp_cmd = str(
                pa.get("experiment_parent_cmdline")
                or pa.get("parent_cmdline")
                or pa.get("cmdline")
                or ""
            )
            exp_cmd = _normalize_cmdline(exp_cmd)
            lines.append(f"  曾: 父进程 pid={exp_pid}  计算进程 pid={child_pid}  {exp_cmd}")
            for lp in extract_log_paths(exp_cmd) + extract_log_paths(
                str(pa.get("cmdline", ""))
            ):
                if lp not in all_logs:
                    all_logs.append(lp)
            check_pid = exp_pid if exp_pid > 0 else child_pid
            if check_pid > 0:
                lines.append(
                    f"      该父进程 PID 仍在: {'是' if pid_alive(check_pid) else '否'}"
                )

        if not all_logs:
            lines.append(
                "  未从命令行解析到日志路径（建议在 nohup 里写绝对路径重定向）。"
            )
            continue

        for lp in all_logs:
            pth = Path(lp).expanduser()
            lines.append(f"  日志: {pth}")
            if not pth.is_absolute():
                lines.append("    （相对路径，cwd 未知）")
            an = analyze_log_tail(pth, tail_lines=tail_lines)
            lines.append(f"    → {an.get('summary', '')}")
            sigs = an.get("signals") or []
            if sigs:
                lines.append("    信号: " + "；".join(sigs[:8]))
            if an.get("last_line"):
                lines.append(f"    末行: {an['last_line']}")

    lines.append("")
    lines.append("提示: 空闲/日志判断为启发式；疑义请直接看 tmux 与完整日志。")
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description="GPU 实验监控与空闲卡日志辅助分析")
    ap.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "reports",
        help="报告输出目录（默认: 脚本目录下 reports/）",
    )
    ap.add_argument(
        "--idle-mem-mib",
        type=int,
        default=DEFAULT_IDLE_MEM_MIB,
        help=f"无 compute 进程时，显存低于该值(MiB)视为空卡（默认 {DEFAULT_IDLE_MEM_MIB}）",
    )
    ap.add_argument(
        "--tail-lines",
        type=int,
        default=400,
        help="分析日志时读取最后多少行（默认 400）",
    )
    ap.add_argument(
        "--expected-gpus",
        type=int,
        default=None,
        metavar="N",
        help="可选：期望 GPU 数量（例如 8）；不一致时在报告顶部给出警告",
    )
    ap.add_argument(
        "--keep-reports",
        type=int,
        default=DEFAULT_KEEP_TXT_REPORTS,
        metavar="N",
        help=f"目录中最多保留多少个 gpu_monitor_*.txt（默认 {DEFAULT_KEEP_TXT_REPORTS}，更早的会删除）",
    )
    args = ap.parse_args()

    out_dir: Path = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    fname = f"{REPORT_PREFIX}{ts}.txt"
    out_path = out_dir / fname

    prev_data, prev_source = load_previous_snapshot(out_dir)

    hostname = socket.gethostname()
    gpus, diag = collect_gpu_states()

    report_body = format_report(
        gpus,
        hostname,
        ts,
        diag,
        prev_source,
        prev_data,
        idle_mem_mib=args.idle_mem_mib,
        tail_lines=args.tail_lines,
        expected_gpus=args.expected_gpus,
    )

    payload = state_to_json_dict(gpus, hostname, ts, diag)
    state_path = out_dir / STATE_FILE

    out_path.write_text(report_body + "\n", encoding="utf-8")
    state_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    # 便于快速查看：reports/latest.txt -> 最近一次报告
    latest = out_dir / "latest.txt"
    try:
        if latest.exists() or latest.is_symlink():
            latest.unlink()
        latest.symlink_to(out_path.name)
    except OSError:
        pass
    print(f"已写入: {out_path}")
    print(f"已写入: {state_path}（供下次对照，不必打开）")
    if latest.exists() or latest.is_symlink():
        print(f"已更新软链: {latest} -> {out_path.name}")
    prune_keep_txt_reports(out_dir, keep=args.keep_reports)
    return 0


if __name__ == "__main__":
    sys.exit(main())
