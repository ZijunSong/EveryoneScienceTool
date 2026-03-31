#!/usr/bin/env python3
"""
将 progress.md 精炼为 progress_refined.md（默认模型 gpt-5-codex-mini）。
- 流水线内：在每次 write_checkpoint 后增量更新（按轮缓存，避免重复调用）。
- 离线：对已有会话目录或单独的 progress.md 文件执行全文解析并生成精炼版。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

PROGRESS_REFINED_FILENAME = "progress_refined.md"
PROGRESS_REFINED_CACHE_FILENAME = "progress_refined.cache.json"
DEFAULT_DISTILL_MODEL = "gpt-5-codex-mini"

def _sha256(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def _get_call_chat():
    from run_pipeline import call_chat

    return call_chat


def _get_normalize_url():
    from run_pipeline import normalize_openai_base_url

    return normalize_openai_base_url


SYSTEM_ROUND = """你是学术辩论记录编辑。下面给出同一轮次中「生成器 proposal」「审稿人」「生成器回应」的全文（可能很长）。
请压成便于事后回顾的精炼摘要（中文 Markdown），必须保留：
- 该轮提出的核心 idea 要点（若有标题/摘要请概括）
- 若审稿含 **ACL总分：X/5** 或多位审稿人分数，**逐位准确抄录分数**；一句话结论须与分数一致（不得把 3 分说成「需重大修改」或拒稿级——除非原文明显矛盾，则加注「注：原文分数与结论文本可能不一致」）
- 生成器回应：是否接受批评、是否修订、是否放弃、下一轮方向

删除套话、重复论证与过细枚举。使用小标题与列表，总长度建议 400–1200 字（若输入极短可更短）。"""

SYSTEM_TAIL = """下面是「已定稿标题与摘要」以及「大纲」相关片段（可能含初稿、审稿意见、修订终稿）。请整理为精炼阅读版（中文 Markdown）：
- 标题与摘要：可直接保留或略压缩
- 大纲：保留章节结构与关键实验点，删去冗长套话

若某段为空或写「尚未生成」，如实标注即可。"""

SYSTEM_MEMORY = """下面是若干条「已否决 idea」的摘要文本（可能很长）。请合并为 3–8 条要点列表（中文），每条一句话，突出否决原因与方向。"""


def distill_round_block(
    client: Any,
    model: str,
    generator_output: str,
    reviewer_output: str,
    reflection_output: str,
    *,
    temperature: float,
    max_tokens: int,
    timeout: float,
) -> str:
    call_chat = _get_call_chat()
    user = (
        "【生成器 proposal】\n\n"
        f"{generator_output}\n\n"
        "---\n\n【审稿人】\n\n"
        f"{reviewer_output}\n\n"
        "---\n\n【生成器回应导师】\n\n"
        f"{reflection_output}"
    )
    if len(user) > 120_000:
        user = user[:120_000] + "\n\n…（已截断，仅摘要前部）"
    return call_chat(
        client,
        model,
        [
            {"role": "system", "content": SYSTEM_ROUND},
            {"role": "user", "content": user},
        ],
        temperature,
        max_tokens,
        timeout,
        max_retries=4,
        base_delay_sec=2.0,
    )


def distill_tail_block(
    client: Any,
    model: str,
    final_title_abstract: str,
    outline_draft: str,
    outline_review: str,
    outline_output: str,
    *,
    temperature: float,
    max_tokens: int,
    timeout: float,
) -> str:
    call_chat = _get_call_chat()
    user = (
        "【最终用于大纲的标题与摘要】\n\n"
        f"{final_title_abstract or '（无）'}\n\n"
        "---\n\n【大纲初稿】\n\n"
        f"{outline_draft or '（无）'}\n\n"
        "---\n\n【审稿人对大纲的意见】\n\n"
        f"{outline_review or '（无）'}\n\n"
        "---\n\n【大纲修订终稿】\n\n"
        f"{outline_output or '（无）'}"
    )
    if len(user) > 120_000:
        user = user[:120_000] + "\n\n…（已截断）"
    return call_chat(
        client,
        model,
        [
            {"role": "system", "content": SYSTEM_TAIL},
            {"role": "user", "content": user},
        ],
        temperature,
        max_tokens,
        timeout,
        max_retries=4,
        base_delay_sec=2.0,
    )


def distill_memory_block(
    client: Any,
    model: str,
    memories: list[str],
    *,
    temperature: float,
    max_tokens: int,
    timeout: float,
) -> str:
    if not memories:
        return "（无）"
    call_chat = _get_call_chat()
    user = "\n\n---\n\n".join(f"条目{i+1}：{m}" for i, m in enumerate(memories))
    if len(user) > 80_000:
        user = user[:80_000] + "\n\n…（已截断）"
    return call_chat(
        client,
        model,
        [
            {"role": "system", "content": SYSTEM_MEMORY},
            {"role": "user", "content": user},
        ],
        temperature,
        max_tokens,
        timeout,
        max_retries=4,
        base_delay_sec=2.0,
    )


def _assemble_refined_md(
    *,
    meta_timestamp: str,
    paper_source: str,
    distill_model: str,
    memory_md: str,
    round_sections: list[tuple[int, int, str]],
    tail_md: str,
) -> str:
    lines = [
        "# 辩论记录（精炼阅读版）\n\n",
        f"> 由 `{distill_model}` 对完整 `progress.md` 各轮生成/审稿及大纲内容做摘要，便于回顾；**完整原文仍以 `progress.md` 为准**。\n\n",
        "## 元信息\n\n",
        f"- **生成时间**：{meta_timestamp}\n",
        f"- **来源**：`{paper_source}`\n\n",
        "## 已否决 idea（精炼）\n\n",
        memory_md + "\n\n",
    ]
    for m, r, body in round_sections:
        lines.append(f"## 尝试 {m} · 第 {r} 轮（精炼）\n\n")
        lines.append(body.strip() + "\n\n")
    lines.append("## 定稿与大纲（精炼）\n\n")
    lines.append(tail_md.strip() + "\n")
    return "".join(lines)


@dataclass
class RoundTriplet:
    macro: int
    round: int
    generator: str
    reviewer: str
    reflection: str


def parse_progress_markdown(text: str) -> tuple[str, list[str], list[RoundTriplet], str]:
    """
    解析 build_progress_markdown / 磁盘 progress.md 结构。
    返回：(preamble 片段), memory 条目列表, 各轮三元组, tail 从「最终用于大纲」起)。
    """
    mem: list[str] = []
    rounds_map: dict[tuple[int, int], dict[str, str]] = {}

    mem_m = re.search(
        r"## 已否决 idea 记忆（精炼摘要）\s*\n\n(.*?)(?=\n## 尝试 |\n## 最终用于大纲)",
        text,
        re.DOTALL,
    )
    preamble_end = 0
    if mem_m:
        block = mem_m.group(1).strip()
        for line in block.splitlines():
            line = line.strip()
            m_num = re.match(r"^(\d+)\.\s*(.+)$", line)
            if m_num:
                mem.append(m_num.group(2).strip())
        preamble_end = mem_m.end()
    else:
        mem_m2 = re.search(r"\n## 尝试 \d+", text)
        if mem_m2:
            preamble_end = mem_m2.start()

    tail = ""
    tail_m = re.search(r"## 最终用于大纲的标题与摘要", text)
    if tail_m:
        tail = text[tail_m.start() :].strip()

    body = text[preamble_end : tail_m.start() if tail_m else len(text)]

    parts = re.split(
        r"(?m)^## 尝试 (\d+) · 第 (\d+) 轮 · (生成器（proposal）|审稿人|生成器回应导师)\s*$",
        body,
    )
    i = 1
    while i + 3 < len(parts):
        try:
            ma = int(parts[i])
            rr = int(parts[i + 1])
            role = parts[i + 2]
            chunk = parts[i + 3].strip()
        except (ValueError, IndexError):
            break
        key = (ma, rr)
        if key not in rounds_map:
            rounds_map[key] = {"generator": "", "reviewer": "", "reflection": ""}
        if "proposal" in role:
            rounds_map[key]["generator"] = chunk
        elif role == "审稿人":
            rounds_map[key]["reviewer"] = chunk
        else:
            rounds_map[key]["reflection"] = chunk
        i += 4

    triplets: list[RoundTriplet] = []
    for (ma, rr) in sorted(rounds_map.keys(), key=lambda x: (x[0], x[1])):
        d = rounds_map[(ma, rr)]
        triplets.append(
            RoundTriplet(
                macro=ma,
                round=rr,
                generator=d.get("generator", ""),
                reviewer=d.get("reviewer", ""),
                reflection=d.get("reflection", ""),
            )
        )

    preamble = text[: preamble_end if preamble_end else 0][:4000]
    return preamble, mem, triplets, tail


def load_cache(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {"version": 1, "rounds": {}, "memory_hash": "", "tail_hash": ""}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {"version": 1, "rounds": {}, "memory_hash": "", "tail_hash": ""}


def save_cache(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def refresh_from_run_state(
    session_dir: Path,
    *,
    paper_source: str,
    meta_timestamp: str,
    rounds: list[Any],
    abandoned_memories: list[str],
    final_title_abstract: str,
    outline_draft: str,
    outline_review_feedback: str,
    outline_output: str,
    client: Any,
    model: str,
    temperature: float,
    max_tokens: int,
    timeout: float,
    dry_run: bool,
    verbose: bool,
) -> Path | None:
    """由 run() 内 state 增量更新 progress_refined.md；失败时打印警告并返回 None。"""
    try:
        return _refresh_impl(
            session_dir=session_dir,
            paper_source=paper_source,
            meta_timestamp=meta_timestamp,
            rounds_from_state=rounds,
            abandoned_memories=abandoned_memories,
            final_title_abstract=final_title_abstract,
            outline_draft=outline_draft,
            outline_review=outline_review_feedback,
            outline_output=outline_output,
            client=client,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout,
            dry_run=dry_run,
            verbose=verbose,
        )
    except Exception as e:
        print(f"[progress 精炼] 失败（已保留完整 progress.md）：{e}", file=sys.stderr, flush=True)
        return None


def _round_key(macro: int, round_id: int) -> str:
    return f"{macro}-{round_id}"


def _refresh_impl(
    *,
    session_dir: Path,
    paper_source: str,
    meta_timestamp: str,
    rounds_from_state: list[Any],
    abandoned_memories: list[str],
    final_title_abstract: str,
    outline_draft: str,
    outline_review: str,
    outline_output: str,
    client: Any,
    model: str,
    temperature: float,
    max_tokens: int,
    timeout: float,
    dry_run: bool,
    verbose: bool,
) -> Path:
    cache_path = session_dir / PROGRESS_REFINED_CACHE_FILENAME
    cache = load_cache(cache_path)
    rounds_cache: dict[str, Any] = cache.get("rounds") or {}

    round_sections: list[tuple[int, int, str]] = []

    for rec in rounds_from_state:
        ma = int(rec.macro_attempt)
        rr = int(rec.round)
        rk = _round_key(ma, rr)
        g = rec.generator_output or ""
        rv = rec.reviewer_output or ""
        rf = rec.reflection_output or ""
        h = _sha256(g + "\0" + rv + "\0" + rf)
        ent = rounds_cache.get(rk) or {}
        if dry_run:
            body = f"（dry-run 精炼占位）\n\n- 生成器约 {_chars_hint(len(g))} 字\n- 审稿人约 {_chars_hint(len(rv))} 字\n- 回应约 {_chars_hint(len(rf))} 字"
            rounds_cache[rk] = {"hash": h, "md": body}
        elif ent.get("hash") == h and ent.get("md"):
            body = ent["md"]
        else:
            if verbose:
                print(f"[progress 精炼] 轮次 M{ma}·R{rr} …", file=sys.stderr, flush=True)
            body = distill_round_block(
                client, model, g, rv, rf, temperature=temperature, max_tokens=max_tokens, timeout=timeout
            )
            rounds_cache[rk] = {"hash": h, "md": body}

    for rec in rounds_from_state:
        ma = int(rec.macro_attempt)
        rr = int(rec.round)
        rk = _round_key(ma, rr)
        ent = rounds_cache.get(rk) or {}
        round_sections.append((ma, rr, ent.get("md", "")))

    mem_hash = _sha256(json.dumps(abandoned_memories, ensure_ascii=False))
    if dry_run:
        mem_md = "（dry-run：已否决 idea 占位）"
    elif not abandoned_memories:
        mem_md = "（本轮尚无已否决记录）"
    elif cache.get("memory_hash") == mem_hash and cache.get("memory_md"):
        mem_md = cache["memory_md"]
    else:
        if verbose:
            print("[progress 精炼] 已否决 idea 摘要 …", file=sys.stderr, flush=True)
        mem_md = distill_memory_block(
            client, model, abandoned_memories, temperature=temperature, max_tokens=max_tokens, timeout=timeout
        )
        cache["memory_md"] = mem_md
        cache["memory_hash"] = mem_hash

    tail_src = "\0".join(
        [final_title_abstract or "", outline_draft or "", outline_review or "", outline_output or ""]
    )
    tail_h = _sha256(tail_src)
    if dry_run:
        tail_md = "（dry-run：定稿与大纲占位）"
    elif cache.get("tail_hash") == tail_h and cache.get("tail_md"):
        tail_md = cache["tail_md"]
    else:
        if verbose:
            print("[progress 精炼] 定稿与大纲 …", file=sys.stderr, flush=True)
        tail_md = distill_tail_block(
            client,
            model,
            final_title_abstract,
            outline_draft,
            outline_review,
            outline_output,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout,
        )
        cache["tail_md"] = tail_md
        cache["tail_hash"] = tail_h

    cache["rounds"] = rounds_cache
    cache["version"] = 1
    cache["distill_model"] = model

    out_md = _assemble_refined_md(
        meta_timestamp=meta_timestamp,
        paper_source=paper_source,
        distill_model=model,
        memory_md=mem_md,
        round_sections=round_sections,
        tail_md=tail_md,
    )
    out_path = session_dir / PROGRESS_REFINED_FILENAME
    session_dir.mkdir(parents=True, exist_ok=True)
    out_path.write_text(out_md, encoding="utf-8")
    save_cache(cache_path, cache)
    if verbose:
        print(f"[progress 精炼] 已写入 {out_path}", file=sys.stderr, flush=True)
    return out_path


def _chars_hint(n: int) -> str:
    if n >= 10_000:
        return f"{n / 1000:.1f}k"
    if n >= 1000:
        return f"~{n / 1000:.1f}k"
    return str(n)


def refresh_from_progress_file(
    progress_md_path: Path,
    *,
    base_url: str,
    api_key: str,
    model: str,
    temperature: float,
    max_tokens: int,
    timeout: float,
    dry_run: bool,
    verbose: bool,
) -> Path:
    """离线：读取 progress.md，解析后写入同目录 progress_refined.md（重建 cache）。"""
    text = progress_md_path.read_text(encoding="utf-8")
    _, mem, triplets, _tail_raw = parse_progress_markdown(text)

    # 从原文 tail 解析各段用于 distill_tail（比正则更稳：按小标题切）
    final_ta, od, orv, oo = _parse_tail_sections(text)

    session_dir = progress_md_path.parent
    cache_path = session_dir / PROGRESS_REFINED_CACHE_FILENAME
    rounds_cache: dict[str, Any] = {}

    client: Any = None
    if not dry_run:
        try:
            from openai import OpenAI
        except ImportError as e:
            raise RuntimeError("请安装 openai: pip install -r requirements.txt") from e
        nu = _get_normalize_url()
        client = OpenAI(base_url=nu(base_url), api_key=api_key)

    round_sections: list[tuple[int, int, str]] = []
    for t in triplets:
        rk = _round_key(t.macro, t.round)
        h = _sha256(t.generator + "\0" + t.reviewer + "\0" + t.reflection)
        if dry_run:
            body = f"（dry-run）M{t.macro}·R{t.round}"
        else:
            if verbose:
                print(f"[progress 精炼] 轮次 M{t.macro}·R{t.round} …", file=sys.stderr, flush=True)
            assert client is not None
            body = distill_round_block(
                client,
                model,
                t.generator,
                t.reviewer,
                t.reflection,
                temperature=temperature,
                max_tokens=max_tokens,
                timeout=timeout,
            )
        rounds_cache[rk] = {"hash": h, "md": body}
        round_sections.append((t.macro, t.round, body))

    mem_hash = _sha256(json.dumps(mem, ensure_ascii=False))
    if dry_run:
        mem_md = "（dry-run）"
    elif not mem:
        mem_md = "（无）"
    else:
        assert client is not None
        mem_md = distill_memory_block(
            client, model, mem, temperature=temperature, max_tokens=max_tokens, timeout=timeout
        )

    tail_h = _sha256("\0".join([final_ta, od, orv, oo]))
    if dry_run:
        tail_md = "（dry-run）"
    else:
        assert client is not None
        tail_md = distill_tail_block(
            client,
            model,
            final_ta,
            od,
            orv,
            oo,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout,
        )

    ts = ""
    m_ts = re.search(r"生成时间[：:]\s*([^\n]+)", text)
    if m_ts:
        ts = m_ts.group(1).strip()
    src = ""
    m_src = re.search(r"来源：`([^`]+)`", text)
    if m_src:
        src = m_src.group(1).strip()

    out_md = _assemble_refined_md(
        meta_timestamp=ts or "",
        paper_source=src or str(progress_md_path),
        distill_model=model,
        memory_md=mem_md,
        round_sections=round_sections,
        tail_md=tail_md,
    )
    out_path = session_dir / PROGRESS_REFINED_FILENAME
    out_path.write_text(out_md, encoding="utf-8")
    save_cache(
        cache_path,
        {
            "version": 1,
            "rounds": rounds_cache,
            "memory_hash": mem_hash,
            "memory_md": mem_md,
            "tail_hash": tail_h,
            "tail_md": tail_md,
            "distill_model": model,
        },
    )
    if verbose:
        print(f"[progress 精炼] 已写入 {out_path}", file=sys.stderr, flush=True)
    return out_path


def _parse_tail_sections(text: str) -> tuple[str, str, str, str]:
    """从 progress.md 抽取定稿与大纲四段。"""
    final_ta = ""
    od = ""
    orv = ""
    oo = ""

    m0 = re.search(
        r"## 最终用于大纲的标题与摘要\s*\n\n(.*?)(?=\n## |\Z)", text, re.DOTALL
    )
    if m0:
        final_ta = m0.group(1).strip()

    m1 = re.search(r"## 论文大纲初稿\s*\n\n(.*?)(?=\n## |\Z)", text, re.DOTALL)
    if m1:
        od = m1.group(1).strip()

    m2 = re.search(r"## 审稿人对大纲的意见\s*\n\n(.*?)(?=\n## |\Z)", text, re.DOTALL)
    if m2:
        orv = m2.group(1).strip()

    m3 = re.search(r"## 论文大纲（修订终稿）\s*\n\n(.*?)(?=\n## |\Z)", text, re.DOTALL)
    if m3:
        oo = m3.group(1).strip()

    return final_ta, od, orv, oo


def main() -> None:
    ap = argparse.ArgumentParser(description="将 progress.md 精炼为 progress_refined.md（gpt-5-codex-mini）")
    ap.add_argument(
        "target",
        type=Path,
        help="会话目录（内含 progress.md）或 progress.md 文件路径",
    )
    ap.add_argument("--base-url", default=os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1"))
    ap.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY", ""))
    ap.add_argument("--model", default=os.environ.get("PROGRESS_DISTILL_MODEL", DEFAULT_DISTILL_MODEL))
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--verbose", "-v", action="store_true")
    args = ap.parse_args()

    if not args.api_key and not args.dry_run:
        print("错误：请设置 OPENAI_API_KEY 或传入 --api-key", file=sys.stderr)
        sys.exit(1)

    target = args.target.resolve()
    if target.is_dir():
        pm = target / "progress.md"
        if not pm.is_file():
            print(f"错误：目录中无 progress.md：{pm}", file=sys.stderr)
            sys.exit(1)
        progress_path = pm
    elif target.is_file():
        progress_path = target
    else:
        print(f"错误：路径不存在：{target}", file=sys.stderr)
        sys.exit(1)

    cfg_path = Path(__file__).resolve().parent / "config.default.yaml"
    temp = 0.35
    max_tok = 8192
    to = 600.0
    if cfg_path.is_file():
        try:
            import yaml

            cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
            rc = cfg.get("run", {})
            temp = float(rc.get("progress_distill_temperature", temp))
            max_tok = int(rc.get("progress_distill_max_tokens", max_tok))
            to = float(rc.get("progress_distill_timeout_sec", to))
        except Exception:
            pass

    refresh_from_progress_file(
        progress_path,
        base_url=args.base_url,
        api_key=args.api_key,
        model=args.model,
        temperature=temp,
        max_tokens=max_tok,
        timeout=to,
        dry_run=args.dry_run,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    main()
