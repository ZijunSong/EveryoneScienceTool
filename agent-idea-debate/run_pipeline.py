#!/usr/bin/env python3
"""
双模型（OpenAI 兼容 API）论文 idea 辩论流水线 + 最终大纲生成。
支持按论文标题目录、每轮 checkpoint、断点恢复与 --fresh 新跑。
成功生成大纲时，会话目录会额外写入 outline_final.md（仅定稿大纲，便于归档与 rank_outlines 排序）。
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import shutil
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None  # type: ignore

from fetch_document import (
    SourceBundle,
    fetch_arxiv_related_context,
    load_source_with_meta,
    normalize_source,
)

CHECKPOINT_VERSION = 3
MAX_MACRO_IDEA_ATTEMPTS = 5

# 当 config 未提供 outline.review_prompt / revision_prompt 时使用
DEFAULT_OUTLINE_REVIEW_PROMPT = """你是 ICML/NeurIPS 风格的严格审稿人，正在审「论文大纲初稿」（不是审 idea 本身）。

已定稿的研究 idea（标题与摘要）：
{final_title_abstract}

论文大纲初稿：
{outline_draft}

种子论文摘录（节选，便于对照贡献边界）：
{paper_excerpt}

请完全使用中文，按以下结构输出：
第一部分：一句话结论（只能二选一：大纲可接受 / 大纲需修订）
第二部分：主要问题（若需修订：结构、主线、Related Work 覆盖、Method 是否可执行、实验是否对齐 claim；若可接受则说明仍可选的小改进）
第三部分：具体修改建议（条目化、可执行；若大纲可接受则给 1—2 条打磨建议即可）

请严肃、具体，不要泛泛而谈。"""

DEFAULT_OUTLINE_REVISION_PROMPT = """你是一位资深研究员。请根据审稿人对「大纲初稿」的意见，将大纲修订为一份可直接据此写作的终稿（Markdown 标题与层级清晰），全部使用中文。

【已定稿 idea】
{final_title_abstract}

【大纲初稿】
{outline_draft}

【审稿意见】
{outline_review}

请输出修订后的完整大纲（可重排结构，不必保留初稿的所有表述）。"""


def normalize_openai_base_url(url: str) -> str:
    u = url.strip().rstrip("/")
    if u.endswith("/v1"):
        return u
    return u + "/v1"


def deep_merge_prompt_round(
    base: dict[str, Any], round_key: str, fallback_key: str | None
) -> dict[str, str]:
    r = base.get(round_key)
    if isinstance(r, dict) and r:
        return r
    if fallback_key and fallback_key in base:
        fb = base[fallback_key]
        if isinstance(fb, dict):
            return fb
    return {}


def format_prompt(template: str, mapping: dict[str, Any]) -> str:
    try:
        return template.format(**mapping)
    except KeyError as e:
        raise KeyError(f"提示词占位符缺失: {e}; 可用键: {list(mapping.keys())}") from e


def _chars_hint(n: int) -> str:
    """简短字数提示，用于 verbose 日志。"""
    if n >= 10_000:
        return f"{n / 1000:.1f}k"
    if n >= 1000:
        return f"~{n / 1000:.1f}k"
    return str(n)


@dataclass
class RoundRecord:
    macro_attempt: int
    round: int
    generator_output: str
    reviewer_output: str
    reflection_output: str


@dataclass
class RunState:
    paper_source: str
    paper_text: str
    rounds: list[RoundRecord] = field(default_factory=list)
    final_title_abstract: str = ""
    outline_draft: str = ""
    outline_review_feedback: str = ""
    outline_output: str = ""


def build_transcript(state: RunState, paper_text: str, only_macro_attempt: int | None = None) -> str:
    lines: list[str] = []
    lines.append("=== 论文/Blog 摘录 ===\n")
    lines.append(paper_text[:12000])
    for r in state.rounds:
        if only_macro_attempt is not None and r.macro_attempt != only_macro_attempt:
            continue
        lines.append(f"\n\n--- 尝试 {r.macro_attempt} · 第 {r.round} 轮 · 生成器 proposal ---\n")
        lines.append(r.generator_output)
        lines.append(f"\n\n--- 尝试 {r.macro_attempt} · 第 {r.round} 轮 · 审稿人 ---\n")
        lines.append(r.reviewer_output)
        lines.append(f"\n\n--- 尝试 {r.macro_attempt} · 第 {r.round} 轮 · 生成器对导师的回应 ---\n")
        lines.append(r.reflection_output)
    return "\n".join(lines)


def reviewer_recommends_abandon(reviewer_text: str) -> bool:
    """审稿人「一句话结论」为「建议放弃」时返回 True；否则为 False（值得做 / 可做但需少量修改 等均视为未放弃）。"""
    t = reviewer_text.strip()
    if re.search(r"建议\s*放弃", t):
        return True
    # 兼容「第一部分：一句话结论」后紧跟的加粗行
    if re.search(r"^\s*[-*]?\s*\*?\s*建议\s*放弃", t, re.MULTILINE):
        return True
    return False


def generator_abandons(reflection: str) -> bool:
    """生成器在反思中决定放弃当前 idea（与审稿人一起否决该版本）。

    若审稿人本轮未写「建议放弃」，主循环会忽略该「放弃」表态（不消耗宏观尝试）。"""
    t = reflection.strip()
    if re.search(r"(?:不值得|不要|别|请勿)\s*放弃", t):
        return False
    if re.search(r"建议\s*放弃", t):
        return True
    if re.search(r"(认同|同意|总体认同).{0,120}放弃", t):
        return True
    if re.search(r"放弃.{0,25}(当前|这个|该|版本|路线|idea)", t, re.IGNORECASE):
        return True
    return False


def summarize_abandoned_idea(gen_out: str, ref_out: str, max_each: int = 900) -> str:
    g = re.sub(r"\s+", " ", gen_out.strip())[:max_each]
    r = re.sub(r"\s+", " ", ref_out.strip())[:max_each]
    return f"idea 要点：{g} | 放弃原因：{r}"


def format_memory_block(memory: list[str]) -> str:
    if not memory:
        return ""
    parts = ["【以下为先前已否决的 idea 摘要（须提出不同方向，避免重复）】"]
    for i, m in enumerate(memory, 1):
        parts.append(f"\n{i}. {m}\n")
    return "\n".join(parts) + "\n\n"


def _is_retriable_openai_error(exc: BaseException) -> bool:
    """网关偶发 403（配额预扣 SQL 错）、429、5xx、超时、断连等可重试。"""
    name = type(exc).__name__
    if name in ("RateLimitError", "APITimeoutError", "APIConnectionError"):
        return True
    if name == "PermissionDeniedError":
        s = str(exc).lower()
        if "pre_consume_token_quota_failed" in s or "transaction within a transaction" in s:
            return True
        if "quota" in s or "sql logic error" in s:
            return True
        return False
    code = getattr(exc, "status_code", None)
    if code in (429, 500, 502, 503, 504):
        return True
    if code == 403:
        s = str(exc).lower()
        return "pre_consume" in s or "quota" in s or "sql" in s
    if name == "InternalServerError":
        return True
    return False


def call_chat(
    client: Any,
    model: str,
    messages: list[dict[str, str]],
    temperature: float,
    max_tokens: int,
    timeout: float,
    *,
    max_retries: int = 6,
    base_delay_sec: float = 2.0,
) -> str:
    last: BaseException | None = None
    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                timeout=timeout,
            )
            choice = resp.choices[0]
            return (choice.message.content or "").strip()
        except Exception as e:
            last = e
            if attempt + 1 >= max_retries or not _is_retriable_openai_error(e):
                break
            wait = min(base_delay_sec * (2**attempt), 120.0) + random.uniform(0, 1.5)
            print(
                f"[API 重试] 第 {attempt + 1}/{max_retries} 次失败（{type(e).__name__}）：{e}\n"
                f"         {wait:.1f}s 后重试…（若为网关配额/SQL 偶发错误，重试常可恢复）",
                file=sys.stderr,
                flush=True,
            )
            time.sleep(wait)
    assert last is not None
    print(
        "\n提示：若为 403 + pre_consume_token_quota_failed，多为上游网关/余额扣费侧短暂故障，可稍后同一命令续跑（checkpoint 已保存）。\n",
        file=sys.stderr,
    )
    raise last


def _first_line_title_text(text: str, max_len: int = 120) -> str | None:
    for line in text.splitlines():
        line = line.strip()
        if len(line) >= 8 and len(line) <= max_len:
            return line
        if len(line) > max_len:
            return line[:max_len]
    return None


def make_paper_slug(title_hint: str | None, paper_text: str, fingerprint: str, max_len: int = 64) -> str:
    base = (title_hint or "").strip()
    base = re.sub(r"\$[^$]+\$", "", base)
    base = re.sub(r"\s+", " ", base).strip()
    if not base or len(base) < 3:
        base = _first_line_title_text(paper_text) or f"paper_{fingerprint[:8]}"
    s = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", base)
    s = re.sub(r"_+", "_", s).strip("_")[:max_len].strip("_")
    if not s:
        s = f"paper_{fingerprint[:8]}"
    return s


def resolve_unique_session_dir(out_dir: Path, slug: str, fingerprint: str) -> Path:
    """同名目录若已存在且 fingerprint 不同，则追加短 hash 避免混用。"""
    base = out_dir / slug
    if not base.is_dir():
        return base
    cp = base / "checkpoint.json"
    if not cp.is_file():
        return base
    try:
        old = json.loads(cp.read_text(encoding="utf-8"))
        if old.get("paper_fingerprint") == fingerprint:
            return base
    except Exception:
        pass
    return out_dir / f"{slug}_{fingerprint[:6]}"


def archive_session_dir(session_dir: Path) -> Path:
    """将当前会话目录下除 _archive 外的文件移到 _archive/<时间戳>/。"""
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    arch = session_dir / "_archive" / ts
    arch.mkdir(parents=True, exist_ok=True)
    for p in session_dir.iterdir():
        if p.name == "_archive":
            continue
        dest = arch / p.name
        if p.is_dir():
            shutil.copytree(p, dest)
            shutil.rmtree(p)
        else:
            shutil.move(str(p), str(dest))
    return arch


def state_from_checkpoint(cp: dict[str, Any]) -> RunState:
    st = RunState(paper_source=cp.get("source", ""), paper_text=cp.get("paper_excerpt", ""))
    st.final_title_abstract = cp.get("final_title_abstract") or ""
    st.outline_draft = cp.get("outline_draft") or ""
    st.outline_review_feedback = cp.get("outline_review_feedback") or ""
    st.outline_output = cp.get("outline") or ""
    for row in cp.get("rounds") or []:
        st.rounds.append(
            RoundRecord(
                macro_attempt=int(row.get("macro_attempt", 1)),
                round=int(row["round"]),
                generator_output=row.get("generator_output", ""),
                reviewer_output=row.get("reviewer_output", ""),
                reflection_output=row.get("reflection_output", ""),
            )
        )
    return st


def _extract_final_idea(text: str) -> str | None:
    if not text:
        return None
    title_m = re.search(r"(?:标题|题目)[:：]\s*(.+?)(?:\n|$)", text, re.DOTALL)
    abs_m = re.search(r"摘要[:：]\s*(.+?)(?:\n\n|---|\Z)", text, re.DOTALL)
    if title_m and abs_m:
        return f"标题：{title_m.group(1).strip()}\n\n摘要：{abs_m.group(1).strip()}"
    return None


def _final_idea_from_proposal(gen_out: str, reflection_out: str | None) -> str:
    t = _extract_final_idea(gen_out)
    if t:
        return t
    if reflection_out:
        t2 = _extract_final_idea(reflection_out)
        if t2:
            return t2
    return "## 标题与摘要（生成器 proposal）\n\n" + gen_out.strip()


def build_progress_markdown(state: RunState, meta: dict[str, Any]) -> str:
    """progress.md：不含论文摘录。"""
    md_lines = [
        f"# Idea 辩论流水线记录\n\n生成时间：{meta.get('timestamp')}\n",
        f"## 输入\n\n- 来源：`{state.paper_source}`\n",
    ]
    mem = meta.get("abandoned_memories") or []
    if mem:
        md_lines.append("\n## 已否决 idea 记忆（精炼摘要）\n\n")
        for i, m in enumerate(mem, 1):
            md_lines.append(f"{i}. {m}\n\n")
    for x in state.rounds:
        md_lines.append(
            f"\n## 尝试 {x.macro_attempt} · 第 {x.round} 轮 · 生成器（proposal）\n\n{x.generator_output}\n"
        )
        md_lines.append(f"\n## 尝试 {x.macro_attempt} · 第 {x.round} 轮 · 审稿人\n\n{x.reviewer_output}\n")
        md_lines.append(f"\n## 尝试 {x.macro_attempt} · 第 {x.round} 轮 · 生成器回应导师\n\n{x.reflection_output}\n")

    md_lines.append("\n## 最终用于大纲的标题与摘要\n\n")
    md_lines.append(state.final_title_abstract or "（尚未生成）")
    if state.outline_draft.strip():
        md_lines.append("\n\n## 论文大纲初稿\n\n")
        md_lines.append(state.outline_draft)
    if state.outline_review_feedback.strip():
        md_lines.append("\n\n## 审稿人对大纲的意见\n\n")
        md_lines.append(state.outline_review_feedback)
    md_lines.append("\n\n## 论文大纲（修订终稿）\n\n")
    md_lines.append(state.outline_output or "（尚未生成或已跳过）")
    return "".join(md_lines)


def build_final_markdown(
    state: RunState,
    meta: dict[str, Any],
    paper_excerpt_limit: int = 20000,
) -> str:
    """final.md：保留论文摘录便于归档。"""
    md_lines = [
        f"# Idea 辩论流水线记录（完整归档）\n\n生成时间：{meta.get('timestamp')}\n",
        f"## 输入\n\n- 来源：`{state.paper_source}`\n",
        "## 论文摘录\n\n```\n",
        state.paper_text[:paper_excerpt_limit],
        "\n```\n",
    ]
    mem = meta.get("abandoned_memories") or []
    if mem:
        md_lines.append("\n## 已否决 idea 记忆\n\n")
        for i, m in enumerate(mem, 1):
            md_lines.append(f"{i}. {m}\n\n")
    for x in state.rounds:
        md_lines.append(
            f"\n## 尝试 {x.macro_attempt} · 第 {x.round} 轮 · 生成器（proposal）\n\n{x.generator_output}\n"
        )
        md_lines.append(f"\n## 尝试 {x.macro_attempt} · 第 {x.round} 轮 · 审稿人\n\n{x.reviewer_output}\n")
        md_lines.append(f"\n## 尝试 {x.macro_attempt} · 第 {x.round} 轮 · 生成器回应导师\n\n{x.reflection_output}\n")

    md_lines.append("\n## 最终用于大纲的标题与摘要\n\n")
    md_lines.append(state.final_title_abstract or "（尚未生成）")
    if state.outline_draft.strip():
        md_lines.append("\n\n## 论文大纲初稿\n\n")
        md_lines.append(state.outline_draft)
    if state.outline_review_feedback.strip():
        md_lines.append("\n\n## 审稿人对大纲的意见\n\n")
        md_lines.append(state.outline_review_feedback)
    md_lines.append("\n\n## 论文大纲（修订终稿）\n\n")
    md_lines.append(state.outline_output or "（尚未生成或已跳过）")
    return "".join(md_lines)


OUTLINE_FINAL_FILENAME = "outline_final.md"


def build_outline_final_markdown(state: RunState, meta: dict[str, Any]) -> str:
    """仅含定稿大纲的独立归档（不含辩论全文与论文摘录）。"""
    parts = [
        "# 论文大纲（定稿）\n\n",
        f"- **生成时间**：{meta.get('timestamp', '')}\n",
        f"- **种子来源**：`{state.paper_source}`\n\n",
        "## Idea（标题与摘要）\n\n",
        (state.final_title_abstract or "（无）").strip(),
        "\n\n---\n\n## 大纲正文\n\n",
        (state.outline_output or "").strip(),
        "\n",
    ]
    return "".join(parts)


def write_checkpoint(
    session_dir: Path,
    state: RunState,
    meta: dict[str, Any],
    *,
    bundle: SourceBundle,
    paper_slug: str,
    phase: str,
    status: str,
    models: dict[str, str],
    abandoned_memories: list[str],
    current_macro_attempt: int,
    inner_next_round: int,
) -> None:
    session_dir.mkdir(parents=True, exist_ok=True)
    meta = dict(meta)
    meta["abandoned_memories"] = list(abandoned_memories)
    data = {
        "version": CHECKPOINT_VERSION,
        "source": state.paper_source,
        "source_normalized": bundle.source_normalized,
        "paper_fingerprint": bundle.fingerprint,
        "paper_title": bundle.title_hint,
        "paper_slug": paper_slug,
        "meta": meta,
        "extra_notes": meta.get("extra_notes", ""),
        "paper_excerpt": state.paper_text,
        "abandoned_memories": list(abandoned_memories),
        "current_macro_attempt": current_macro_attempt,
        "inner_next_round": inner_next_round,
        "rounds": [
            {
                "macro_attempt": x.macro_attempt,
                "round": x.round,
                "generator_output": x.generator_output,
                "reviewer_output": x.reviewer_output,
                "reflection_output": x.reflection_output,
            }
            for x in state.rounds
        ],
        "final_title_abstract": state.final_title_abstract,
        "outline_draft": state.outline_draft,
        "outline_review_feedback": state.outline_review_feedback,
        "outline": state.outline_output,
        "phase": phase,
        "status": status,
        "models": models,
    }
    path = session_dir / "checkpoint.json"
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    progress = session_dir / "progress.md"
    progress.write_text(build_progress_markdown(state, meta), encoding="utf-8")


def save_final_artifacts(
    session_dir: Path, state: RunState, meta: dict[str, Any]
) -> tuple[Path, Path, Path, Path | None]:
    session_dir.mkdir(parents=True, exist_ok=True)
    base_data = {
        "meta": meta,
        "extra_notes": meta.get("extra_notes", ""),
        "paper_source": state.paper_source,
        "paper_excerpt": state.paper_text,
        "abandoned_memories": meta.get("abandoned_memories", []),
        "rounds": [
            {
                "macro_attempt": x.macro_attempt,
                "round": x.round,
                "generator_output": x.generator_output,
                "reviewer_output": x.reviewer_output,
                "reflection_output": x.reflection_output,
            }
            for x in state.rounds
        ],
        "final_title_abstract": state.final_title_abstract,
        "outline_draft": state.outline_draft,
        "outline_review_feedback": state.outline_review_feedback,
        "outline": state.outline_output,
    }
    json_path = session_dir / "final.json"
    json_path.write_text(json.dumps(base_data, ensure_ascii=False, indent=2), encoding="utf-8")

    md_path = session_dir / "final.md"
    md_path.write_text(build_final_markdown(state, meta), encoding="utf-8")

    txt_path = session_dir / "final_debate.txt"
    txt_path.write_text(md_path.read_text(encoding="utf-8"), encoding="utf-8")

    outline_final_path: Path | None = None
    if (state.outline_output or "").strip():
        outline_final_path = session_dir / OUTLINE_FINAL_FILENAME
        outline_final_path.write_text(build_outline_final_markdown(state, meta), encoding="utf-8")

    return json_path, md_path, txt_path, outline_final_path


def write_failure_no_idea(
    session_dir: Path,
    *,
    paper_source: str,
    memory: list[str],
    meta: dict[str, Any],
) -> None:
    session_dir.mkdir(parents=True, exist_ok=True)
    lines = [
        "# 无法依据该文给出较好 idea\n\n",
        f"来源：`{paper_source}`\n\n",
        "在允许的最大次数内，生成器对每一版 idea 与审稿人商榷后均选择放弃。",
        "这通常表示：在当前约束与设定下，难以从该论文可靠地提炼出符合 oral 标准的独立研究问题。\n\n",
        "## 已尝试并否决的方向摘要\n\n",
    ]
    for i, m in enumerate(memory, 1):
        lines.append(f"{i}. {m}\n\n")
    lines.append("\n---\n\n")
    lines.append(f"meta: {json.dumps(meta, ensure_ascii=False)}\n")
    p = session_dir / "failure_no_idea.md"
    p.write_text("".join(lines), encoding="utf-8")


def run(
    *,
    source: str,
    bundle: SourceBundle,
    base_url: str,
    api_key: str,
    default_model: str,
    generator_model: str | None,
    reviewer_model: str | None,
    outline_model: str | None,
    config_path: Path,
    max_rounds: int,
    max_macro_attempts: int,
    session_dir: Path,
    dry_run: bool,
    extra_notes: str,
    skip_outline: bool,
    resume_state: RunState | None,
    memory_log: list[str] | None,
    resume_macro_attempt: int,
    resume_inner_start_round: int,
    phase_resume: str | None,
    meta: dict[str, Any],
    paper_slug: str,
    related_work_context: str,
    min_debate_rounds_before_outline: int,
    verbose_progress: bool,
) -> RunState:
    if OpenAI is None:
        raise RuntimeError("请安装 openai: pip install -r requirements.txt")

    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    run_cfg = cfg.get("run", {})
    temperature = float(run_cfg.get("temperature", 0.7))
    max_tokens = int(run_cfg.get("max_tokens", 8192))
    timeout = float(run_cfg.get("request_timeout_sec", 600))
    api_max_retries = int(run_cfg.get("api_max_retries", 6))
    api_retry_base_delay_sec = float(run_cfg.get("api_retry_base_delay_sec", 2.0))

    models_cfg = cfg.get("models") or {}
    gen_m = generator_model or models_cfg.get("generator") or default_model
    rev_m = reviewer_model or models_cfg.get("reviewer") or default_model
    out_m = outline_model or models_cfg.get("outline") or default_model
    models = {"generator": gen_m, "reviewer": rev_m, "outline": out_m}

    client = OpenAI(base_url=normalize_openai_base_url(base_url), api_key=api_key)

    def log(msg: str) -> None:
        print(msg, flush=True, file=sys.stderr)

    def api_chat(model: str, messages: list[dict[str, str]]) -> str:
        return call_chat(
            client,
            model,
            messages,
            temperature,
            max_tokens,
            timeout,
            max_retries=api_max_retries,
            base_delay_sec=api_retry_base_delay_sec,
        )

    paper_text = bundle.text
    state = resume_state or RunState(paper_source=source, paper_text=paper_text)
    if not state.paper_text:
        state.paper_text = paper_text

    memory: list[str] = list(memory_log or [])
    meta = dict(meta)
    meta["abandoned_memories"] = list(memory)

    prompts_root = cfg.get("prompts", {})
    fallback_multi = prompts_root.get("fallback_multi_round", "round2")

    macro_attempt = resume_macro_attempt
    inner_start = resume_inner_start_round

    def checkpoint(phase: str, status: str) -> None:
        write_checkpoint(
            session_dir,
            state,
            meta,
            bundle=bundle,
            paper_slug=paper_slug,
            phase=phase,
            status=status,
            models=models,
            abandoned_memories=memory,
            current_macro_attempt=macro_attempt,
            inner_next_round=inner_start,
        )
        if verbose_progress:
            log(f"[进度] 已保存 checkpoint（phase={phase}, status={status}）→ {session_dir / 'checkpoint.json'}")
        else:
            log(f"[进度] ckpt {phase}/{status}")

    enable_outline_review = bool(run_cfg.get("enable_outline_review", True))

    def run_outline_step() -> None:
        nonlocal state
        outline_cfg = cfg.get("outline") or {}
        outline_t = (outline_cfg.get("prompt") or "").strip()
        if not outline_t:
            raise RuntimeError("配置 outline.prompt 为空")
        outline_user = format_prompt(
            outline_t, {"final_title_abstract": state.final_title_abstract, "paper_text": paper_text}
        )
        if skip_outline:
            log("[进度] 跳过大纲（--skip-outline）")
            state.outline_draft = ""
            state.outline_review_feedback = ""
            state.outline_output = ""
            return
        if dry_run:
            state.outline_draft = "[dry-run] outline draft"
            state.outline_review_feedback = "[dry-run] outline review"
            state.outline_output = state.outline_draft + "\n\n--- 修订 ---\n" + state.outline_review_feedback
            return
        if verbose_progress:
            log(f"[进度] 大纲 · ① 初稿 ({out_m}) …")
        else:
            log("[进度] 大纲 ① 初稿 …")
        draft = api_chat(
            out_m,
            [
                {"role": "system", "content": "你是一位资深研究员，按要求用中文输出论文大纲。"},
                {"role": "user", "content": outline_user},
            ],
        )
        state.outline_draft = draft
        if verbose_progress:
            log(f"[进度] 大纲初稿完成（约 {len(draft)} 字）。")
        else:
            log(f"[进度] 大纲 ① ✓ {_chars_hint(len(draft))}字")
        if not enable_outline_review:
            state.outline_review_feedback = ""
            state.outline_output = draft
            log("[进度] 大纲 无审稿修订（enable_outline_review=false）")
            return
        review_t = (outline_cfg.get("review_prompt") or "").strip() or DEFAULT_OUTLINE_REVIEW_PROMPT
        paper_excerpt = paper_text[:8000]
        review_user = format_prompt(
            review_t,
            {
                "final_title_abstract": state.final_title_abstract,
                "outline_draft": draft,
                "paper_excerpt": paper_excerpt,
                "paper_text": paper_text,
            },
        )
        if verbose_progress:
            log(f"[进度] 大纲 · ② 审稿人审大纲 ({rev_m}) …")
        else:
            log("[进度] 大纲 ② 审稿 …")
        outline_review = api_chat(
            rev_m,
            [
                {"role": "system", "content": "你是一位严格的顶会审稿人，按要求用中文审大纲。"},
                {"role": "user", "content": review_user},
            ],
        )
        state.outline_review_feedback = outline_review
        if verbose_progress:
            log(f"[进度] 大纲审稿完成（约 {len(outline_review)} 字）。")
        else:
            log(f"[进度] 大纲 ② ✓ {_chars_hint(len(outline_review))}字")
        revision_t = (outline_cfg.get("revision_prompt") or "").strip() or DEFAULT_OUTLINE_REVISION_PROMPT
        revision_user = format_prompt(
            revision_t,
            {
                "final_title_abstract": state.final_title_abstract,
                "outline_draft": draft,
                "outline_review": outline_review,
                "paper_text": paper_text,
            },
        )
        if verbose_progress:
            log(f"[进度] 大纲 · ③ 据审稿意见修订终稿 ({out_m}) …")
        else:
            log("[进度] 大纲 ③ 修订终稿 …")
        state.outline_output = api_chat(
            out_m,
            [
                {"role": "system", "content": "你是一位资深研究员，根据审稿意见把大纲修订为可执行的终稿，全部使用中文。"},
                {"role": "user", "content": revision_user},
            ],
        )
        if verbose_progress:
            log(f"[进度] 大纲终稿完成（约 {len(state.outline_output)} 字）。")
        else:
            log(f"[进度] 大纲 ③ ✓ {_chars_hint(len(state.outline_output))}字")

    # 仅恢复大纲阶段
    if phase_resume == "outline" and state.rounds:
        last = state.rounds[-1]
        state.final_title_abstract = _final_idea_from_proposal(last.generator_output, last.reflection_output)
        run_outline_step()
        checkpoint("done", "completed")
        save_final_artifacts(session_dir, state, meta)
        return state

    def load_prompts(r: int) -> tuple[str, str, str]:
        rk = f"round{r}"
        pr = prompts_root.get(rk) or {}
        if r == 1:
            gen_t = (pr.get("generator") or "").strip()
            rev_t = (pr.get("reviewer") or "").strip()
            ref_t = (pr.get("reflection") or "").strip()
        else:
            merged = deep_merge_prompt_round(prompts_root, rk, fallback_multi)
            gen_t = (merged.get("generator") or "").strip()
            rev_t = (merged.get("reviewer") or "").strip()
            ref_t = (merged.get("reflection") or "").strip()
            if not gen_t:
                merged = deep_merge_prompt_round(prompts_root, "round2", None)
                gen_t = (merged.get("generator") or "").strip()
            if not rev_t:
                merged = deep_merge_prompt_round(prompts_root, "round2", None)
                rev_t = (merged.get("reviewer") or "").strip()
            if not ref_t:
                merged = deep_merge_prompt_round(prompts_root, "round2", None)
                ref_t = (merged.get("reflection") or "").strip()
        return gen_t, rev_t, ref_t

    # 失败后 checkpoint 常含 current_macro_attempt = max+1；续跑时 while 条件为假，勿落到文件末尾
    if macro_attempt > max_macro_attempts:
        log(
            f"[进度] 宏观尝试序号 {macro_attempt} 已超过上限 {max_macro_attempts}，"
            "本会话已结束（与「无法产出 idea」一致）。"
        )
        fail_md = session_dir / "failure_no_idea.md"
        if not fail_md.is_file():
            write_failure_no_idea(session_dir, paper_source=source, memory=memory, meta=meta)
        checkpoint("failed", "abandoned_all")
        print(
            f"\n该论文宏观尝试已用尽。说明：{fail_md}\n"
            "若需重新开题，请加 --fresh。\n",
            file=sys.stderr,
        )
        sys.exit(3)

    while macro_attempt <= max_macro_attempts:
        meta["abandoned_memories"] = list(memory)
        if verbose_progress:
            log(
                f"[进度] === 宏观尝试 {macro_attempt}/{max_macro_attempts}（内层每尝试最多 {max_rounds} 轮；"
                f"积极结论下定稿前至少 {min_debate_rounds_before_outline} 轮完整 rebuttal）==="
            )
        else:
            log(
                f"[进度] === 宏观{macro_attempt}/{max_macro_attempts} · "
                f"≤{max_rounds}轮/试 · 积极定稿≥{min_debate_rounds_before_outline}轮 ==="
            )

        for r in range(inner_start, max_rounds + 1):
            tag = f"M{macro_attempt}·R{r}/{max_rounds}"
            gen_t, rev_t, ref_t = load_prompts(r)
            full_transcript = build_transcript(state, paper_text, only_macro_attempt=macro_attempt)
            gen_user = format_memory_block(memory) + format_prompt(
                gen_t,
                {
                    "paper_text": paper_text,
                    "related_work_context": related_work_context,
                    "round": r,
                    "full_transcript": full_transcript,
                    "generator_output": state.rounds[-1].generator_output if state.rounds else "",
                    "reviewer_output": state.rounds[-1].reviewer_output if state.rounds else "",
                },
            )
            if extra_notes.strip():
                gen_user += "\n\n【用户额外补充说明】\n" + extra_notes.strip()
            gen_messages = [
                {"role": "system", "content": "你是一位严谨的科研合作者，按要求用中文输出。"},
                {"role": "user", "content": gen_user},
            ]

            if verbose_progress:
                log(f"[进度] 尝试 {macro_attempt} · 第 {r}/{max_rounds} 轮 · ① 生成器 ({gen_m}) …")
            else:
                log(f"[进度] {tag} ①生成 …")
            if dry_run:
                gen_out = "[dry-run] generator"
            else:
                gen_out = api_chat(gen_m, gen_messages)
            if verbose_progress:
                log(f"[进度] ① 生成器完成（约 {len(gen_out)} 字）")
            else:
                log(f"[进度] {tag} ①✓ {_chars_hint(len(gen_out))}字")

            rev_user = format_prompt(
                rev_t,
                {
                    "paper_text": paper_text,
                    "round": r,
                    "full_transcript": full_transcript,
                    "generator_output": gen_out,
                    "reviewer_output": "",
                },
            )
            rev_messages = [
                {"role": "system", "content": "你是一位严格的顶会审稿人，按要求用中文输出。"},
                {"role": "user", "content": rev_user},
            ]
            if verbose_progress:
                log(f"[进度] 尝试 {macro_attempt} · 第 {r}/{max_rounds} 轮 · ② 审稿人 ({rev_m}) …")
            else:
                log(f"[进度] {tag} ②审稿 …")
            if dry_run:
                rev_out = (
                    "[dry-run] reviewer\n\n第一部分：一句话结论\n\n**建议放弃**\n\n"
                    "（dry-run：默认模拟审稿人建议放弃，以覆盖完整辩论分支；实跑以模型为准。）"
                )
            else:
                rev_out = api_chat(rev_m, rev_messages)
            if verbose_progress:
                log(f"[进度] ② 审稿人完成（约 {len(rev_out)} 字）")
            else:
                log(f"[进度] {tag} ②✓ {_chars_hint(len(rev_out))}字")

            ref_user = format_prompt(
                ref_t,
                {
                    "paper_text": paper_text,
                    "round": r,
                    "full_transcript": full_transcript,
                    "generator_output": gen_out,
                    "reviewer_output": rev_out,
                },
            )
            ref_messages = [
                {"role": "system", "content": "你是一位 Agent 方向博士生，按要求用中文反思与决策。"},
                {"role": "user", "content": ref_user},
            ]
            if verbose_progress:
                log(f"[进度] 尝试 {macro_attempt} · 第 {r}/{max_rounds} 轮 · ③ 生成器回应导师 ({gen_m}) …")
            else:
                log(f"[进度] {tag} ③回应 …")
            if dry_run:
                ref_out = "[dry-run] reflection"
            else:
                ref_out = api_chat(gen_m, ref_messages)
            if verbose_progress:
                log(f"[进度] ③ 完成（约 {len(ref_out)} 字）")
            else:
                log(f"[进度] {tag} ③✓ {_chars_hint(len(ref_out))}字")

            state.rounds.append(
                RoundRecord(
                    macro_attempt=macro_attempt,
                    round=r,
                    generator_output=gen_out,
                    reviewer_output=rev_out,
                    reflection_output=ref_out,
                )
            )
            inner_start = r + 1
            checkpoint("debating", "in_progress")

            if generator_abandons(ref_out):
                # 审稿人未写「建议放弃」时，生成器单方面「放弃」与流程矛盾，不记入宏观否决
                if not reviewer_recommends_abandon(rev_out):
                    if verbose_progress:
                        log(
                            "[进度] 生成器表态放弃，但审稿人本轮未建议放弃 → 单方放弃无效，"
                            "不消耗宏观尝试，继续下一轮。"
                        )
                    else:
                        log(f"[进度] {tag} 放弃主张无效（审稿人未否定）→ 续")
                    if r >= max_rounds:
                        if verbose_progress:
                            log(
                                "[进度] 已达最大内层轮次，仍以当前 proposal/回应定稿并生成大纲"
                                "（不视为宏观放弃）。"
                            )
                        else:
                            log(f"[进度] {tag} 已达最大轮次 → 定稿（单方放弃无效）")
                        state.final_title_abstract = _final_idea_from_proposal(gen_out, ref_out)
                        checkpoint("outline", "in_progress")
                        run_outline_step()
                        checkpoint("done", "completed")
                        save_final_artifacts(session_dir, state, meta)
                        return state
                    continue

                memory.append(summarize_abandoned_idea(gen_out, ref_out))
                meta["abandoned_memories"] = list(memory)
                macro_attempt += 1
                inner_start = 1
                if verbose_progress:
                    log(f"[进度] 生成器决定放弃当前 idea，已记入记忆；下一宏观尝试 = {macro_attempt}")
                else:
                    log(f"[进度] 宏观放弃 → {macro_attempt}/{max_macro_attempts}")
                if macro_attempt > max_macro_attempts:
                    write_failure_no_idea(session_dir, paper_source=source, memory=memory, meta=meta)
                    checkpoint("failed", "abandoned_all")
                    print(
                        f"\n未能从该论文得到稳定 idea：已连续 {max_macro_attempts} 次在商榷后放弃。\n"
                        f"说明已写入：{session_dir / 'failure_no_idea.md'}\n",
                        file=sys.stderr,
                    )
                    sys.exit(3)
                checkpoint("debating", "in_progress")
                break

            positive = not reviewer_recommends_abandon(rev_out)
            if positive and r >= min_debate_rounds_before_outline:
                if verbose_progress:
                    log(
                        f"[进度] 审稿人未建议放弃，且已完成至少 {min_debate_rounds_before_outline} 轮"
                        f"「生成→审稿→回应」→ 定稿并进入大纲。"
                    )
                else:
                    log(f"[进度] {tag} 审稿人未否定+轮次≥{min_debate_rounds_before_outline} → 定稿")
                state.final_title_abstract = _final_idea_from_proposal(gen_out, ref_out)
                checkpoint("outline", "in_progress")
                run_outline_step()
                checkpoint("done", "completed")
                save_final_artifacts(session_dir, state, meta)
                return state

            if r >= max_rounds:
                if verbose_progress:
                    log("[进度] 内层轮次已用尽，以最后一版 proposal/回应 定稿并生成大纲。")
                else:
                    log(f"[进度] {tag} 内层用尽 → 定稿")
                state.final_title_abstract = _final_idea_from_proposal(gen_out, ref_out)
                checkpoint("outline", "in_progress")
                run_outline_step()
                checkpoint("done", "completed")
                save_final_artifacts(session_dir, state, meta)
                return state

            if positive and r < min_debate_rounds_before_outline:
                if verbose_progress:
                    log(
                        f"[进度] 审稿人未建议放弃，但尚未达到最少辩论轮次（当前已完成 {r} 轮，需至少 "
                        f"{min_debate_rounds_before_outline} 轮完整 rebuttal）→ 继续下一轮。"
                    )
                else:
                    log(f"[进度] {tag} 审稿人未否定，未达最少轮次({r}/{min_debate_rounds_before_outline})→ 续")
            else:
                if verbose_progress:
                    log("[进度] 审稿人建议放弃或认为需大幅修订 → 继续下一轮迭代。")
                else:
                    log(f"[进度] {tag} 审稿人否定 → 续")
        else:
            # 内层 for 未执行（例如 inner_start > max_rounds）或异常：尽力落盘
            if verbose_progress:
                log("[进度] 内层辩论轮次未正常推进，尝试以最后一轮记录定稿并生成大纲。")
            else:
                log("[进度] 内层未执行/异常 → 尽力定稿")
            if not state.rounds:
                log("[进度] 内部异常：无辩论记录。")
                sys.exit(1)
            last = state.rounds[-1]
            state.final_title_abstract = _final_idea_from_proposal(last.generator_output, last.reflection_output)
            checkpoint("outline", "in_progress")
            run_outline_step()
            checkpoint("done", "completed")
            save_final_artifacts(session_dir, state, meta)
            return state

    log("[进度] 内部异常：主循环未正常 return（请附带 checkpoint.json 反馈）。")
    sys.exit(1)


def load_checkpoint(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def main() -> None:
    ap = argparse.ArgumentParser(description="双模型论文 idea 辩论 + 大纲（OpenAI 兼容 API）")
    ap.add_argument("source", help="论文/Blog URL，或本地 .pdf/.html/.txt 路径")
    ap.add_argument("--base-url", default=os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1"))
    ap.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY", ""))
    ap.add_argument("--model", default=os.environ.get("OPENAI_MODEL", "gpt-4o"), help="默认模型（三角色未单独指定时用）")
    ap.add_argument("--generator-model", default=None)
    ap.add_argument("--reviewer-model", default=None)
    ap.add_argument("--outline-model", default=None)
    ap.add_argument("--config", type=Path, default=Path(__file__).resolve().parent / "config.default.yaml")
    ap.add_argument("--max-chars", type=int, default=80000, help="拉取正文最大字符（防超长）")
    ap.add_argument("--max-rounds", type=int, default=None, help="覆盖配置文件中的 max_rounds")
    ap.add_argument(
        "--min-debate-rounds",
        type=int,
        default=None,
        help="审稿人未建议放弃时，至少完成多少轮「生成→审稿→回应」才允许定稿进大纲（默认读配置 min_debate_rounds_before_outline，且不超过 max_rounds）",
    )
    ap.add_argument("--out-dir", type=Path, default=Path(__file__).resolve().parent / "outputs")
    ap.add_argument("--dry-run", action="store_true", help="不调用 API，仅测试拉取与落盘")
    ap.add_argument(
        "--notes",
        default="",
        help="附加给每轮「生成器」用户消息的说明（偏好、约束、对论文的额外背景等）",
    )
    ap.add_argument("--skip-outline", action="store_true", help="仅跑辩论轮次，不调用大纲模型（省费用）")
    ap.add_argument(
        "--fresh",
        action="store_true",
        help="同一来源也强制重新跑：将当前会话目录内容归档到 _archive/ 后从头生成新 idea",
    )
    ap.add_argument(
        "--no-resume",
        action="store_true",
        help="不尝试从 checkpoint 恢复（仍会写入同一论文目录；与 --fresh 二选一即可）",
    )
    ap.add_argument(
        "--ignore-fingerprint",
        action="store_true",
        help="checkpoint 中论文指纹与当前文件不一致时仍尝试恢复（慎用）",
    )
    ap.add_argument(
        "--max-idea-attempts",
        type=int,
        default=None,
        help="生成器在放弃 idea 后，带记忆重新开题的最大宏观次数（默认 5）",
    )
    ap.add_argument(
        "--no-related-search",
        action="store_true",
        help="关闭从 arXiv 自动检索相关工作摘要（生成器仅依赖种子论文与模型自身知识）",
    )
    ap.add_argument(
        "--verbose-progress",
        action="store_true",
        help="详细进度（字数、完整 checkpoint 路径等）；也可在配置 run.verbose_progress: true",
    )

    args = ap.parse_args()
    if not args.api_key and not args.dry_run:
        print("错误：请设置 --api-key 或环境变量 OPENAI_API_KEY", file=sys.stderr)
        sys.exit(1)

    cfg_path = args.config
    if not cfg_path.is_file():
        print(f"找不到配置文件: {cfg_path}", file=sys.stderr)
        sys.exit(1)

    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    max_rounds = args.max_rounds if args.max_rounds is not None else int(cfg.get("run", {}).get("max_rounds", 5))
    run_cfg_main = cfg.get("run", {})
    min_debate_cfg = int(run_cfg_main.get("min_debate_rounds_before_outline", 2))
    min_debate_rounds = args.min_debate_rounds if args.min_debate_rounds is not None else min_debate_cfg
    min_debate_rounds = max(1, min(min_debate_rounds, max_rounds))
    verbose_progress = bool(args.verbose_progress or run_cfg_main.get("verbose_progress", False))
    max_idea_attempts = (
        args.max_idea_attempts
        if args.max_idea_attempts is not None
        else int(cfg.get("run", {}).get("max_idea_attempts", MAX_MACRO_IDEA_ATTEMPTS))
    )
    if max_idea_attempts < 1:
        print("错误：--max-idea-attempts / max_idea_attempts 至少为 1。", file=sys.stderr)
        sys.exit(1)

    meta = {
        "timestamp": datetime.now().isoformat(),
        "base_url": args.base_url,
        "model_default": args.model,
        "generator_model": args.generator_model,
        "reviewer_model": args.reviewer_model,
        "outline_model": args.outline_model,
        "max_rounds": max_rounds,
        "min_debate_rounds_before_outline": min_debate_rounds,
        "config": str(cfg_path),
        "extra_notes": args.notes,
        "skip_outline": args.skip_outline,
        "max_idea_attempts": max_idea_attempts,
        "no_related_search": args.no_related_search,
        "verbose_progress": verbose_progress,
    }

    def log(msg: str) -> None:
        print(msg, flush=True, file=sys.stderr)

    log(f"[进度] 拉取 {args.source!r} …")
    bundle = load_source_with_meta(args.source, args.max_chars)
    if verbose_progress:
        log(f"[进度] 正文就绪（约 {len(bundle.text)} 字）· fp={bundle.fingerprint[:12]}…")
    else:
        log(f"[进度] 正文 {_chars_hint(len(bundle.text))}字 · fp={bundle.fingerprint[:12]}…")

    run_cfg = cfg.get("run", {})
    if args.no_related_search:
        related_work_context = (
            "（已关闭 arXiv 相关工作检索；请仅依据种子论文与自身知识做领域调研。）"
        )
    elif args.dry_run:
        related_work_context = "（dry-run：跳过 arXiv 相关工作检索。）"
    else:
        log("[进度] arXiv 相关工作检索 …")
        related_work_context = fetch_arxiv_related_context(
            title_hint=bundle.title_hint,
            paper_text=bundle.text,
            source_url=args.source,
            max_results=int(run_cfg.get("arxiv_related_max_results", 8)),
            max_chars=int(run_cfg.get("arxiv_related_max_chars", 14000)),
        )
        if verbose_progress:
            log(f"[进度] 相关工作片段就绪（约 {len(related_work_context)} 字）。")
        else:
            log(f"[进度] 相关工作 {_chars_hint(len(related_work_context))}字")

    slug = make_paper_slug(bundle.title_hint, bundle.text, bundle.fingerprint)
    session_dir = resolve_unique_session_dir(args.out_dir, slug, bundle.fingerprint)

    cp_path = session_dir / "checkpoint.json"
    cp = load_checkpoint(cp_path) if not args.fresh and not args.no_resume else None

    if args.no_resume and cp_path.is_file() and not args.fresh:
        print(
            "错误：目录中已有 checkpoint.json，但指定了 --no-resume。"
            "若要丢弃旧进度从头跑，请加 --fresh；若要续跑，请去掉 --no-resume。",
            file=sys.stderr,
        )
        sys.exit(2)

    if args.fresh and session_dir.is_dir() and any(session_dir.iterdir()):
        arch = archive_session_dir(session_dir)
        log(f"[进度] --fresh：已将会话内容归档到 {arch}")
        cp = None

    resume_state: RunState | None = None
    phase_resume: str | None = None
    memory_log: list[str] | None = None
    resume_macro_attempt = 1
    resume_inner_start_round = 1

    if cp and not args.no_resume:
        if cp.get("source_normalized") != bundle.source_normalized:
            print(
                f"错误：checkpoint 来源与当前输入不一致（{cp.get('source_normalized')} vs {bundle.source_normalized}）。"
                f"请使用 --fresh 或更换目录。",
                file=sys.stderr,
            )
            sys.exit(2)
        if cp.get("paper_fingerprint") != bundle.fingerprint and not args.ignore_fingerprint:
            print(
                "错误：同一目录下 checkpoint 的论文指纹与当前文件不一致（论文可能已更新）。"
                "请使用 --fresh 重新跑，或加 --ignore-fingerprint 强制续跑。",
                file=sys.stderr,
            )
            sys.exit(2)

        status = cp.get("status")
        phase = cp.get("phase", "debating")
        n_done = len(cp.get("rounds") or [])
        memory_log = list(cp.get("abandoned_memories") or [])
        resume_macro_attempt = int(cp.get("current_macro_attempt", 1))
        if cp.get("inner_next_round") is not None:
            resume_inner_start_round = int(cp["inner_next_round"])
        else:
            rs_same = [
                x
                for x in (cp.get("rounds") or [])
                if int(x.get("macro_attempt", 1)) == resume_macro_attempt
            ]
            resume_inner_start_round = len(rs_same) + 1

        if status == "completed" and phase == "done":
            print(
                f"该论文流水线已完整跑过，结果在目录：\n  {session_dir}\n"
                f"主要文件：checkpoint.json, final.md, final.json\n"
                f"若要同一篇论文重新生成 idea，请使用：./run.sh ... --fresh",
                file=sys.stderr,
            )
            sys.exit(0)

        if cp.get("status") == "failed" and cp.get("phase") == "abandoned_all" and not args.fresh:
            print(
                f"上次运行已判定无法依据该文给出较好 idea（见 {session_dir / 'failure_no_idea.md'}）。\n"
                f"若要重新尝试，请加 --fresh。",
                file=sys.stderr,
            )
            sys.exit(4)

        if n_done >= max_rounds and phase == "outline" and not (cp.get("outline") or "").strip():
            resume_state = state_from_checkpoint(cp)
            phase_resume = "outline"
            if verbose_progress:
                log("[进度] 从 checkpoint 恢复：仅执行大纲步骤（含 skip-outline 时的空大纲落盘）。")
            else:
                log("[进度] 续跑 → 仅大纲")
        elif n_done > 0 and phase != "done":
            resume_state = state_from_checkpoint(cp)
            if verbose_progress:
                log(
                    f"[进度] 从 checkpoint 恢复：宏观尝试 {resume_macro_attempt}，"
                    f"内层从第 {resume_inner_start_round} 轮继续。"
                )
            else:
                log(f"[进度] 续跑 M{resume_macro_attempt}·自R{resume_inner_start_round} …")
        elif n_done >= max_rounds and (cp.get("outline") or "").strip():
            print(
                f"辩论与大纲均已完成，见：\n  {session_dir}\n使用 --fresh 可重新跑。",
                file=sys.stderr,
            )
            sys.exit(0)

    state = run(
        source=args.source,
        bundle=bundle,
        base_url=args.base_url,
        api_key=args.api_key or "dummy",
        default_model=args.model,
        generator_model=args.generator_model,
        reviewer_model=args.reviewer_model,
        outline_model=args.outline_model,
        config_path=cfg_path,
        max_rounds=max_rounds,
        max_macro_attempts=max_idea_attempts,
        session_dir=session_dir,
        dry_run=args.dry_run,
        extra_notes=args.notes,
        skip_outline=args.skip_outline,
        resume_state=resume_state,
        memory_log=memory_log,
        resume_macro_attempt=resume_macro_attempt,
        resume_inner_start_round=resume_inner_start_round,
        phase_resume=phase_resume,
        meta=meta,
        paper_slug=slug,
        related_work_context=related_work_context,
        min_debate_rounds_before_outline=min_debate_rounds,
        verbose_progress=verbose_progress,
    )

    lines = [
        "已写入会话目录：",
        f"  {session_dir}",
        f"  {session_dir / 'final.json'}",
        f"  {session_dir / 'final.md'}",
        f"  {session_dir / 'final_debate.txt'}",
        f"  {session_dir / 'checkpoint.json'}",
        f"  {session_dir / 'progress.md'}",
    ]
    ofp = session_dir / OUTLINE_FINAL_FILENAME
    if ofp.is_file():
        lines.append(f"  {ofp}  （仅定稿大纲）")
    print("\n".join(lines), flush=True)


if __name__ == "__main__":
    main()
