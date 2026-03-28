#!/usr/bin/env python3
"""
手动：对多篇「论文大纲」按「最值得投入」排序（默认调用高推理档位模型，如 gpt-5.4 xhigh，具体以网关可用名为准）。

示例：
  export OPENAI_API_KEY=...
  python3 rank_outlines.py \\
    outputs/paper_a/ outputs/paper_b/ \\
    --model gpt-5.4 \\
    -o outputs/outline_ranking.md

也可直接传 outline_final.md 或 final.json 路径。
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from datetime import datetime
from pathlib import Path

from run_pipeline import (
    OUTLINE_FINAL_FILENAME,
    OpenAI,
    call_chat,
    normalize_openai_base_url,
)

if OpenAI is None:
    raise RuntimeError("请安装 openai: pip install -r requirements.txt")


def _truncate(s: str, max_chars: int) -> tuple[str, bool]:
    s = s.strip()
    if len(s) <= max_chars:
        return s, False
    return s[: max_chars - 80] + "\n\n…（以下已截断，排序时请仅依据上文）…\n", True


def _extract_idea_from_outline_final(text: str) -> str:
    m = re.search(r"##\s*Idea（标题与摘要）\s*\n(.+?)(?=\n---|\n##\s)", text, re.DOTALL)
    if m:
        return m.group(1).strip()
    return ""


def _extract_outline_body_from_outline_final(text: str) -> str:
    m = re.search(r"##\s*大纲正文\s*\n(.*)", text, re.DOTALL)
    if m:
        return m.group(1).strip()
    return text.strip()


def _extract_idea_from_final_md(text: str) -> str:
    m = re.search(r"##\s*最终用于大纲的标题与摘要\s*\n(.+?)(?=\n##\s)", text, re.DOTALL)
    if m:
        return m.group(1).strip()
    return ""


def extract_outline_from_final_md(text: str) -> str | None:
    for pat in (
        r"##\s*论文大纲（修订终稿）\s*\n(.+?)(?=\n##\s|\Z)",
        r"##\s*论文大纲（末步模型）\s*\n(.+?)(?=\n##\s|\Z)",
    ):
        m = re.search(pat, text, re.DOTALL)
        if m:
            return m.group(1).strip()
    return None


def load_item(path: Path) -> tuple[str, str, str]:
    """
    返回 (label, idea_hint, outline_body)。
    label 用于在报告中标识来源；idea_hint 可为空。
    """
    if not path.exists():
        raise FileNotFoundError(path)

    if path.is_file():
        return _load_from_file(path)

    if path.is_dir():
        of = path / OUTLINE_FINAL_FILENAME
        if of.is_file():
            t = of.read_text(encoding="utf-8", errors="replace")
            return path.name, _extract_idea_from_outline_final(t), _extract_outline_body_from_outline_final(t)
        fj = path / "final.json"
        if fj.is_file():
            data = json.loads(fj.read_text(encoding="utf-8"))
            o = (data.get("outline") or "").strip()
            ta = (data.get("final_title_abstract") or "").strip()
            if o:
                return path.name, ta, o
        fm = path / "final.md"
        if fm.is_file():
            tx = fm.read_text(encoding="utf-8", errors="replace")
            body = extract_outline_from_final_md(tx)
            if body:
                return path.name, _extract_idea_from_final_md(tx), body
        raise ValueError(
            f"目录中未找到 {OUTLINE_FINAL_FILENAME}、含 outline 的 final.json，"
            f"或可解析的 final.md：{path}"
        )

    raise ValueError(f"unsupported path: {path}")


def _load_from_file(path: Path) -> tuple[str, str, str]:
    text = path.read_text(encoding="utf-8", errors="replace")
    if path.suffix.lower() == ".json":
        data = json.loads(text)
        o = (data.get("outline") or "").strip()
        ta = (data.get("final_title_abstract") or "").strip()
        if not o:
            raise ValueError(f"{path} 无 outline 字段")
        return str(path), ta, o

    if path.name == OUTLINE_FINAL_FILENAME or "论文大纲（定稿）" in text[:600]:
        return str(path), _extract_idea_from_outline_final(text), _extract_outline_body_from_outline_final(text)

    body = extract_outline_from_final_md(text)
    if body:
        return str(path), _extract_idea_from_final_md(text), body

    return str(path), "", text  # 整份当大纲


def build_rank_prompt(
    items: list[tuple[str, str, str]],
    *,
    max_chars_per_outline: int,
) -> str:
    blocks: list[str] = []
    for i, (label, idea, outline) in enumerate(items, start=1):
        o, truncated = _truncate(outline, max_chars_per_outline)
        head = f"### 编号{i}\n\n**来源标签**：`{label}`\n"
        if idea:
            head += f"\n**Idea 摘要**：\n{idea}\n\n"
        if truncated:
            head += "\n（大纲正文已截断）\n"
        head += f"\n**大纲正文**：\n\n{o}\n"
        blocks.append(head)
    bodies = "\n\n---\n\n".join(blocks)
    n = len(items)
    return f"""你是一位资深机器学习研究导师，熟悉顶会（NeurIPS/ICLR/ICML）论文选题与落地可行性。

下面给出 **{n}** 份**论文大纲**（来自不同种子论文或 idea），每份已编号为 `编号1` … `编号{n}`。

你的任务**只**是：按「**若只能选一个方向长期投入，最值得优先做的**」程度，给出**从强到弱**的排序（**强 = 更值得做**）。

**排序时请综合考虑**（不必逐条写公式，但在理由中体现）：
- 新颖性 / 与原工作的差异与不可替代性；
- 问题的重要性与可发表潜力（oral 潜力）；
- 在有限算力与周期内**可完成性**；
- 实验设计是否闭环、是否可证伪。

**输出格式（严格使用 Markdown，中文）**：
1. `# 大纲排序总览`：用有序列表写出编号顺序，例如 `1. 编号3 … 2. 编号1 …`（从最值得做到相对靠后）。
2. `# 各编号简要理由`：对每个编号写 2–4 句**为什么**排在这个位置。
3. `# 风险与建议`：整体 1 段话：若选前两名，各有什么主要风险；若选第三名及以后，通常缺什么。

**禁止**输出与排序无关的寒暄；**禁止**改变编号与大纲内容的对应关系（编号仅指本消息中的 `编号k`）。

---

{bodies}
"""


def main() -> None:
    ap = argparse.ArgumentParser(description="对多篇论文大纲按「最值得做」排序（OpenAI 兼容 API）")
    ap.add_argument(
        "paths",
        nargs="+",
        type=Path,
        help=f"会话目录（含 {OUTLINE_FINAL_FILENAME} / final.json）或 .md / .json 文件",
    )
    ap.add_argument(
        "--model",
        default=os.environ.get("OPENAI_RANK_MODEL", "gpt-5.4"),
        help="排序所用模型（默认 gpt-5.4；若网关提供 xhigh 档位请在网关侧配置或换用对应模型名）",
    )
    ap.add_argument("--base-url", default=os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1"))
    ap.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY", ""))
    ap.add_argument("--temperature", type=float, default=0.35)
    ap.add_argument("--max-tokens", type=int, default=8192)
    ap.add_argument("--timeout", type=float, default=600.0)
    ap.add_argument(
        "--max-chars-per-outline",
        type=int,
        default=48_000,
        help="单份大纲最大字符（超长会截断并注明）",
    )
    ap.add_argument(
        "-o",
        "--out",
        type=Path,
        default=None,
        help="写入排序报告 Markdown（默认：当前目录 outline_ranking_<时间>.md）",
    )
    args = ap.parse_args()

    if not args.api_key:
        print("错误：请设置 --api-key 或环境变量 OPENAI_API_KEY", file=sys.stderr)
        sys.exit(1)

    items: list[tuple[str, str, str]] = []
    for p in args.paths:
        try:
            items.append(load_item(p.resolve()))
        except Exception as e:
            print(f"[错误] 跳过 {p}: {e}", file=sys.stderr)
            sys.exit(2)

    if len(items) < 2:
        print("错误：至少需要 2 个有效大纲来源。", file=sys.stderr)
        sys.exit(2)

    user_prompt = build_rank_prompt(items, max_chars_per_outline=args.max_chars_per_outline)

    client = OpenAI(base_url=normalize_openai_base_url(args.base_url), api_key=args.api_key)
    print(f"[rank_outlines] 调用模型 {args.model!r} 排序 {len(items)} 份大纲…", file=sys.stderr, flush=True)
    report = call_chat(
        client,
        args.model,
        [
            {
                "role": "system",
                "content": "你是资深研究导师，按要求用中文输出 Markdown 排序报告，不输出无关寒暄。",
            },
            {"role": "user", "content": user_prompt},
        ],
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        timeout=args.timeout,
    )

    out_path = args.out
    if out_path is None:
        out_path = Path.cwd() / f"outline_ranking_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.md"

    header = (
        f"# 论文大纲排序报告\n\n"
        f"- **生成时间**：{datetime.now().isoformat()}\n"
        f"- **模型**：`{args.model}`\n"
        f"- **来源**：{len(items)} 个路径\n\n"
        f"---\n\n"
    )
    full = header + report.strip() + "\n"
    out_path.write_text(full, encoding="utf-8")
    print(f"\n已写入：{out_path.resolve()}", flush=True)


if __name__ == "__main__":
    main()
