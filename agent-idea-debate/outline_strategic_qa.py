#!/usr/bin/env python3
"""
用高推理档位模型（如 gpt-5.4 xhigh，以网关模型名为准）阅读仓库内**全部**已定稿论文大纲，
回答「两篇工作如何选、如何排期到 AAAI / ICLR 截稿」类战略问题。

示例：
  export OPENAI_API_KEY=...
  cd /data/ppnm/agent-idea-debate
  python3 outline_strategic_qa.py -o outputs/two_papers_plan.md
  python3 outline_strategic_qa.py --model gpt-5.4 -o plan.md
  python3 outline_strategic_qa.py --dry-run   # 仅列出将纳入的大纲文件，不调 API
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

from run_pipeline import OUTLINE_FINAL_FILENAME, OpenAI, call_chat, normalize_openai_base_url

if OpenAI is None:
    raise RuntimeError("请安装 openai: pip install -r requirements.txt")

# 默认问题（可按需改 --question-file 或 -q）
DEFAULT_QUESTION = """## 时间与会议（假设）

- **今天**：2026 年 5 月 1 日  
- **AAAI 截稿**：2026 年 8 月 1 日  
- **ICLR 截稿**：2026 年 10 月 1 日  

## 请你回答

我是一名研究者，希望**连续展开两篇**独立但可形成叙事承接的工作。下面附上本仓库目前已生成的**全部**论文大纲（每份对应不同 idea）。

请用**中文**回答：

1. **应选取哪两个方向开展工作？**  
   请明确对应下面哪两份大纲（用「来源标签」指代），并说明：与两次截稿的匹配、两篇之间的依赖/承接、主要风险与缓解思路。

2. **我应如何做到这一点？**  
   给出从 **2026-05-01** 起至 **AAAI 与 ICLR 两次截稿** 的**可执行时间线**（里程碑、何时锁题、实验/写作并行建议、缓冲）。

若现有大纲信息不足以判断，请**明确写出假设**，仍给出一版可执行的默认排期。"""


def discover_outline_paths(root: Path) -> list[Path]:
    """递归查找 outline_final.md；按路径排序保证稳定。"""
    if not root.is_dir():
        return []
    paths = sorted(root.rglob(OUTLINE_FINAL_FILENAME))
    return [p for p in paths if p.is_file()]


def discover_fallback_json(root: Path) -> list[Path]:
    """无 outline_final 时，用含 outline 的 final.json。"""
    out: list[Path] = []
    for fj in sorted(root.rglob("final.json")):
        if (fj.parent / OUTLINE_FINAL_FILENAME).is_file():
            continue
        try:
            data = json.loads(fj.read_text(encoding="utf-8"))
        except Exception:
            continue
        o = (data.get("outline") or "").strip()
        if len(o) < 200:
            continue
        out.append(fj)
    return out


def load_outline_bundle(path: Path) -> tuple[str, str]:
    """
    返回 (来源标签, 全文或拼接文本)，供模型阅读。
    path 可为 outline_final.md 或 final.json。
    """
    if path.name == OUTLINE_FINAL_FILENAME:
        label = path.parent.name
        text = path.read_text(encoding="utf-8", errors="replace")
        return label, text.strip()

    if path.name == "final.json":
        label = path.parent.name
        data = json.loads(path.read_text(encoding="utf-8"))
        ta = (data.get("final_title_abstract") or "").strip()
        o = (data.get("outline") or "").strip()
        body = ""
        if ta:
            body += "## Idea（标题与摘要）\n\n" + ta + "\n\n"
        body += "## 大纲（来自 final.json）\n\n" + o
        return label, body.strip()

    raise ValueError(f"不支持的文件：{path}")


def build_user_message(
    question: str,
    bundles: list[tuple[str, str]],
    *,
    max_chars_per_doc: int,
) -> str:
    parts: list[str] = [question.strip(), "\n\n---\n\n## 仓库内全部大纲（按路径顺序）\n"]
    for i, (label, text) in enumerate(bundles, start=1):
        rel = text
        truncated = False
        if len(rel) > max_chars_per_doc:
            rel = rel[: max_chars_per_doc - 100] + "\n\n…（该份已截断）…\n"
            truncated = True
        parts.append(f"\n### 来源标签 {i}：`{label}`\n")
        if truncated:
            parts.append("（正文过长已截断，请优先依据未截断部分推理。）\n\n")
        parts.append(rel)
        parts.append("\n\n---\n")
    return "".join(parts)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="用模型阅读仓库内全部大纲并回答战略问题（截稿、两篇工作选线与排期）"
    )
    ap.add_argument(
        "--root",
        type=Path,
        default=None,
        help="搜索大纲的根目录（默认：本脚本所在目录下的 outputs）",
    )
    ap.add_argument(
        "--include-final-json",
        action="store_true",
        help="同时纳入无 outline_final.md 但 final.json 中含较长 outline 的会话",
    )
    ap.add_argument(
        "--model",
        default=os.environ.get("OPENAI_ADVISOR_MODEL", "gpt-5.4"),
        help="模型名（默认 gpt-5.4；xhigh 等由网关侧配置或换用对应模型名）",
    )
    ap.add_argument("--base-url", default=os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1"))
    ap.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY", ""))
    ap.add_argument("--temperature", type=float, default=0.25)
    ap.add_argument("--max-tokens", type=int, default=16384)
    ap.add_argument("--timeout", type=float, default=900.0)
    ap.add_argument(
        "--max-chars-per-doc",
        type=int,
        default=56_000,
        help="单份大纲纳入上下文的最大字符（避免超长）",
    )
    ap.add_argument("-q", "--question", default=None, help="自定义问题全文（覆盖内置默认）")
    ap.add_argument("--question-file", type=Path, default=None, help="从文件读取问题（UTF-8）")
    ap.add_argument("-o", "--out", type=Path, default=None, help="写入回答 Markdown（默认打印到 stdout）")
    ap.add_argument("--dry-run", action="store_true", help="只列出将纳入的文件，不调用 API")
    args = ap.parse_args()

    script_dir = Path(__file__).resolve().parent
    root = args.root if args.root is not None else script_dir / "outputs"
    root = root.resolve()

    outline_paths = discover_outline_paths(root)
    if args.include_final_json:
        outline_paths = outline_paths + discover_fallback_json(root)

    if not outline_paths:
        print(f"错误：在 {root} 下未找到任何 {OUTLINE_FINAL_FILENAME}", file=sys.stderr)
        if not args.include_final_json:
            print("提示：可尝试加 --include-final-json", file=sys.stderr)
        sys.exit(2)

    bundles: list[tuple[str, str]] = []
    for p in outline_paths:
        try:
            bundles.append(load_outline_bundle(p))
        except Exception as e:
            print(f"[警告] 跳过 {p}: {e}", file=sys.stderr)

    if not bundles:
        print("错误：没有可读入的大纲内容。", file=sys.stderr)
        sys.exit(2)

    print(f"[outline_strategic_qa] 根目录: {root}", file=sys.stderr)
    print(f"[outline_strategic_qa] 纳入 {len(bundles)} 份大纲", file=sys.stderr)
    for p in outline_paths[:20]:
        try:
            disp = p.relative_to(root)
        except ValueError:
            disp = p
        print(f"  · {disp}", file=sys.stderr)
    if len(outline_paths) > 20:
        print(f"  · … 共 {len(outline_paths)} 个文件", file=sys.stderr)

    if args.dry_run:
        sys.exit(0)

    if not args.api_key:
        print("错误：请设置 --api-key 或 OPENAI_API_KEY", file=sys.stderr)
        sys.exit(1)

    question = DEFAULT_QUESTION
    if args.question_file is not None:
        question = args.question_file.read_text(encoding="utf-8")
    elif args.question is not None:
        question = args.question

    user_content = build_user_message(question, bundles, max_chars_per_doc=args.max_chars_per_doc)

    client = OpenAI(base_url=normalize_openai_base_url(args.base_url), api_key=args.api_key)
    print(f"[outline_strategic_qa] 调用 {args.model!r} …", file=sys.stderr, flush=True)

    answer = call_chat(
        client,
        args.model,
        [
            {
                "role": "system",
                "content": "你是资深机器学习研究导师，熟悉顶会投稿节奏与项目排期。请严格用中文回答，结构清晰，可执行。",
            },
            {"role": "user", "content": user_content},
        ],
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        timeout=args.timeout,
    )

    header = (
        f"# 两篇工作与截稿排期（战略问答）\n\n"
        f"- **生成时间**：{datetime.now().isoformat()}\n"
        f"- **模型**：`{args.model}`\n"
        f"- **大纲份数**：{len(bundles)}\n"
        f"- **搜索根目录**：`{root}`\n\n"
        f"---\n\n"
    )
    full = header + answer.strip() + "\n"

    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(full, encoding="utf-8")
        print(f"已写入：{args.out.resolve()}", flush=True)
    else:
        print(full)


if __name__ == "__main__":
    main()
