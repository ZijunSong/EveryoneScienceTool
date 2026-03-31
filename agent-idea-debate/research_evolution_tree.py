#!/usr/bin/env python3
"""
从 outputs 下多篇已定稿大纲出发，调用 LLM 推断**研究方向演化关系**（树或森林），
输出 JSON + Mermaid，便于画成「演化树」而非一堆互不相连的叶子。

示例：
  export OPENAI_API_KEY=...
  cd agent-idea-debate
  python3 research_evolution_tree.py -o outputs/research_evolution_tree.json
  python3 research_evolution_tree.py -o outputs/tree.json --md outputs/tree.md
  python3 research_evolution_tree.py --dry-run
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

from run_pipeline import OpenAI, call_chat, normalize_openai_base_url

if OpenAI is None:
    raise RuntimeError("请安装 openai: pip install -r requirements.txt")

from outline_strategic_qa import (
    discover_fallback_json,
    discover_outline_paths,
    load_outline_bundle,
)

SYSTEM_PROMPT = """你是计算/机器学习方向的研究史与课题结构分析助手。你的任务是根据多份「独立产生的研究大纲或 idea 摘要」，推断它们之间的**演化关系**，形成一棵或多棵**有根树**（森林），而不是把所有题目都画成指向同一个空洞目标的叶子。

必须遵守：
1. **禁止**使用「所有方向都指向同一个泛化终点」（例如单独一个「未来研究趋势」汇点）作为唯一结构；若必须表达趋势，应作为**若干条边的文字说明**，而不是图中唯一的父节点。
2. 节点应体现：**谁更基础/更早/更宽**，谁**特化/改进/回应了谁的局限**；边要有具体关系类型（见输出格式）。
3. 每个输入来源（用 source_tag 标识）原则上对应**至少一个**节点；若两条材料明显是同一脉络的重复表述，可合并为一个节点并 `merged_from` 列出多个 source_tag。
4. 若仅靠现有材料无法连边，宁可保持**多棵小树**（森林），也不要强行全部连到虚构的总根；允许最多增加 **3 个**「概念性内部节点」（如更粗的子领域名），且必须在 `synthetic` 字段标为 true，并简短说明理由。
5. 输出必须是**单一 JSON 对象**（不要 Markdown 围栏外的解释文字）；若需解释，放在 JSON 的 `notes` 字符串字段中。

输出 JSON Schema（字段必须齐全，无则 null 或 []）：
{
  "version": 1,
  "notes": "可选：整体说明、不确定性",
  "nodes": [
    {
      "id": "稳定英文短 id，如 n1",
      "title": "短标题（中文可）",
      "source_tags": ["与输入一致的目录/标签名，或合并时多个"],
      "summary": "一两句，概括该节点在研究脉络中的位置",
      "synthetic": false,
      "merged_from": null
    }
  ],
  "edges": [
    {
      "from_id": "父/前驱侧",
      "to_id": "子/后继侧",
      "relation": "下列之一：broadens | specializes | extends | addresses_limitation | empirical_followup | orthogonal_branch | same_theme_refinement",
      "rationale": "一句中文依据"
    }
  ],
  "roots": ["作为树根的 id 列表，入边为零的节点"]
}

关系类型说明：broadens=泛化扩展；specializes=特化；extends=在同一路线上延伸；addresses_limitation=针对前序工作局限；empirical_followup=实证跟进；orthogonal_branch=相关但分叉；same_theme_refinement=同主题打磨。"""


def build_user_payload(bundles: list[tuple[str, str]], max_chars_per_doc: int) -> str:
    parts: list[str] = [
        "下面是多份独立会话的研究材料（每份有「来源标签」与正文）。请推断演化树/森林，按要求只输出 JSON。\n\n",
    ]
    for i, (label, text) in enumerate(bundles, start=1):
        body = text.strip()
        if len(body) > max_chars_per_doc:
            body = body[: max_chars_per_doc - 80] + "\n\n…（已截断）…\n"
        parts.append(f"### 来源标签 {i}：`{label}`\n\n{body}\n\n---\n\n")
    return "".join(parts)


def extract_json_object(raw: str) -> dict[str, Any]:
    raw = raw.strip()
    m = re.search(r"```(?:json)?\s*([\s\S]*?)```", raw)
    if m:
        raw = m.group(1).strip()
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        # 尝试找第一个 { 到最后一个 }
        start = raw.find("{")
        end = raw.rfind("}")
        if start >= 0 and end > start:
            data = json.loads(raw[start : end + 1])
        else:
            raise
    if not isinstance(data, dict):
        raise ValueError("根对象不是 JSON object")
    return data


def _mermaid_escape(s: str) -> str:
    return s.replace('"', "#quot;").replace("[", "(").replace("]", ")").replace("\n", " ")


def json_to_mermaid(data: dict[str, Any]) -> str:
    nodes = data.get("nodes") or []
    edges = data.get("edges") or []
    roots = data.get("roots") or []
    lines: list[str] = [
        "```mermaid",
        "flowchart TB",
    ]
    id_set = {n["id"] for n in nodes if isinstance(n, dict) and "id" in n}
    for n in nodes:
        if not isinstance(n, dict):
            continue
        nid = str(n.get("id", "")).strip()
        if not nid:
            continue
        title = _mermaid_escape(str(n.get("title", nid))[:80])
        syn = " 🞄" if n.get("synthetic") else ""
        lines.append(f'  {nid}["{title}{syn}"]')
    for e in edges:
        if not isinstance(e, dict):
            continue
        a, b = e.get("from_id"), e.get("to_id")
        if not a or not b or a not in id_set or b not in id_set:
            continue
        rel = _mermaid_escape(str(e.get("relation", ""))[:40])
        lines.append(f"  {a} -->|{rel}| {b}")
    if roots:
        lines.append("  %% roots: " + ", ".join(str(r) for r in roots if r in id_set))
    lines.append("```")
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(
        description="根据 outputs 中多篇大纲，用 LLM 生成研究方向演化树（JSON + 可选 Mermaid）"
    )
    ap.add_argument(
        "--root",
        type=Path,
        default=None,
        help="搜索大纲的根目录（默认：脚本目录下 outputs）",
    )
    ap.add_argument(
        "--include-final-json",
        action="store_true",
        help="纳入无 outline_final 但 final.json 含长 outline 的会话",
    )
    ap.add_argument(
        "--model",
        default=os.environ.get("OPENAI_ADVISOR_MODEL", "gpt-5.4"),
        help="模型名（默认 gpt-5.4）",
    )
    ap.add_argument("--base-url", default=os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1"))
    ap.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY", ""))
    ap.add_argument("--temperature", type=float, default=0.2)
    ap.add_argument("--max-tokens", type=int, default=8192)
    ap.add_argument("--timeout", type=float, default=600.0)
    ap.add_argument(
        "--max-chars-per-doc",
        type=int,
        default=48_000,
        help="单份材料最大字符",
    )
    ap.add_argument(
        "-o",
        "--out-json",
        type=Path,
        default=None,
        help="写入树结构 JSON（默认：outputs/research_evolution_tree.json）",
    )
    ap.add_argument(
        "--md",
        type=Path,
        default=None,
        help="额外写入 Markdown（含 Mermaid 与说明）",
    )
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    script_dir = Path(__file__).resolve().parent
    root = args.root if args.root is not None else script_dir / "outputs"
    root = root.resolve()

    outline_paths = discover_outline_paths(root)
    if args.include_final_json:
        outline_paths = outline_paths + discover_fallback_json(root)

    if not outline_paths:
        print(f"错误：在 {root} 下未找到任何 outline_final.md", file=sys.stderr)
        if not args.include_final_json:
            print("提示：可尝试 --include-final-json", file=sys.stderr)
        return 2

    bundles: list[tuple[str, str]] = []
    for p in outline_paths:
        try:
            bundles.append(load_outline_bundle(p))
        except Exception as e:
            print(f"[警告] 跳过 {p}: {e}", file=sys.stderr)

    if len(bundles) < 2:
        print("错误：演化树至少需要 2 份可读材料。", file=sys.stderr)
        return 2

    out_json = args.out_json
    if out_json is None:
        out_json = root / "research_evolution_tree.json"

    print(f"[research_evolution_tree] 根目录: {root}", file=sys.stderr)
    print(f"[research_evolution_tree] 纳入 {len(bundles)} 份材料", file=sys.stderr)

    if args.dry_run:
        for p in outline_paths[:30]:
            try:
                disp = p.relative_to(root)
            except ValueError:
                disp = p
            print(f"  · {disp}", file=sys.stderr)
        return 0

    if not args.api_key:
        print("错误：请设置 OPENAI_API_KEY 或 --api-key", file=sys.stderr)
        return 1

    user_content = build_user_payload(bundles, args.max_chars_per_doc)
    client = OpenAI(base_url=normalize_openai_base_url(args.base_url), api_key=args.api_key)
    print(f"[research_evolution_tree] 调用 {args.model!r} …", file=sys.stderr, flush=True)

    raw = call_chat(
        client,
        args.model,
        [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ],
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        timeout=args.timeout,
    )

    try:
        data = extract_json_object(raw)
    except Exception as e:
        print(f"错误：无法解析模型返回的 JSON：{e}\n---\n{raw[:4000]}\n---", file=sys.stderr)
        return 3

    data.setdefault("version", 1)
    data["_meta"] = {
        "generated_at": datetime.now().isoformat(),
        "model": args.model,
        "source_root": str(root),
        "num_sources": len(bundles),
    }

    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(
        json.dumps(data, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"已写入 JSON：{out_json.resolve()}", flush=True)

    md_path = args.md
    if md_path is None:
        md_path = out_json.with_suffix(".md")

    mermaid = json_to_mermaid(data)
    notes_section = ""
    if data.get("notes"):
        notes_section = f"## 说明\n\n{data['notes']}\n\n---\n\n"
    md_body = (
        f"# 研究方向演化树（LLM 推断）\n\n"
        f"- **生成时间**：{data['_meta']['generated_at']}\n"
        f"- **模型**：`{data['_meta']['model']}`\n"
        f"- **材料份数**：{data['_meta']['num_sources']}\n"
        f"- **根目录**：`{data['_meta']['source_root']}`\n\n"
        f"结构由模型根据各篇摘要/大纲推断，**非客观时间线**；重要决策请人工核对。\n\n"
        f"---\n\n"
        f"{notes_section}"
        f"## 可视化（Mermaid）\n\n{mermaid}\n\n"
        f"---\n\n"
        f"## 原始 JSON 路径\n\n`{out_json.name}`\n"
    )

    md_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text(md_body, encoding="utf-8")
    print(f"已写入 Markdown：{md_path.resolve()}", flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
