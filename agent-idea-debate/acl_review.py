"""
ACL 风格 5 分制三审稿人：解析分数、拼接审稿正文、大纲审稿提示。
"""

from __future__ import annotations

import re
from typing import Any

ACL_IDEA_USER_SUFFIX = """

---

【审稿模式：ACL 5 分制（与顶会审稿口径对齐）】
请像 ACL 领域审稿人一样评价上述 idea（仅针对当前生成器输出）。输出必须全部使用中文，并**严格**遵守下列格式（便于程序解析）。

**评分口径（务必自洽；分数与「一句话结论」不得矛盾）：**
- **1–2 分**：工作存在明显问题或贡献不足，整体偏弱。
- **2.5 分（可写 2.5）**：borderline findings，偏弱，需很大力度修订。
- **3 分**：**findings 档可争**，核心贡献可讲清，仍需澄清与补实验，但**不是**「拒稿级重大修改」。
- **3.5 分（可写 3.5）**：接近主会 borderline，亮点更足。
- **4 分**：扎实、主会水平可争。
- **5 分**：非常强。

**硬性规则（违反则视为不合格输出）：**
1. 第一行且仅含分数，格式：`ACL总分：N/5`，N 可为 **1、1.5、2、2.5、3、3.5、4、4.5、5**（半分可选）。
2. **若 ACL总分 ≥3**：**禁止**在「一句话结论」中使用 **「建议放弃」** 或 **「需重大修改」**（二者仅允许在总分 <3 时出现）。
3. 若总分 **<3**：必须写 **「改进方向」**（2–5 条可执行建议）；若总分 **≥3 且 <4**：可写「主要待澄清/待补实验」；若 **≥4**：可写「可选微调」或「无」。
4. 最后一行且仅一行：`一句话结论：` 后接**与分数匹配**的结论，**仅允许**下列之一（请按分数选用）：
   - 总分 **<2.5** → `建议放弃`
   - 总分 **≥2.5 且 <3** → `borderline findings（需强力修订）`
   - 总分 **≥3 且 <3.5** → `findings 档可接收（需小幅修改与澄清）`
   - 总分 **≥3.5 且 <4** → `borderline main（需补强后冲主会）`
   - 总分 **≥4 且 <5** → `主会水平可争（扎实）`
   - 总分 **=5** → `非常强（主会 strong）`
"""

ACL_OUTLINE_USER_SUFFIX = """

---

【审稿模式：ACL 5 分制 · 审大纲】
请像 ACL 领域审稿人一样评价上述**论文大纲**。输出必须全部使用中文，并**严格**遵守下列格式。

**评分口径**：与 idea 审稿相同（2.5≈borderline findings；**3≈findings 可争**；3.5≈borderline main；4≈主会扎实）。

**硬性规则：**
1. 第一行：`ACL总分：N/5`（N 可为半分，同上）。
2. **若 ACL总分 ≥3**：**禁止**在「一句话结论」中使用 **「建议放弃」** 或 **「需重大修改」**。
3. 若总分 **<3**：必须写 **「改进方向」**；若 **≥3**：不得把大纲说成「必须推倒重来」，应写可执行的澄清与补强。
4. 最后一行：`一句话结论：` 及结论短语**必须与 idea 审稿相同的六档选项**（按分数选用，同上表）。
"""


def parse_acl_total_score(text: str) -> float:
    """解析 ACL 总分，支持半分；失败时返回 3.0。"""
    t = text.strip()
    if not t:
        return 3.0
    m = re.search(r"ACL\s*总分[:：]\s*(\d+(?:\.\d+)?)\s*/\s*5", t, re.I)
    if m:
        v = float(m.group(1))
        return min(5.0, max(1.0, v))
    m = re.search(r"ACL\s*总分[:：]\s*(\d+(?:\.\d+)?)", t, re.I)
    if m:
        v = float(m.group(1))
        return min(5.0, max(1.0, v))
    m = re.search(r'"overall"\s*:\s*(\d+(?:\.\d+)?)', t)
    if m:
        v = float(m.group(1))
        return min(5.0, max(1.0, v))
    m = re.search(r"总分[:：]\s*(\d+(?:\.\d+)?)\s*/\s*5", t)
    if m:
        v = float(m.group(1))
        return min(5.0, max(1.0, v))
    return 3.0


def acl_scores_pass(scores: list[float], threshold: float) -> bool:
    if not scores:
        return False
    return min(scores) >= threshold - 1e-9


def acl_verdict_line_mismatch(score: float, text: str) -> bool:
    """分数≥3 但「一句话结论」行仍写拒稿级用语 → True（便于打日志，不自动改分）。"""
    if score < 3.0:
        return False
    for ln in reversed([x.strip() for x in text.splitlines() if x.strip()]):
        if "一句话结论" in ln:
            return bool(re.search(r"需重大修改|建议放弃", ln))
    return False


def combine_acl_reviews(
    role_label: str,
    items: list[dict[str, Any]],
) -> str:
    """items: [{"model": str, "score": int, "text": str}, ...]"""
    parts: list[str] = []
    for i, it in enumerate(items, 1):
        model = it.get("model", f"审稿人{i}")
        score = it.get("score", "?")
        body = (it.get("text") or "").strip()
        parts.append(f"## {role_label} {i}（模型 `{model}`）· ACL总分：**{score}/5**\n\n{body}")
    parts.append(
        "\n---\n\n**汇总**：三审稿人平均分 "
        f"{sum(x.get('score', 0) for x in items) / max(len(items), 1):.2f}"
        f"/5；最低分 {min((x.get('score') or 0) for x in items)}/5。"
    )
    return "\n\n".join(parts)


def outline_revision_user(
    final_title_abstract: str,
    outline_candidate: str,
    combined_acl_feedback: str,
    paper_text: str,
) -> str:
    return f"""你是一位资深研究员。请根据三位 ACL 风格审稿人对「大纲」的意见，将大纲修订为一份可直接据此写作的终稿（Markdown 标题与层级清晰），全部使用中文。

【已定稿 idea】
{final_title_abstract}

【当前大纲版本】
{outline_candidate}

【三位审稿人意见（含 ACL 分数与改进方向）】
{combined_acl_feedback}

【种子论文摘录（节选）】
{paper_text[:12000]}
"""