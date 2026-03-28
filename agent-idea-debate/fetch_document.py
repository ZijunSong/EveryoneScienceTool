"""从 URL 或本地路径拉取论文/HTML/PDF 文本（尽量提取正文）。"""

from __future__ import annotations

import hashlib
import re
import urllib.parse
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urljoin, urlparse, urlunparse

import httpx
import trafilatura
from bs4 import BeautifulSoup
from pypdf import PdfReader


def _truncate(text: str, max_chars: int) -> str:
    text = text.strip()
    if len(text) <= max_chars:
        return text
    return text[: max_chars // 2] + "\n\n[... 中间省略 ...]\n\n" + text[-max_chars // 2 :]


def normalize_source(url_or_path: str) -> str:
    """用于会话匹配：URL 规范化；本地路径为 resolve 后的绝对路径。"""
    s = url_or_path.strip()
    if s.startswith("http://") or s.startswith("https://"):
        p = urlparse(s)
        path = (p.path or "").rstrip("/")
        # 统一 arxiv abs 形式（去掉尾部 /v1 等差异可再扩展）
        return urlunparse((p.scheme.lower(), (p.netloc or "").lower(), path or "/", "", "", ""))
    return str(Path(s).expanduser().resolve())


def _sha256_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8", errors="replace")).hexdigest()


@dataclass
class SourceBundle:
    """拉取结果：正文、标题线索、内容指纹、规范化来源。"""

    text: str
    title_hint: str | None
    fingerprint: str
    source_normalized: str


def _first_line_title(text: str, max_len: int = 120) -> str | None:
    for line in text.splitlines():
        line = line.strip()
        if len(line) >= 8 and len(line) <= max_len:
            return line
        if len(line) > max_len:
            return line[:max_len]
    return None


def _pdf_meta_title(data: bytes) -> str | None:
    from io import BytesIO

    try:
        reader = PdfReader(BytesIO(data))
        meta = reader.metadata
        if not meta:
            return None
        t = meta.get("/Title") or meta.get("/title") or meta.get("Title")
        if t:
            s = str(t).strip()
            if s and s.lower() not in ("untitled", "no title"):
                return s
    except Exception:
        pass
    return None


def _arxiv_title_from_html(html: str) -> str | None:
    soup = BeautifulSoup(html, "lxml")
    el = soup.select_one("h1.title") or soup.select_one(".title.mathjax")
    if el:
        t = el.get_text(" ", strip=True)
        return t or None
    og = soup.find("meta", property="og:title")
    if og and og.get("content"):
        return og["content"].strip()
    return None


def fetch_from_url(url: str, timeout: float = 120.0) -> str:
    return fetch_from_url_bundle(url, timeout).text


def fetch_from_url_bundle(url: str, timeout: float = 120.0) -> SourceBundle:
    parsed = urlparse(url)
    path_lower = (parsed.path or "").lower()
    norm = normalize_source(url)

    headers = {
        "User-Agent": "Mozilla/5.0 (compatible; AgentIdeaDebate/1.0; +https://arxiv.org)",
        "Accept": "text/html,application/pdf,*/*",
    }

    title_hint: str | None = None

    with httpx.Client(timeout=timeout, follow_redirects=True, headers=headers) as client:
        resp = client.get(url)
        resp.raise_for_status()
        content_type = (resp.headers.get("content-type") or "").lower()

        if "pdf" in content_type or path_lower.endswith(".pdf"):
            raw = _pdf_bytes_to_text(resp.content)
            title_hint = _pdf_meta_title(resp.content) or _first_line_title(raw)
            fp = _sha256_text(raw)
            return SourceBundle(text=raw, title_hint=title_hint, fingerprint=fp, source_normalized=norm)

        html = resp.text
        if "arxiv.org" in (parsed.netloc or ""):
            title_hint = _arxiv_title_from_html(html)
            text = _arxiv_html_to_text(html, url, client)
            if text:
                fp = _sha256_text(text)
                if not title_hint:
                    title_hint = _first_line_title(text)
                return SourceBundle(text=text, title_hint=title_hint, fingerprint=fp, source_normalized=norm)

        extracted = trafilatura.extract(html)
        if extracted and len(extracted.strip()) > 200:
            raw = extracted.strip()
            fp = _sha256_text(raw)
            title_hint = title_hint or _first_line_title(raw)
            return SourceBundle(text=raw, title_hint=title_hint, fingerprint=fp, source_normalized=norm)

        soup = BeautifulSoup(html, "lxml")
        for tag in soup(["script", "style", "nav", "footer"]):
            tag.decompose()
        plain = soup.get_text("\n", strip=True)
        raw = plain if plain.strip() else html[:50000]
        fp = _sha256_text(raw)
        title_hint = title_hint or _first_line_title(raw)
        return SourceBundle(text=raw, title_hint=title_hint, fingerprint=fp, source_normalized=norm)


def _arxiv_html_to_text(html: str, page_url: str, client: httpx.Client) -> str | None:
    """优先尝试 arxiv PDF 全文；失败则 HTML 摘要。"""
    m = re.search(r"href=\"([^\"]+/pdf/[^\"]+)\"", html)
    pdf_url = None
    if m:
        pdf_url = urljoin(page_url, m.group(1)).replace("http://", "https://")
    else:
        m2 = re.search(r"arxiv\.org/abs/([\d.]+)", page_url)
        if m2:
            pdf_url = f"https://arxiv.org/pdf/{m2.group(1)}.pdf"

    if pdf_url:
        try:
            r = client.get(pdf_url)
            if r.status_code == 200 and r.content[:4] == b"%PDF":
                return _pdf_bytes_to_text(r.content)
        except Exception:
            pass

    extracted = trafilatura.extract(html)
    if extracted and len(extracted.strip()) > 100:
        return extracted.strip()
    return None


def _pdf_bytes_to_text(data: bytes) -> str:
    from io import BytesIO

    reader = PdfReader(BytesIO(data))
    parts: list[str] = []
    for page in reader.pages:
        t = page.extract_text() or ""
        parts.append(t)
    return "\n\n".join(parts).strip()


def load_local_bundle(path: str | Path) -> SourceBundle:
    p = Path(path).expanduser().resolve()
    if not p.is_file():
        raise FileNotFoundError(str(p))
    norm = normalize_source(str(p))
    suf = p.suffix.lower()
    if suf == ".pdf":
        data = p.read_bytes()
        raw = _pdf_bytes_to_text(data)
        title_hint = _pdf_meta_title(data) or _first_line_title(raw)
    else:
        raw = p.read_text(encoding="utf-8", errors="replace")
        title_hint = _first_line_title(raw)
    fp = _sha256_text(raw)
    return SourceBundle(text=raw, title_hint=title_hint, fingerprint=fp, source_normalized=norm)


def load_source(url_or_path: str, max_chars: int) -> str:
    return load_source_with_meta(url_or_path, max_chars).text


_ARXIV_API = "http://export.arxiv.org/api/query"
_ATOM_NS = {"atom": "http://www.w3.org/2005/Atom"}


def extract_arxiv_id_from_url(url: str) -> str | None:
    m = re.search(r"arxiv\.org/(?:abs|pdf)/(\d{4}\.\d{4,5})(?:v\d+)?", url, re.I)
    return m.group(1) if m else None


def _keywords_for_arxiv_search(title_hint: str | None, paper_excerpt: str, max_kw: int = 6) -> list[str]:
    text = f"{title_hint or ''} {paper_excerpt[:2000]}"
    words = re.findall(r"[a-zA-Z][a-zA-Z\-]{2,}", text)
    stop = {
        "the",
        "and",
        "for",
        "with",
        "from",
        "this",
        "that",
        "paper",
        "our",
        "are",
        "was",
        "has",
        "have",
        "not",
        "can",
        "use",
        "using",
        "via",
        "all",
        "new",
        "two",
        "one",
    }
    out: list[str] = []
    for w in words:
        wl = w.lower()
        if wl in stop or len(wl) < 3:
            continue
        if wl not in out:
            out.append(wl)
        if len(out) >= max_kw:
            break
    if len(out) < 2:
        out = ["agent", "language", "model", "reasoning"]
    return out[:max_kw]


def fetch_arxiv_related_context(
    *,
    title_hint: str | None,
    paper_text: str,
    source_url: str,
    max_results: int = 8,
    max_chars: int = 14000,
    timeout: float = 45.0,
) -> str:
    """
    用 arXiv API 按标题/正文关键词检索若干篇相关工作摘要，供生成器构建 idea 时对照领域脉络。
    失败或离线时返回简短说明字符串，不抛异常。
    """
    kws = _keywords_for_arxiv_search(title_hint, paper_text)
    # OR 提高召回；与种子论文同领域关键词
    q = " OR ".join(f"all:{k}" for k in kws[:5])
    params = urllib.parse.urlencode(
        {
            "search_query": q,
            "start": "0",
            "max_results": str(max(3, max_results)),
            "sortBy": "relevance",
        }
    )
    url = f"{_ARXIV_API}?{params}"
    seed_id = extract_arxiv_id_from_url(source_url)
    try:
        with httpx.Client(timeout=timeout, follow_redirects=True) as client:
            r = client.get(url)
            r.raise_for_status()
    except Exception:
        return (
            "（未能从 arXiv 拉取相关工作：网络不可用或超时。"
            "请你在构思 idea 时主动对照该领域内近年的代表工作与你熟悉的文献脉络。）"
        )

    try:
        root = ET.fromstring(r.text)
    except ET.ParseError:
        return "（arXiv 返回解析失败；请依赖自身知识做领域调研。）"

    lines: list[str] = [
        "【以下为根据种子论文标题/摘要关键词从 arXiv 检索到的若干篇「可能相关」工作（标题+摘要节选，供你对照领域格局；非穷尽）】\n"
    ]
    n = 0
    for entry in root.findall("atom:entry", _ATOM_NS):
        if n >= max_results:
            break
        tit_el = entry.find("atom:title", _ATOM_NS)
        summ_el = entry.find("atom:summary", _ATOM_NS)
        id_el = entry.find("atom:id", _ATOM_NS)
        tit = (tit_el.text or "").strip().replace("\n", " ") if tit_el is not None else ""
        summ = (summ_el.text or "").strip().replace("\n", " ") if summ_el is not None else ""
        aid = (id_el.text or "").strip() if id_el is not None else ""
        if seed_id and seed_id in aid:
            continue
        summ_short = summ[:1200] + ("…" if len(summ) > 1200 else "")
        n += 1
        lines.append(f"\n--- 相关 {n} ---\n标题：{tit}\n链接：{aid}\n摘要节选：{summ_short}\n")

    if n == 0:
        return (
            "（arXiv 未返回可展示的条目；请你在构思时主动检索/回忆该子领域近年的顶会论文与典型 setting。）"
        )

    blob = "\n".join(lines)
    if len(blob) > max_chars:
        blob = blob[: max_chars // 2] + "\n\n[... 中间省略 ...]\n\n" + blob[-max_chars // 2 :]
    return blob


def load_source_with_meta(url_or_path: str, max_chars: int) -> SourceBundle:
    s = url_or_path.strip()
    if s.startswith("http://") or s.startswith("https://"):
        b = fetch_from_url_bundle(s)
    else:
        b = load_local_bundle(s)
    truncated = _truncate(b.text, max_chars)
    # 指纹：截断前全文（与旧行为一致，便于同 URL 再跑时匹配）
    return SourceBundle(
        text=truncated,
        title_hint=b.title_hint,
        fingerprint=b.fingerprint,
        source_normalized=b.source_normalized,
    )
