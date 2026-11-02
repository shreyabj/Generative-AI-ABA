"""Text extraction for uploaded files and fetched pages.

Every extractor returns the same :class:`Extracted` shape: one cleaned text, plus the
offsets of every page and heading inside it. Offsets are what let a chunk produced much
later by the lattice report "page 7, section 'Risk factors'" without the lattice knowing
anything about PDFs.

Uploaded files are parsed as data and never executed: no macros, no embedded scripts,
no external entity resolution. The file type is decided by content (magic bytes) as well
as by extension, so a renamed binary is rejected instead of being parsed as text.
"""

from __future__ import annotations

import csv
import io
import re
import zipfile
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

SUPPORTED = {
    ".pdf": "application/pdf",
    ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    ".txt": "text/plain",
    ".md": "text/markdown",
    ".markdown": "text/markdown",
    ".csv": "text/csv",
    ".html": "text/html",
    ".htm": "text/html",
}

MAX_CSV_ROWS = 20_000


class ExtractionError(ValueError):
    """Raised for files that are invalid, unsupported or contain no extractable text."""


@dataclass
class Extracted:
    text: str
    title: str = ""
    kind: str = "text"
    pages: list[tuple[int, int, int]] = field(default_factory=list)  # (page_no, start, end)
    headings: list[tuple[str, int]] = field(default_factory=list)  # (heading, start offset)
    meta: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    @property
    def page_count(self) -> int | None:
        return len(self.pages) or None


# ===================================================================== sniffing
def sniff_kind(data: bytes, filename: str) -> str:
    """Decide the parser from content first, extension second."""
    ext = "." + filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    head = data[:8]
    if head.startswith(b"%PDF"):
        return "pdf"
    if head.startswith(b"PK\x03\x04"):
        try:
            with zipfile.ZipFile(io.BytesIO(data)) as zf:
                if "word/document.xml" in zf.namelist():
                    return "docx"
        except zipfile.BadZipFile as exc:
            raise ExtractionError("corrupt archive") from exc
        raise ExtractionError("zip archives other than .docx are not accepted")
    if ext in (".pdf", ".docx"):
        raise ExtractionError(f"file content does not match its {ext} extension")
    if b"\x00" in data[:4096]:
        raise ExtractionError("binary content is not a supported text format")
    if ext in (".md", ".markdown"):
        return "markdown"
    if ext == ".csv":
        return "csv"
    if ext in (".html", ".htm"):
        return "html"
    if ext in (".txt", ""):
        return "text"
    raise ExtractionError(f"unsupported file type '{ext}'. Supported: {', '.join(sorted(SUPPORTED))}")


def decode_text(data: bytes) -> str:
    for enc in ("utf-8-sig", "utf-8"):
        try:
            return data.decode(enc)
        except UnicodeDecodeError:
            continue
    try:
        from charset_normalizer import from_bytes

        best = from_bytes(data).best()
        if best is not None:
            return str(best)
    except Exception:  # noqa: BLE001
        pass
    return data.decode("latin-1", errors="replace")


# ===================================================================== cleaning
_CTRL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_HYPHEN_BREAK = re.compile(r"(\w)-\n(\w)")
_MULTI_BLANK = re.compile(r"\n{3,}")
_SPACES = re.compile(r"[ \t\u00a0]+")


def clean_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _CTRL.sub("", text)
    text = _HYPHEN_BREAK.sub(r"\1\2", text)
    lines = [_SPACES.sub(" ", ln).strip() for ln in text.split("\n")]
    text = "\n".join(lines)
    return _MULTI_BLANK.sub("\n\n", text).strip()


def _unwrap(text: str) -> str:
    """Join hard-wrapped lines inside a paragraph (PDF text layers wrap at column width)."""
    out: list[str] = []
    for para in text.split("\n\n"):
        lines = [ln for ln in para.split("\n") if ln]
        merged = ""
        for ln in lines:
            if not merged:
                merged = ln
            elif merged.endswith((".", ":", ";", "?", "!", "\u0964")) or _looks_like_heading(ln) \
                    or ln[:1] in "•-–*" or re.match(r"^\(?\d+[.)]\s", ln):
                merged += "\n" + ln
            else:
                merged += " " + ln
        out.append(merged)
    return "\n\n".join(p for p in out if p)


def _looks_like_heading(line: str) -> bool:
    s = line.strip()
    if not (3 <= len(s) <= 90) or s.endswith((".", ",", ";")):
        return False
    if re.match(r"^(\d+(\.\d+)*|[IVX]+\.|[A-Z]\.)\s+\S", s):
        return True
    letters = [c for c in s if c.isalpha()]
    if letters and sum(c.isupper() for c in letters) / len(letters) > 0.8 and len(letters) >= 4:
        return True
    words = s.split()
    return 1 <= len(words) <= 9 and all(w[:1].isupper() or not w[:1].isalpha() for w in words) and len(words) >= 2


def _strip_repeated_lines(pages: list[str]) -> tuple[list[str], int]:
    """Drop running headers/footers: short lines that recur on most pages."""
    if len(pages) < 4:
        return pages, 0
    counts: Counter[str] = Counter()
    for p in pages:
        lines = {re.sub(r"\d+", "#", ln.strip()) for ln in p.split("\n") if 0 < len(ln.strip()) <= 80}
        counts.update(lines)
    threshold = max(3, int(len(pages) * 0.5))
    noisy = {ln for ln, n in counts.items() if n >= threshold}
    if not noisy:
        return pages, 0
    removed = 0
    out = []
    for p in pages:
        keep = []
        for ln in p.split("\n"):
            if re.sub(r"\d+", "#", ln.strip()) in noisy:
                removed += 1
                continue
            keep.append(ln)
        out.append("\n".join(keep))
    return out, removed


def _assemble(parts: list[str]) -> tuple[str, list[tuple[int, int, int]]]:
    """Join per-page texts and record each page's span in the joined text."""
    text, pages, cursor = "", [], 0
    for i, part in enumerate(parts, start=1):
        if not part.strip():
            continue
        if text:
            text += "\n\n"
            cursor += 2
        start = cursor
        text += part
        cursor += len(part)
        pages.append((i, start, cursor))
    return text, pages


def _find_headings(text: str) -> list[tuple[str, int]]:
    out = []
    pos = 0
    for para in text.split("\n\n"):
        first = para.split("\n", 1)[0]
        if _looks_like_heading(first) and len(para) < 140:
            out.append((first.strip(), pos))
        pos += len(para) + 2
    return out


# =================================================================== extractors
def extract_pdf(data: bytes) -> Extracted:
    from pypdf import PdfReader
    from pypdf.errors import PdfReadError

    try:
        reader = PdfReader(io.BytesIO(data), strict=False)
        if reader.is_encrypted and not reader.decrypt(""):
            raise ExtractionError("PDF is password-protected")
    except PdfReadError as exc:
        raise ExtractionError(f"unreadable PDF: {exc}") from exc

    raw_pages: list[str] = []
    warnings: list[str] = []
    for page in reader.pages:
        try:
            raw_pages.append(clean_text(page.extract_text() or ""))
        except Exception as exc:  # noqa: BLE001 - one bad page must not lose the document
            raw_pages.append("")
            warnings.append(f"page {len(raw_pages)} could not be read: {type(exc).__name__}")
    raw_pages, removed = _strip_repeated_lines(raw_pages)
    if removed:
        warnings.append(f"removed {removed} repeated header/footer lines")
    pages_txt = [_unwrap(clean_text(p)) for p in raw_pages]
    text, pages = _assemble(pages_txt)
    if len(text.strip()) < 40:
        raise ExtractionError(
            "no extractable text - the PDF is probably scanned images (OCR is not supported)"
        )
    empty = sum(1 for p in pages_txt if not p.strip())
    if empty:
        warnings.append(f"{empty} of {len(pages_txt)} pages had no text layer")

    info: dict[str, Any] = {}
    try:
        md = reader.metadata or {}
        info = {k.lstrip("/").lower(): str(v) for k, v in md.items() if v and len(str(v)) < 300}
    except Exception:  # noqa: BLE001
        pass
    title = info.get("title", "").strip()
    if not title or title.lower() in {"untitled", "microsoft word", "powerpoint presentation"}:
        title = next((ln for ln in text.split("\n") if 6 <= len(ln) <= 120), "")
    published = _pdf_date(info.get("creationdate") or info.get("moddate") or "")
    return Extracted(
        text=text, title=title, kind="pdf", pages=pages, headings=_find_headings(text),
        meta={"pdf_info": {k: info[k] for k in ("author", "producer", "creator") if k in info},
              "published_at": published, "page_total": len(reader.pages)},
        warnings=warnings,
    )


def _pdf_date(raw: str) -> str | None:
    m = re.search(r"(\d{4})(\d{2})(\d{2})", raw)
    return f"{m.group(1)}-{m.group(2)}-{m.group(3)}" if m else None


def extract_docx(data: bytes) -> Extracted:
    import docx

    try:
        d = docx.Document(io.BytesIO(data))
    except Exception as exc:  # noqa: BLE001
        raise ExtractionError(f"unreadable DOCX: {type(exc).__name__}") from exc
    parts: list[str] = []
    headings: list[tuple[str, int]] = []
    cursor = 0

    def add(block: str, heading: bool = False) -> None:
        nonlocal cursor
        block = clean_text(block)
        if not block:
            return
        if parts:
            cursor += 2
        if heading:
            headings.append((block, cursor))
        parts.append(block)
        cursor += len(block)

    body = d.element.body
    from docx.table import Table
    from docx.text.paragraph import Paragraph

    for child in body.iterchildren():
        tag = child.tag.rsplit("}", 1)[-1]
        if tag == "p":
            p = Paragraph(child, d)
            style = (p.style.name if p.style is not None else "") or ""
            add(p.text, heading=style.lower().startswith(("heading", "title")))
        elif tag == "tbl":
            t = Table(child, d)
            rows = []
            for r in t.rows:
                cells = [clean_text(c.text).replace("\n", " ") for c in r.cells]
                rows.append(" | ".join(cells))
            add("\n".join(rows))
    text = "\n\n".join(parts)
    if len(text.strip()) < 20:
        raise ExtractionError("document contains no text")
    cp = d.core_properties
    title = (cp.title or "").strip() or (headings[0][0] if headings else text.split("\n", 1)[0][:120])
    published = cp.created.date().isoformat() if cp.created else None
    return Extracted(text=text, title=title, kind="docx", headings=headings,
                     meta={"author": cp.author or None, "published_at": published})


_MD_HEADING = re.compile(r"^(#{1,6})\s+(.+?)\s*#*\s*$")
_MD_LINK = re.compile(r"!?\[([^\]]*)\]\([^)]*\)")
_MD_EMPH = re.compile(r"(\*\*|__|\*|_|`)(?=\S)(.+?)(?<=\S)\1")


def extract_markdown(raw: str, *, kind: str = "markdown") -> Extracted:
    out_lines: list[str] = []
    headings: list[tuple[str, int]] = []
    cursor = 0
    in_code = False
    for line in raw.replace("\r\n", "\n").split("\n"):
        if line.strip().startswith("```"):
            in_code = not in_code
            continue
        m = None if in_code else _MD_HEADING.match(line)
        if m:
            heading = _MD_EMPH.sub(r"\2", _MD_LINK.sub(r"\1", m.group(2))).strip()
            if out_lines and out_lines[-1] != "":
                out_lines.append("")
                cursor += 1
            headings.append((heading, cursor))
            out_lines.extend([heading, ""])
            cursor += len(heading) + 2
            continue
        ln = line if in_code else _MD_EMPH.sub(r"\2", _MD_LINK.sub(r"\1", line))
        ln = re.sub(r"^\s*[-*+]\s+", "• ", ln)
        out_lines.append(ln)
        cursor += len(ln) + 1
    text = "\n".join(out_lines)
    cleaned = clean_text(text)
    # Cleaning can shift offsets; recompute heading positions against the cleaned text.
    fixed: list[tuple[str, int]] = []
    pos = 0
    for h, _ in headings:
        i = cleaned.find(h, pos)
        if i >= 0:
            fixed.append((h, i))
            pos = i + len(h)
    if len(cleaned) < 20:
        raise ExtractionError("document contains no text")
    title = fixed[0][0] if fixed else cleaned.split("\n", 1)[0][:120]
    return Extracted(text=cleaned, title=title, kind=kind, headings=fixed)


def extract_text_plain(raw: str) -> Extracted:
    text = _unwrap(clean_text(raw))
    if len(text) < 20:
        raise ExtractionError("document contains no text")
    title = next((ln for ln in text.split("\n") if 4 <= len(ln) <= 120), "")
    return Extracted(text=text, title=title, kind="text", headings=_find_headings(text))


def extract_csv(raw: str, filename: str) -> Extracted:
    try:
        dialect = csv.Sniffer().sniff(raw[:4096], delimiters=",;\t|")
    except csv.Error:
        dialect = csv.excel
    reader = csv.reader(io.StringIO(raw), dialect)
    rows = list(reader)
    if not rows:
        raise ExtractionError("CSV is empty")
    header = [h.strip() or f"column_{i + 1}" for i, h in enumerate(rows[0])]
    body = rows[1:]
    warnings = []
    if len(body) > MAX_CSV_ROWS:
        warnings.append(f"truncated to the first {MAX_CSV_ROWS} of {len(body)} rows")
        body = body[:MAX_CSV_ROWS]
    lines: list[str] = []
    headings: list[tuple[str, int]] = []
    cursor = 0
    block = 50
    for i, r in enumerate(body):
        if i % block == 0:
            h = f"Rows {i + 1}-{min(i + block, len(body))}"
            if lines:
                lines.append("")
                cursor += 1
            headings.append((h, cursor))
            lines.append(h)
            cursor += len(h) + 1
        cells = [f"{header[j] if j < len(header) else f'column_{j + 1}'}: {v.strip()}"
                 for j, v in enumerate(r) if v.strip()]
        line = f"Row {i + 1}. " + "; ".join(cells) + "."
        lines.append(line)
        cursor += len(line) + 1
    text = "\n".join(lines)
    if not body:
        raise ExtractionError("CSV has a header but no rows")
    return Extracted(text=text, title=filename.rsplit(".", 1)[0], kind="csv", headings=headings,
                     meta={"columns": header, "rows": len(body)}, warnings=warnings)


def extract_html(raw: str, url: str | None = None) -> Extracted:
    import trafilatura

    md = trafilatura.extract(
        raw, url=url, output_format="markdown", include_tables=True, include_links=False,
        include_comments=False, favor_precision=True,
    )
    if not md or len(md.strip()) < 40:
        raise ExtractionError("page has no extractable main content")
    meta_obj = trafilatura.extract_metadata(raw, default_url=url)
    meta: dict[str, Any] = {}
    if meta_obj is not None:
        d = meta_obj.as_dict() if hasattr(meta_obj, "as_dict") else dict(meta_obj)
        meta = {k: d.get(k) for k in ("title", "author", "date", "sitename", "description") if d.get(k)}
    ex = extract_markdown(md, kind="html")
    if meta.get("title"):
        ex.title = str(meta["title"])
    ex.meta.update({"site": meta.get("sitename"), "author": meta.get("author"),
                    "published_at": meta.get("date"), "description": meta.get("description")})
    return ex


def extract(data: bytes, filename: str, *, url: str | None = None, content_type: str | None = None) -> Extracted:
    """Dispatch by sniffed content type."""
    if not data:
        raise ExtractionError("empty file")
    if content_type and "html" in content_type:
        kind = "html"
    elif content_type and "pdf" in content_type and data[:4] == b"%PDF":
        kind = "pdf"
    else:
        kind = sniff_kind(data, filename)
    if kind == "pdf":
        return extract_pdf(data)
    if kind == "docx":
        return extract_docx(data)
    raw = decode_text(data)
    if kind == "markdown":
        return extract_markdown(raw)
    if kind == "csv":
        return extract_csv(raw, filename)
    if kind == "html":
        return extract_html(raw, url)
    return extract_text_plain(raw)


__all__ = ["Extracted", "ExtractionError", "SUPPORTED", "clean_text", "extract", "sniff_kind"]
