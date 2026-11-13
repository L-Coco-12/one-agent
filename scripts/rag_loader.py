"""RAG 文档加载与质检。

这个模块只做两件事：

1. 把不同格式的文档（Markdown / 纯文本 / PDF）统一成 ``Document`` + ``Page`` 结构；
2. 给出可解释的质量指标，帮助在"切分 / 向量化"之前发现脏数据。

故意不依赖 LangChain：这一步只关心"文本是否读得干净"，
切分、embedding、向量库都属于后面的流水线。
"""

from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Iterator

SUPPORTED_SUFFIXES = {".md", ".markdown", ".txt", ".pdf"}

# 阈值（超过就给出 WARN）：都是经验值，可按自己的语料调整
EMPTY_PAGE_WARN_RATIO = 0.2
REPEATED_LINE_WARN_RATIO = 0.3
MOJIBAKE_WARN_RATIO = 0.001
MIN_AVG_CHARS_PER_PAGE = 50

# U+FFFD 是解码失败时的替换字符，出现得多说明编码有问题
REPLACEMENT_CHAR = "\ufffd"

FRONTMATTER_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n?", re.DOTALL)


@dataclass
class Page:
    """文档中的一页（Markdown/TXT 只有一页，页码为 1）。"""

    page: int
    text: str


@dataclass
class Document:
    """一篇加载完成的文档。"""

    source: str  # 相对 docs_dir 的路径，统一用 "/" 分隔
    title: str
    pages: list[Page] = field(default_factory=list)
    encoding: str = "utf-8"
    metadata: dict[str, object] = field(default_factory=dict)

    @property
    def text(self) -> str:
        """所有页拼成的纯文本。"""
        return "\n\n".join(page.text for page in self.pages if page.text.strip())


def _normalize_text(text: str) -> str:
    """统一换行、去掉行尾空白、去掉首尾空行。"""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = [line.rstrip() for line in text.split("\n")]
    return "\n".join(lines).strip()


def _normalize_metadata(value: object) -> object:
    """把 YAML 解析出的值转成 JSON/向量库可接受的类型。

    YAML 会把 ``2026-11-01`` 解析成 ``datetime.date``，直接写 JSON 会抛
    ``TypeError: Object of type date is not JSON serializable``；
    而向量库的 metadata 也只接受 str/int/float/bool，所以统一转成字符串。
    """
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, (list, tuple)):
        return [_normalize_metadata(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _normalize_metadata(item) for key, item in value.items()}
    return str(value)


def split_frontmatter(text: str) -> tuple[dict[str, object], str]:
    """解析 Markdown 头部的 YAML frontmatter。

    形如::

        ---
        title: 退换货政策
        updated_at: 2026-11-01
        ---
        正文……

    没有 frontmatter、格式不合法、或缺少 pyyaml 时，返回 ``({}, 原文)``。
    """
    if not text.startswith("---"):
        return {}, text

    match = FRONTMATTER_RE.match(text)
    if not match:
        return {}, text

    try:
        import yaml
    except ImportError:  # pragma: no cover - pyyaml 是项目依赖
        return {}, text

    try:
        parsed = yaml.safe_load(match.group(1)) or {}
    except Exception:
        return {}, text

    if not isinstance(parsed, dict):
        return {}, text

    metadata = {str(key): _normalize_metadata(value) for key, value in parsed.items()}
    return metadata, text[match.end():]


def iter_documents(docs_dir: Path) -> Iterator[Path]:
    """递归列出所有支持的文档，跳过隐藏目录/文件（如 .git、.rag）。"""
    docs_dir = Path(docs_dir)
    for path in sorted(docs_dir.rglob("*")):
        if not path.is_file():
            continue
        relative_parts = path.relative_to(docs_dir).parts
        if any(part.startswith(".") for part in relative_parts):
            continue
        if path.suffix.lower() in SUPPORTED_SUFFIXES:
            yield path


def load_document(path: Path, docs_dir: Path) -> Document:
    """按扩展名分派到对应的加载器。"""
    source = path.relative_to(docs_dir).as_posix()
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return load_pdf_document(path, source)
    return load_text_document(path, source)


def load_text_document(path: Path, source: str | None = None) -> Document:
    """加载 Markdown / TXT。

    编码策略：先按 UTF-8 严格解码；失败则按 GB18030 解码并记录实际编码。
    这样既能处理 UTF-8 文档，也不会把 GBK 文档读成乱码。
    """
    raw = path.read_bytes()
    encoding = "utf-8"
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        text = raw.decode("gb18030", errors="replace")
        encoding = "gb18030"

    metadata, body = split_frontmatter(text)
    body = _normalize_text(body)
    title = str(metadata.get("title") or path.stem)
    metadata.setdefault("format", path.suffix.lower().lstrip("."))
    return Document(
        source=source or path.name,
        title=title,
        pages=[Page(page=1, text=body)],
        encoding=encoding,
        metadata=metadata,
    )


def load_pdf_document(path: Path, source: str | None = None) -> Document:
    """加载 PDF（逐页抽取文本）。

    注意：PDF 是"页面 + 绘字指令"，pypdf 只能按字符顺序拼文本，
    不还原版面。双栏会串行、表格会变形——用质检指标暴露问题，
    必要时再换 pdfplumber / unstructured 或走 OCR。
    """
    try:
        from pypdf import PdfReader
    except ImportError as exc:  # pragma: no cover - 由 uv add pypdf 保证
        raise RuntimeError("读取 PDF 需要 pypdf，请先执行 uv add pypdf") from exc

    reader = PdfReader(str(path))
    pages = [
        Page(page=index + 1, text=_normalize_text(page.extract_text() or ""))
        for index, page in enumerate(reader.pages)
    ]
    return Document(
        source=source or path.name,
        title=path.stem,
        pages=pages,
        encoding="pdf",
        metadata={"format": "pdf", "page_count": len(pages)},
    )


def load_documents(docs_dir: Path) -> tuple[list[Document], list[tuple[str, str]]]:
    """加载整个目录，返回 (成功的文档, [(失败的文件, 错误信息)])。"""
    docs_dir = Path(docs_dir)
    documents: list[Document] = []
    failures: list[tuple[str, str]] = []
    for path in iter_documents(docs_dir):
        try:
            documents.append(load_document(path, docs_dir))
        except Exception as exc:  # noqa: BLE001 - 加载失败要整体汇报，不能中断
            failures.append((path.relative_to(docs_dir).as_posix(), f"{type(exc).__name__}: {exc}"))
    return documents, failures


def line_repetition_ratio(text: str) -> float:
    """重复行比例：出现 >= 3 次的非空行占全部非空行的比例。

    页眉、页脚、水印、页码通常会让这个值明显偏高。
    """
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return 0.0
    counts = Counter(lines)
    repeated = sum(count for count in counts.values() if count >= 3)
    return repeated / len(lines)


def document_metrics(doc: Document) -> dict[str, float]:
    """计算一篇文档的质量指标（用于报告与阈值判断）。"""
    page_texts = [page.text for page in doc.pages]
    total_chars = sum(len(text) for text in page_texts)
    page_count = len(page_texts)
    empty_pages = sum(1 for text in page_texts if not text.strip())
    return {
        "pages": page_count,
        "chars": total_chars,
        "empty_page_ratio": empty_pages / page_count if page_count else 0.0,
        "repeated_line_ratio": line_repetition_ratio(doc.text),
        "mojibake_ratio": (doc.text.count(REPLACEMENT_CHAR) / total_chars) if total_chars else 0.0,
        "avg_chars_per_page": (total_chars / page_count) if page_count else 0.0,
    }


def check_quality(doc: Document) -> list[str]:
    """返回所有可疑点的中文说明；空列表表示没有发现问题。"""
    metrics = document_metrics(doc)
    warnings: list[str] = []

    if metrics["pages"] > 1 and metrics["empty_page_ratio"] > EMPTY_PAGE_WARN_RATIO:
        warnings.append(f"空页率 {metrics['empty_page_ratio']:.0%}，疑似扫描版 PDF 或空白页")
    if metrics["repeated_line_ratio"] > REPEATED_LINE_WARN_RATIO:
        warnings.append(f"重复行率 {metrics['repeated_line_ratio']:.0%}，疑似页眉/页脚/水印")
    if metrics["mojibake_ratio"] > MOJIBAKE_WARN_RATIO:
        warnings.append(f"乱码率 {metrics['mojibake_ratio']:.2%}，检查编码或字体嵌入")
    if metrics["pages"] > 1 and metrics["avg_chars_per_page"] < MIN_AVG_CHARS_PER_PAGE:
        warnings.append(f"平均每页仅 {metrics['avg_chars_per_page']:.0f} 字符，可能抽不出文字")

    return warnings


def document_id(source: str) -> str:
    """把相对路径转成安全的文件名前缀：``product/faq.md`` -> ``product__faq.md``。"""
    return source.replace("/", "__")


def extract_document(doc: Document, out_dir: Path) -> tuple[Path, Path]:
    """把加载结果落盘：``<id>.txt``（纯文本）与 ``<id>.json``（含分页）。"""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    file_id = document_id(doc.source)

    text_path = out_dir / f"{file_id}.txt"
    text_path.write_text(doc.text, encoding="utf-8")

    json_path = out_dir / f"{file_id}.json"
    payload = {
        "source": doc.source,
        "title": doc.title,
        "encoding": doc.encoding,
        "metadata": doc.metadata,
        "metrics": document_metrics(doc),
        "pages": [{"page": page.page, "text": page.text} for page in doc.pages],
    }
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return text_path, json_path
