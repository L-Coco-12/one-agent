"""把抽取后的文档切成适合向量检索的 chunk（第②步）。

设计要点：
1. Markdown 先按标题层级切（MarkdownHeaderTextSplitter），保住“章节”语义；
   过长的章节再用递归字符切分器按句子/标点切。
2. 每个 chunk 的正文前面补上“章节路径”（如 ``退换货政策 > 退货流程``），
   这样即使 chunk 被单独召回，模型也知道它属于哪一节。
3. PDF 按页切，chunk 记录 page_start，便于回答时标注“第几页”。
4. chunk 的切分参数必须记录在报告里——参数一变，向量索引就必须重建。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)

from rag_loader import Document, document_id

# 默认参数：中文资料下 500 字符 ≈ 350–450 token，远低于智谱 embedding-3 的 3072 token 上限。
DEFAULT_CHUNK_SIZE = 500
DEFAULT_CHUNK_OVERLAP = 100  # 约 20%，防止一句话被切断后两边都答不全

# 中文标点全部纳入分隔符：优先在段落、句号处断开，最后才按空格/字符切。
CHINESE_SEPARATORS = ["\n\n", "\n", "。", "！", "？", "；", "，", " ", ""]

MARKDOWN_HEADERS = [("#", "h1"), ("##", "h2"), ("###", "h3")]

# 统计用：低于这个字符数的 chunk 记为“偏小片段”（可能是空章节或孤立小标题）。
TINY_CHUNK_CHARS = 50


@dataclass
class Chunk:
    """一个可被 embedding / 召回的文本片段。"""

    id: str
    text: str
    source: str
    title: str
    chunk_index: int
    section: str
    page_start: int | None
    metadata: dict[str, object]

    def to_dict(self) -> dict[str, object]:
        return {
            "id": self.id,
            "text": self.text,
            "source": self.source,
            "title": self.title,
            "chunk_index": self.chunk_index,
            "section": self.section,
            "page_start": self.page_start,
            "metadata": self.metadata,
        }


def _build_recursive_splitter(
    chunk_size: int, chunk_overlap: int
) -> RecursiveCharacterTextSplitter:
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=CHINESE_SEPARATORS,
        keep_separator=True,
        length_function=len,
    )


def _section_path(header_metadata: dict[str, object]) -> str:
    """把 h1/h2/h3 拼成 ``一级 > 二级 > 三级`` 的章节路径。"""
    parts = [
        str(header_metadata[key])
        for key in ("h1", "h2", "h3")
        if header_metadata.get(key)
    ]
    return " > ".join(parts)


def _chunk_text(section: str, body: str) -> str:
    """章节路径作为上下文前缀；正文为空时返回空串（调用方会跳过）。"""
    body = body.strip()
    if not body:
        return ""
    return f"{section}\n\n{body}" if section else body


def _make_chunk(
    *,
    source: str,
    title: str,
    index: int,
    section: str,
    body: str,
    page_start: int | None,
    document_metadata: dict[str, object],
) -> Chunk:
    text = _chunk_text(section, body)
    metadata = dict(document_metadata)
    metadata["chunk_index"] = index
    if section:
        metadata["section"] = section
    if page_start is not None:
        metadata["page_start"] = page_start
    return Chunk(
        id=f"{document_id(source)}-c{index:03d}",
        text=text,
        source=source,
        title=title,
        chunk_index=index,
        section=section,
        page_start=page_start,
        metadata=metadata,
    )


def _split_markdown(
    document: Document, chunk_size: int, chunk_overlap: int
) -> list[Chunk]:
    text = document.text
    if not text.strip():
        return []

    header_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=MARKDOWN_HEADERS,
        strip_headers=True,  # 标题另行拼成 section 前缀，正文里不重复
    )
    recursive_splitter = _build_recursive_splitter(chunk_size, chunk_overlap)

    chunks: list[Chunk] = []
    for section_doc in header_splitter.split_text(text):
        section = _section_path(section_doc.metadata)
        body = section_doc.page_content
        if not body.strip():
            continue
        pieces = (
            [body]
            if len(body) <= chunk_size
            else recursive_splitter.split_text(body)
        )
        for piece in pieces:
            chunks.append(
                _make_chunk(
                    source=document.source,
                    title=document.title,
                    index=len(chunks),
                    section=section,
                    body=piece,
                    page_start=None,
                    document_metadata=document.metadata,
                )
            )
    return chunks


def _split_plain_text(
    document: Document, chunk_size: int, chunk_overlap: int
) -> list[Chunk]:
    """非 Markdown（txt/单页）整体切；PDF 按页切并记录页码。"""
    recursive_splitter = _build_recursive_splitter(chunk_size, chunk_overlap)
    chunks: list[Chunk] = []

    # PDF 超过一页时按页切，保留 page_start 便于溯源；其他文档走整体切分。
    is_paginated_pdf = document.metadata.get("format") == "pdf" and len(document.pages) > 1
    if is_paginated_pdf:
        segments: Iterator[tuple[int | None, str]] = (
            (page.page, page.text) for page in document.pages
        )
    else:
        segments = iter([(None, document.text)])

    for page_number, segment_text in segments:
        if not segment_text.strip():
            continue
        for piece in recursive_splitter.split_text(segment_text):
            chunks.append(
                _make_chunk(
                    source=document.source,
                    title=document.title,
                    index=len(chunks),
                    section="",
                    body=piece,
                    page_start=page_number,
                    document_metadata=document.metadata,
                )
            )
    return chunks


def split_document(
    document: Document,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[Chunk]:
    """切分单篇文档；返回按 chunk_index 递增的列表。"""
    if chunk_size <= 0:
        raise ValueError("chunk_size 必须大于 0")
    if chunk_overlap < 0:
        raise ValueError("chunk_overlap 不能为负数")
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap 必须小于 chunk_size，否则切分会原地打转")

    suffix = Path(document.source).suffix.lower()
    if suffix in {".md", ".markdown"}:
        return _split_markdown(document, chunk_size, chunk_overlap)
    return _split_plain_text(document, chunk_size, chunk_overlap)


def split_documents(
    documents: list[Document],
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[Chunk]:
    chunks: list[Chunk] = []
    for document in documents:
        chunks.extend(split_document(document, chunk_size, chunk_overlap))
    return chunks


def chunk_stats(chunks: list[Chunk]) -> dict[str, object]:
    """汇总 chunk 长度分布，用于判断切分参数是否合理。"""
    lengths = [len(chunk.text) for chunk in chunks]
    if not lengths:
        return {
            "count": 0,
            "avg_chars": 0.0,
            "min_chars": 0,
            "max_chars": 0,
            "tiny_count": 0,
        }
    return {
        "count": len(lengths),
        "avg_chars": sum(lengths) / len(lengths),
        "min_chars": min(lengths),
        "max_chars": max(lengths),
        "tiny_count": sum(1 for length in lengths if length < TINY_CHUNK_CHARS),
    }
