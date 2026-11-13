"""scripts/rag_splitter.py 与 scripts/inspect_chunks.py 的单元测试。"""

from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from inspect_chunks import main  # noqa: E402
from rag_loader import Document, Page  # noqa: E402
from rag_splitter import chunk_stats, split_document, split_documents  # noqa: E402

LONG_BODY = "这是一句用于测试切分的句子。" * 80


def _document(
    *,
    source: str = "product/faq.md",
    title: str = "退换货政策",
    pages: list[Page] | None = None,
    metadata: dict[str, object] | None = None,
) -> Document:
    return Document(
        source=source,
        title=title,
        pages=pages or [Page(page=1, text="# 退换货政策\n\n## 七天无理由退货\n\n支持七天无理由退货。")],
        encoding="utf-8",
        metadata=metadata if metadata is not None else {"format": "md"},
    )


def test_markdown_chunks_carry_section_path() -> None:
    chunks = split_document(_document(), chunk_size=500, chunk_overlap=50)

    assert len(chunks) == 1
    chunk = chunks[0]
    assert chunk.section == "退换货政策 > 七天无理由退货"
    assert chunk.text.startswith("退换货政策 > 七天无理由退货\n\n")
    assert chunk.text.endswith("支持七天无理由退货。")
    assert chunk.metadata["section"] == "退换货政策 > 七天无理由退货"


def test_long_markdown_section_is_split_with_overlap() -> None:
    document = _document(
        pages=[Page(page=1, text=f"# 退换货政策\n\n## 长章节\n\n{LONG_BODY}")]
    )

    chunks = split_document(document, chunk_size=200, chunk_overlap=40)

    assert len(chunks) > 1
    bodies = [chunk.text.split("\n\n", 1)[1] for chunk in chunks]
    assert all(len(body) <= 200 for body in bodies)
    # 相邻 chunk 应该有重叠：前一个 chunk 的尾部出现在后一个 chunk 里
    assert any(bodies[index][-20:] in bodies[index + 1] for index in range(len(bodies) - 1))
    # 有 overlap 时，切分后的总长度必然大于原文
    assert sum(len(body) for body in bodies) > len(LONG_BODY)


def test_chunk_ids_are_stable_and_unique() -> None:
    document = _document(
        pages=[Page(page=1, text=f"# 退换货政策\n\n## 长章节\n\n{LONG_BODY}")]
    )

    first = split_document(document, chunk_size=200, chunk_overlap=40)
    second = split_document(document, chunk_size=200, chunk_overlap=40)

    assert [chunk.id for chunk in first] == [chunk.id for chunk in second]
    assert len({chunk.id for chunk in first}) == len(first)
    assert first[0].id.startswith("product__faq.md-c")


def test_chunk_metadata_copies_document_metadata() -> None:
    document = _document(metadata={"format": "md", "owner": "客服团队"})

    chunk = split_document(document, chunk_size=500, chunk_overlap=50)[0]

    assert chunk.metadata["owner"] == "客服团队"
    assert chunk.metadata["format"] == "md"
    assert chunk.metadata["chunk_index"] == 0
    assert "page_start" not in chunk.metadata


def test_pdf_chunks_track_page_numbers() -> None:
    document = _document(
        source="manual/guide.pdf",
        title="用户手册",
        pages=[
            Page(page=1, text="第一页内容"),
            Page(page=2, text="第二页内容"),
            Page(page=3, text="第三页内容"),
        ],
        metadata={"format": "pdf"},
    )

    chunks = split_document(document, chunk_size=500, chunk_overlap=50)

    assert [chunk.page_start for chunk in chunks] == [1, 2, 3]
    assert all(chunk.metadata["page_start"] == chunk.page_start for chunk in chunks)


def test_plain_text_document_has_no_section() -> None:
    document = _document(
        source="notes.txt",
        pages=[Page(page=1, text="第一段。\n\n第二段。")],
        metadata={"format": ".txt"},
    )

    chunks = split_document(document, chunk_size=500, chunk_overlap=50)

    assert len(chunks) == 1
    assert chunks[0].section == ""
    assert chunks[0].page_start is None


def test_overlap_must_be_smaller_than_chunk_size() -> None:
    try:
        split_document(_document(), chunk_size=100, chunk_overlap=100)
    except ValueError as exc:
        assert "chunk_overlap" in str(exc)
    else:  # pragma: no cover - 断言失败路径
        raise AssertionError("expected ValueError")


def test_chunk_stats_handles_empty_and_tiny() -> None:
    assert chunk_stats([])["count"] == 0

    document = _document(pages=[Page(page=1, text="## 短\n\n短")])
    chunks = split_document(document, chunk_size=500, chunk_overlap=50)
    stats = chunk_stats(chunks)

    assert stats["count"] == len(chunks)
    assert stats["tiny_count"] == len(chunks)


def test_split_documents_flattens_all_sources() -> None:
    documents = [
        _document(source="a.md"),
        _document(source="b.md", pages=[Page(page=1, text="# B\n\n内容")]),
    ]

    chunks = split_documents(documents, chunk_size=500, chunk_overlap=50)

    assert {chunk.source for chunk in chunks} == {"a.md", "b.md"}


def test_cli_writes_jsonl_and_report(tmp_path: Path) -> None:
    docs_dir = tmp_path / "kb"
    (docs_dir / "product").mkdir(parents=True)
    (docs_dir / "product" / "faq.md").write_text(
        "# 退换货政策\n\n## 退货流程\n\n1. 申请售后。\n2. 客服审核。\n",
        encoding="utf-8",
    )
    out_dir = tmp_path / "chunks"

    exit_code = main(["--docs-dir", str(docs_dir), "--out-dir", str(out_dir), "--chunk-size", "300"])

    assert exit_code == 0
    jsonl_path = out_dir / "product__faq.md.jsonl"
    assert jsonl_path.exists()
    lines = [json.loads(line) for line in jsonl_path.read_text(encoding="utf-8").splitlines()]
    assert lines and lines[0]["source"] == "product/faq.md"

    report = json.loads((tmp_path / "chunks_report.json").read_text(encoding="utf-8"))
    assert report["chunk_size"] == 300
    assert report["chunk_count"] == len(lines)
    assert report["document_count"] == 1


def test_cli_returns_one_when_docs_dir_missing(tmp_path: Path) -> None:
    exit_code = main(["--docs-dir", str(tmp_path / "missing")])
    assert exit_code == 1
