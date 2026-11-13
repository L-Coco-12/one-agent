"""scripts/rag_loader.py 与 scripts/inspect_docs.py 的单元测试。

注意：scripts/ 不是安装包，这里通过 sys.path 直接导入，
避免为两个脚本引入额外的打包配置。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from inspect_docs import main  # noqa: E402
from rag_loader import (  # noqa: E402
    Document,
    Page,
    check_quality,
    document_metrics,
    extract_document,
    iter_documents,
    load_text_document,
    split_frontmatter,
)


def test_split_frontmatter_extracts_metadata() -> None:
    metadata, body = split_frontmatter(
        "---\ntitle: 退换货政策\nupdated_at: 2026-11-01\n---\n正文内容\n"
    )
    assert metadata["title"] == "退换货政策"
    assert metadata["updated_at"] == "2026-11-01"
    assert body.strip() == "正文内容"


def test_split_frontmatter_without_header_keeps_original_text() -> None:
    text = "# 标题\n\n普通正文"
    metadata, body = split_frontmatter(text)
    assert metadata == {}
    assert body == text


def test_load_text_document_falls_back_to_gb18030(tmp_path: Path) -> None:
    path = tmp_path / "legacy.txt"
    path.write_bytes("退换货政策：七天无理由退货。".encode("gb18030"))

    document = load_text_document(path)

    assert document.encoding == "gb18030"
    assert "退换货政策" in document.text


def test_iter_documents_skips_hidden_and_unsupported_files(tmp_path: Path) -> None:
    (tmp_path / "guide.md").write_text("# guide", encoding="utf-8")
    (tmp_path / "notes.txt").write_text("notes", encoding="utf-8")
    (tmp_path / "image.png").write_bytes(b"png")
    (tmp_path / ".hidden").mkdir()
    (tmp_path / ".hidden" / "secret.md").write_text("# secret", encoding="utf-8")
    (tmp_path / ".rag").mkdir()
    (tmp_path / ".rag" / "cached.md").write_text("# cached", encoding="utf-8")

    names = {path.name for path in iter_documents(tmp_path)}

    assert names == {"guide.md", "notes.txt"}


def test_check_quality_flags_repeated_headers() -> None:
    text = "\n".join(["某某公司内部资料"] * 3 + ["正文只有一句"])
    document = Document(source="a.md", title="a", pages=[Page(page=1, text=text)])

    warnings = check_quality(document)

    assert any("重复行" in warning for warning in warnings)


def test_check_quality_flags_scanned_pdf() -> None:
    document = Document(
        source="scan.pdf",
        title="scan",
        pages=[Page(page=i, text="") for i in range(1, 4)] + [Page(page=4, text="x")],
    )

    warnings = check_quality(document)
    metrics = document_metrics(document)

    assert metrics["empty_page_ratio"] == 0.75
    assert any("空页率" in warning for warning in warnings)
    assert any("平均每页" in warning for warning in warnings)


def test_extract_document_writes_text_and_json(tmp_path: Path) -> None:
    document = Document(
        source="product/faq.md",
        title="退换货政策",
        pages=[Page(page=1, text="第一段"), Page(page=2, text="第二段")],
        metadata={"format": "md"},
    )

    text_path, json_path = extract_document(document, tmp_path / "extracted")

    assert text_path.read_text(encoding="utf-8") == "第一段\n\n第二段"
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["source"] == "product/faq.md"
    assert payload["pages"][1] == {"page": 2, "text": "第二段"}


def test_main_writes_report_and_returns_zero(tmp_path: Path) -> None:
    docs_dir = tmp_path / "kb"
    (docs_dir / "product").mkdir(parents=True)
    (docs_dir / "product" / "faq.md").write_text(
        "---\ntitle: 常见问题\n---\n\n这里是一段足够长的正文，用来通过质量检查，"
        "长度超过平均每页下限的要求，因此不应该出现告警。",
        encoding="utf-8",
    )
    out_dir = tmp_path / "extracted"

    exit_code = main(["--docs-dir", str(docs_dir), "--out-dir", str(out_dir)])

    assert exit_code == 0
    report = json.loads((tmp_path / "extraction_report.json").read_text(encoding="utf-8"))
    assert report["document_count"] == 1
    assert report["failure_count"] == 0
    assert report["documents"][0]["title"] == "常见问题"


def test_main_returns_one_when_docs_dir_missing(tmp_path: Path) -> None:
    exit_code = main(["--docs-dir", str(tmp_path / "missing")])
    assert exit_code == 1
