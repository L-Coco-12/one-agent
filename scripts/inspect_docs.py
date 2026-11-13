"""命令行工具：加载知识库文档，输出可读性与质量报告。

用法::

    uv run python scripts/inspect_docs.py
    uv run python scripts/inspect_docs.py --docs-dir knowledge --out-dir .rag/extracted
    uv run python scripts/inspect_docs.py --preview-chars 300

环境变量（与项目其他配置风格一致）::

    RAG_DOCS_DIR     原始资料目录，默认 knowledge
    RAG_EXTRACT_DIR  抽取结果目录，默认 .rag/extracted

退出码：0 = 全部加载成功；1 = 有文件加载失败。
WARN 不影响退出码 —— 质检的目的是"让人看一眼"，不是自动失败。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from rag_loader import (  # noqa: E402  (需要先设置 sys.path)
    check_quality,
    document_metrics,
    extract_document,
    load_documents,
)


def resolve_path(value: str | None, env_name: str, default: str) -> Path:
    """命令行参数 > 环境变量 > 默认值；相对路径按仓库根目录解析。"""
    raw = value or os.getenv(env_name) or default
    path = Path(raw).expanduser()
    return path if path.is_absolute() else PROJECT_ROOT / path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="检查 RAG 知识库文档的可读性")
    parser.add_argument("--docs-dir", help="原始资料目录（默认取 RAG_DOCS_DIR 或 knowledge）")
    parser.add_argument("--out-dir", help="抽取结果目录（默认取 RAG_EXTRACT_DIR 或 .rag/extracted）")
    parser.add_argument(
        "--preview-chars",
        type=int,
        default=120,
        help="每篇文档打印多少字符的预览（默认 120）",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    docs_dir = resolve_path(args.docs_dir, "RAG_DOCS_DIR", "knowledge")
    out_dir = resolve_path(args.out_dir, "RAG_EXTRACT_DIR", ".rag/extracted")

    print(f"知识库目录: {docs_dir}")
    if not docs_dir.is_dir():
        print(f"[FAILED] 目录不存在：{docs_dir}")
        print("        请先把资料放进 knowledge/，或用 --docs-dir 指定目录。")
        return 1

    documents, failures = load_documents(docs_dir)
    print(f"共 {len(documents) + len(failures)} 个文件，成功 {len(documents)}，失败 {len(failures)}")
    print()

    report_documents: list[dict[str, object]] = []
    warn_count = 0

    for doc in documents:
        warnings = check_quality(doc)
        metrics = document_metrics(doc)
        status = "[WARN]" if warnings else "[OK]"
        warn_count += 1 if warnings else 0
        print(
            f"{status} {doc.source:<36} "
            f"{int(metrics['pages'])} 页  {int(metrics['chars'])} 字符  ({doc.encoding})"
        )
        preview = doc.text[: args.preview_chars].replace("\n", " ")
        if preview:
            print(f"       前 {args.preview_chars} 字: {preview}")
        for warning in warnings:
            print(f"       ! {warning}")

        text_path, json_path = extract_document(doc, out_dir)
        report_documents.append(
            {
                "source": doc.source,
                "title": doc.title,
                "encoding": doc.encoding,
                "metrics": metrics,
                "warnings": warnings,
                "extracted_text": str(text_path),
                "extracted_json": str(json_path),
            }
        )

    for source, error in failures:
        print(f"[FAILED] {source:<34} {error}")

    report_path = out_dir.parent / "extraction_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "docs_dir": str(docs_dir),
        "out_dir": str(out_dir),
        "document_count": len(documents),
        "failure_count": len(failures),
        "warning_count": warn_count,
        "documents": report_documents,
        "failures": [{"source": source, "error": error} for source, error in failures],
    }
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print()
    print(f"抽取结果: {out_dir}")
    print(f"质检报告: {report_path}")
    if failures:
        print(f"[FAILED] {len(failures)} 个文件加载失败，请先修复再进入切分阶段。")
        return 1
    if warn_count:
        print(f"[WARN] {warn_count} 个文件有可疑点，建议先人工确认预览内容。")
    else:
        print("[OK] 全部文档可读，可以进入下一步（切分）。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
