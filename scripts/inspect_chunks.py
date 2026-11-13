"""命令行切分质检工具（第②步）。

读取 knowledge/ 下的原始资料，按 rag_splitter 的规则切分，
把每个 chunk 写成 JSONL，并生成切分报告，方便先用肉眼检查再建向量索引。

用法：
    uv run python scripts/inspect_chunks.py
    uv run python scripts/inspect_chunks.py --chunk-size 700 --overlap 120 --samples 3
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
sys.path.insert(0, str(SCRIPT_DIR))

from rag_loader import document_id, load_documents  # noqa: E402
from rag_splitter import (  # noqa: E402
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_CHUNK_SIZE,
    TINY_CHUNK_CHARS,
    Chunk,
    chunk_stats,
    split_documents,
)


def resolve_path(value: str | None, env_name: str, default: Path) -> Path:
    """优先级：命令行参数 > 环境变量 > 默认值；相对路径按项目根目录解析。"""
    raw = value or os.getenv(env_name) or str(default)
    path = Path(raw)
    return path if path.is_absolute() else PROJECT_ROOT / path


def resolve_int(value: int | None, env_name: str, default: int) -> int:
    if value is not None:
        return value
    raw = os.getenv(env_name)
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise SystemExit(f"[FAILED] 环境变量 {env_name}={raw!r} 不是整数") from exc


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="切分 knowledge/ 下的文档并生成质检报告")
    parser.add_argument("--docs-dir", help="原始资料目录，默认 knowledge/（环境变量 RAG_DOCS_DIR）")
    parser.add_argument("--out-dir", help="chunk 输出目录，默认 .rag/chunks（环境变量 RAG_CHUNKS_DIR）")
    parser.add_argument("--chunk-size", type=int, help="每个 chunk 的最大字符数，默认 500")
    parser.add_argument("--overlap", type=int, help="相邻 chunk 的重叠字符数，默认 100")
    parser.add_argument("--samples", type=int, default=2, help="每篇文档预览几个 chunk，默认 2")
    return parser


def preview(text: str, limit: int = 160) -> str:
    text = text.replace("\n", " ")
    return text if len(text) <= limit else text[:limit] + "…"


def write_chunks(chunks: list[Chunk], out_dir: Path) -> list[Path]:
    """每篇文档一个 JSONL 文件；返回写入的文件路径列表。"""
    grouped: dict[str, list[Chunk]] = {}
    for chunk in chunks:
        grouped.setdefault(chunk.source, []).append(chunk)

    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for source, source_chunks in grouped.items():
        path = out_dir / f"{document_id(source)}.jsonl"
        lines = [json.dumps(chunk.to_dict(), ensure_ascii=False) for chunk in source_chunks]
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        written.append(path)
    return written


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    docs_dir = resolve_path(args.docs_dir, "RAG_DOCS_DIR", Path("knowledge"))
    out_dir = resolve_path(args.out_dir, "RAG_CHUNKS_DIR", Path(".rag/chunks"))
    chunk_size = resolve_int(args.chunk_size, "RAG_CHUNK_SIZE", DEFAULT_CHUNK_SIZE)
    chunk_overlap = resolve_int(args.overlap, "RAG_CHUNK_OVERLAP", DEFAULT_CHUNK_OVERLAP)

    print(f"资料目录: {docs_dir}")
    print(f"切分参数: chunk_size={chunk_size}, overlap={chunk_overlap}")
    if not docs_dir.exists():
        print(f"[FAILED] 资料目录不存在: {docs_dir}")
        return 1

    documents, failures = load_documents(docs_dir)
    for source, error in failures:
        print(f"[FAILED] {source}: {error}")

    try:
        chunks = split_documents(
            documents, chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )
    except ValueError as exc:
        print(f"[FAILED] 切分参数不合法: {exc}")
        return 1

    grouped: dict[str, list[Chunk]] = {}
    for chunk in chunks:
        grouped.setdefault(chunk.source, []).append(chunk)

    document_reports: list[dict[str, object]] = []
    for document in documents:
        source_chunks = grouped.get(document.source, [])
        stats = chunk_stats(source_chunks)
        print(
            f"\n[OK] {document.source}  {stats['count']} 个 chunk  "
            f"平均 {stats['avg_chars']:.0f} 字符  "
            f"最短 {stats['min_chars']}  最长 {stats['max_chars']}  偏小 {stats['tiny_count']}"
        )
        for chunk in source_chunks[: max(0, args.samples)]:
            section = f" [{chunk.section}]" if chunk.section else ""
            print(f"   #{chunk.chunk_index}{section} ({len(chunk.text)} 字符) {preview(chunk.text)}")
        if stats["tiny_count"]:
            print(
                f"   [WARN] 有 {stats['tiny_count']} 个 chunk 少于 {TINY_CHUNK_CHARS} 字符，"
                "可能是孤立小标题或空章节，建议检查原文结构"
            )
        if int(stats["max_chars"]) > chunk_size * 2:
            print("   [WARN] 最长 chunk 超过 chunk_size 的 2 倍，检查是否有超长无标点段落")

        document_reports.append(
            {
                "source": document.source,
                "title": document.title,
                "encoding": document.encoding,
                "metrics": asdict_chunk_metrics(stats),
                "chunk_ids": [chunk.id for chunk in source_chunks],
            }
        )

    written = write_chunks(chunks, out_dir)
    total_stats = chunk_stats(chunks)
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "docs_dir": str(docs_dir),
        "out_dir": str(out_dir),
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
        "document_count": len(documents),
        "chunk_count": len(chunks),
        "tiny_chunk_count": total_stats["tiny_count"],
        "avg_chars": total_stats["avg_chars"],
        "documents": document_reports,
        "failures": [{"source": source, "error": error} for source, error in failures],
    }
    report_path = out_dir.parent / "chunks_report.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(f"\nchunk 文件: {len(written)} 个 -> {out_dir}")
    for path in written:
        print(f"  {path.name}")
    print(f"切分报告: {report_path}")
    if failures:
        return 1
    if not chunks:
        print("[WARN] 没有切出任何 chunk，请检查资料是否为空")
        return 1
    print("[OK] 切分完成，下一步：生成 embedding 并写入 Chroma（第③步）。")
    return 0


def asdict_chunk_metrics(stats: dict[str, object]) -> dict[str, object]:
    """把 chunk_stats 的返回值转成可 JSON 序列化的 dict（当前已是基础类型）。"""
    return {
        "count": stats["count"],
        "avg_chars": round(float(stats["avg_chars"]), 2),
        "min_chars": stats["min_chars"],
        "max_chars": stats["max_chars"],
        "tiny_count": stats["tiny_count"],
    }


if __name__ == "__main__":
    raise SystemExit(main())
