"""检索验证脚本：用一句话在 Chroma 索引里找最相关的 chunk。

第③步只做“验证检索质量”，不接 agent；第④步再把它包装成 ``search_docs`` 工具。

前置：Chroma 服务已启动（``docker compose up -d chroma``）且索引已构建。

用法::

    uv run python scripts/search_docs.py --query "退货要多久才能退款？"
    uv run python scripts/search_docs.py --query "token 有效期多久？" --top-k 3 --json
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from dotenv import load_dotenv  # noqa: E402

# CLI 直接运行时自动读取仓库根目录的 .env
load_dotenv(PROJECT_ROOT / ".env")

from build_rag_index import (  # noqa: E402  (依赖上面的 sys.path 注入)
    DEFAULT_CHROMA_HOST,
    DEFAULT_CHROMA_PORT,
    DEFAULT_COLLECTION,
    chroma_settings,
    create_chroma_client,
)
from rag_embeddings import ZhipuEmbeddings, ZhipuEmbeddingsError  # noqa: E402

DEFAULT_TOP_K = 4
DEFAULT_PREVIEW_CHARS = 160


def open_vector_store(client: Any, collection: str, embeddings: ZhipuEmbeddings) -> Any:
    from langchain_chroma import Chroma

    return Chroma(
        client=client,
        collection_name=collection,
        embedding_function=embeddings,
    )


def search(vector_store: Any, query: str, *, top_k: int = DEFAULT_TOP_K) -> list[dict[str, Any]]:
    """返回按相似度倒序的结果；score = 1 - cosine 距离，越大越相似。"""
    pairs = vector_store.similarity_search_with_score(query, k=top_k)
    results: list[dict[str, Any]] = []
    for document, distance in pairs:
        metadata = document.metadata or {}
        results.append(
            {
                "score": round(1.0 - float(distance), 4),
                "distance": round(float(distance), 4),
                "source": str(metadata.get("source") or ""),
                "section": str(metadata.get("section") or ""),
                "title": str(metadata.get("title") or ""),
                "text": document.page_content,
            }
        )
    return results


def format_result(rank: int, item: dict[str, Any], *, preview_chars: int) -> str:
    text = " ".join(str(item.get("text") or "").split())
    if len(text) > preview_chars:
        text = text[:preview_chars].rstrip() + "…"
    header = f"[{rank}] score={item['score']:.4f}  source={item['source']}"
    if item.get("section"):
        header += f"  section={item['section']}"
    return f"{header}\n    {text}"


def build_parser() -> Any:
    import argparse

    parser = argparse.ArgumentParser(description="在 Chroma 索引里做语义检索验证")
    parser.add_argument("--query", required=True, help="查询文本")
    parser.add_argument("--top-k", type=int, default=None, help="返回条数，默认 4")
    parser.add_argument("--collection", default=None, help="collection 名，默认 RAG_COLLECTION 或 knowledge")
    parser.add_argument("--chroma-host", default=None, help="Chroma 主机，默认取 RAG_CHROMA_HOST 或 127.0.0.1")
    parser.add_argument("--chroma-port", type=int, default=None, help="Chroma 端口，默认取 RAG_CHROMA_PORT 或 8100")
    parser.add_argument("--preview-chars", type=int, default=DEFAULT_PREVIEW_CHARS, help="每条预览字符数")
    parser.add_argument("--json", action="store_true", help="以 JSON 输出结果，便于脚本消费")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    query = (args.query or "").strip()
    if not query:
        print("[FAILED] 查询文本为空。")
        return 1

    collection = args.collection or os.getenv("RAG_COLLECTION") or DEFAULT_COLLECTION
    top_k = args.top_k or DEFAULT_TOP_K
    if top_k <= 0:
        print("[FAILED] --top-k 必须大于 0。")
        return 1

    env_host, env_port, use_ssl = chroma_settings()
    host = args.chroma_host or env_host
    port = args.chroma_port or env_port

    embeddings: ZhipuEmbeddings | None = None
    try:
        client = create_chroma_client(host, port, use_ssl)
        embeddings = ZhipuEmbeddings.from_env()
        vector_store = open_vector_store(client, collection, embeddings)
        results = search(vector_store, query, top_k=top_k)
    except (RuntimeError, ZhipuEmbeddingsError, ValueError) as exc:
        print(f"[FAILED] {exc}")
        return 1
    finally:
        if embeddings is not None:
            embeddings.close()

    if not results:
        print("[WARN] 没有检索到任何结果。")
        return 0

    if args.json:
        print(json.dumps({"query": query, "results": results}, ensure_ascii=False, indent=2))
        return 0

    print(f"查询：{query}")
    print(f"[OK] 命中 {len(results)} 条（score 越大越相似，cosine）")
    for rank, item in enumerate(results, start=1):
        print(format_result(rank, item, preview_chars=args.preview_chars))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
