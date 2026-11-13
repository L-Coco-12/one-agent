"""构建 RAG 向量索引：chunk → 智谱 embedding → Chroma 服务 → manifest。

前置条件
--------
先启动本地 Chroma 服务（见 ``docker-compose.yml``）::

    docker compose up -d chroma

流水线
------
1. 从 ``.rag/chunks/*.jsonl`` 读第②步产出的 chunk；
2. 用 :class:`scripts.rag_embeddings.ZhipuEmbeddings` 把 chunk 转成向量；
3. 写入 Chroma（cosine 距离），由 Chroma 服务持久化；
4. 写 ``.rag/index_manifest.json``，记录 embedding/chunking 参数与文档 hash，
   下次构建时据此判断“是否需要重建”。

全量重建策略（v1）
------------------
- manifest 存在、参数与 chunk hash 都没变、且 Chroma 里条数一致 → 跳过（省钱）；
- 任一不匹配 / 传了 ``--force`` → 删除旧 collection，重新全量构建。

用法::

    docker compose up -d chroma
    uv run python scripts/build_rag_index.py            # 需要重建时才调 API
    uv run python scripts/build_rag_index.py --force    # 强制重建
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from dotenv import load_dotenv  # noqa: E402

# CLI 直接运行时自动读取仓库根目录的 .env（与 agent 的行为一致）
load_dotenv(PROJECT_ROOT / ".env")

from rag_embeddings import (  # noqa: E402  (依赖上面的 sys.path 注入)
    DEFAULT_BATCH_SIZE,
    DEFAULT_DIMENSIONS,
    DEFAULT_MODEL,
    ZhipuEmbeddings,
    ZhipuEmbeddingsError,
)

SCHEMA_VERSION = 1
DEFAULT_CHUNKS_DIR = Path(".rag/chunks")
DEFAULT_MANIFEST = Path(".rag/index_manifest.json")
DEFAULT_COLLECTION = "knowledge"
DEFAULT_CHROMA_HOST = "127.0.0.1"
DEFAULT_CHROMA_PORT = 8100
COSINE_SPACE = "cosine"
# 智谱 embedding 价格：0.5 元 / 百万 tokens
PRICE_CNY_PER_MILLION_TOKENS = 0.5


# --------------------------------------------------------------------- 工具
def resolve_path(value: str | Path | None, env_name: str, default: Path) -> Path:
    """参数 > 环境变量 > 默认值；相对路径按项目根目录解析。"""
    raw = value if value is not None else os.getenv(env_name) or default
    path = Path(raw)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path.name} 第 {line_number} 行不是合法 JSON：{exc}") from exc
    return rows


def chunk_text(chunk: dict[str, Any]) -> str:
    text = chunk.get("text")
    if not isinstance(text, str) or not text.strip():
        raise ValueError(f"chunk {chunk.get('id')!r} 的 text 为空")
    return text


def load_chunks(chunks_dir: Path) -> tuple[list[dict[str, Any]], dict[str, str]]:
    """读取所有 chunk 文件，返回 (chunks, 每个源文件的 sha256)。"""
    if not chunks_dir.is_dir():
        raise FileNotFoundError(f"chunk 目录不存在：{chunks_dir}（请先运行第②步切分）")
    files = sorted(chunks_dir.glob("*.jsonl"))
    if not files:
        raise FileNotFoundError(f"{chunks_dir} 下没有 *.jsonl 文件（请先运行第②步切分）")

    chunks: list[dict[str, Any]] = []
    hashes: dict[str, str] = {}
    for file_path in files:
        rows = load_jsonl(file_path)
        if not rows:
            continue
        for row in rows:
            chunk_text(row)  # 提前校验，避免 embedding 到一半才失败
            chunks.append(row)
        source = str(rows[0].get("source") or file_path.stem)
        hashes[source] = sha256_file(file_path)
    return chunks, hashes


def load_chunking_params(chunks_dir: Path) -> tuple[int, int]:
    """优先读第②步的 chunks_report.json，回退到环境变量/默认值。"""
    report_path = chunks_dir.parent / "chunks_report.json"
    if report_path.is_file():
        try:
            report = json.loads(report_path.read_text(encoding="utf-8"))
            return int(report["chunk_size"]), int(report["chunk_overlap"])
        except (KeyError, TypeError, ValueError):
            pass
    return (
        int(os.getenv("RAG_CHUNK_SIZE") or 500),
        int(os.getenv("RAG_CHUNK_OVERLAP") or 100),
    )


def to_document(chunk: dict[str, Any]) -> Any:
    """把 chunk 转成 langchain Document，并把 metadata 压成 Chroma 允许的标量。"""
    from langchain_core.documents import Document

    metadata: dict[str, Any] = {}
    for key in ("source", "title", "section"):
        value = chunk.get(key)
        if isinstance(value, str) and value:
            metadata[key] = value
    metadata["chunk_index"] = int(chunk.get("chunk_index") or 0)
    if chunk.get("page_start") is not None:
        metadata["page_start"] = int(chunk["page_start"])
    for key, value in (chunk.get("metadata") or {}).items():
        if isinstance(value, (str, int, float, bool)):
            metadata[key] = value
        elif isinstance(value, list):
            # Chroma 只接受标量 metadata，列表压成逗号分隔字符串
            metadata[key] = ", ".join(str(item) for item in value)
    return Document(page_content=chunk["text"], metadata=metadata)


# ------------------------------------------------------------------ Chroma
def chroma_settings() -> tuple[str, int, bool]:
    """从环境变量读 Chroma 连接信息（host, port, ssl）。"""
    host = os.getenv("RAG_CHROMA_HOST") or DEFAULT_CHROMA_HOST
    port = int(os.getenv("RAG_CHROMA_PORT") or DEFAULT_CHROMA_PORT)
    use_ssl = (os.getenv("RAG_CHROMA_SSL") or "false").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    return host, port, use_ssl


def create_chroma_client(host: str, port: int, use_ssl: bool = False) -> Any:
    """连接 Chroma 服务并做一次 heartbeat 探活。"""
    import chromadb

    client = chromadb.HttpClient(host=host, port=port, ssl=use_ssl)
    try:
        client.heartbeat()
    except Exception as exc:  # 连接失败、服务未启动等
        raise RuntimeError(
            f"无法连接 Chroma 服务 {host}:{port}（请先执行 docker compose up -d chroma）：{exc}"
        ) from exc
    return client


def delete_collection(client: Any, collection: str) -> None:
    """删除旧 collection；不存在时静默跳过。"""
    try:
        client.delete_collection(collection)
    except Exception:  # NotFoundError 等：collection 本来就不存在
        pass


def collection_count(client: Any, collection: str) -> int | None:
    """查 collection 里的向量条数；collection 不存在返回 None。"""
    try:
        return int(client.get_collection(collection).count())
    except Exception:
        return None


# ------------------------------------------------------------------ manifest
def load_manifest(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def manifest_stale_reasons(
    manifest: dict[str, Any] | None,
    *,
    model: str,
    dimensions: int,
    chunk_size: int,
    chunk_overlap: int,
    collection: str,
    chunk_count: int,
    source_hashes: dict[str, str],
) -> list[str]:
    """返回“为什么必须重建”的原因列表；空列表表示可以跳过构建。"""
    if manifest is None:
        return ["manifest 不存在"]

    reasons: list[str] = []
    if manifest.get("schema_version") != SCHEMA_VERSION:
        reasons.append(f"manifest 版本 {manifest.get('schema_version')} != {SCHEMA_VERSION}")

    embedding = manifest.get("embedding") or {}
    if embedding.get("model") != model:
        reasons.append(f"embedding 模型 {embedding.get('model')} != {model}")
    if int(embedding.get("dimensions") or 0) != dimensions:
        reasons.append(f"向量维度 {embedding.get('dimensions')} != {dimensions}")

    chunking = manifest.get("chunking") or {}
    if int(chunking.get("chunk_size") or 0) != chunk_size:
        reasons.append(f"chunk_size {chunking.get('chunk_size')} != {chunk_size}")
    if int(chunking.get("chunk_overlap") or 0) != chunk_overlap:
        reasons.append(
            f"chunk_overlap {chunking.get('chunk_overlap')} != {chunk_overlap}"
        )

    if (manifest.get("collection") or {}).get("name") != collection:
        reasons.append(
            f"collection {((manifest.get('collection') or {}).get('name'))} != {collection}"
        )

    recorded_chunks = manifest.get("chunks") or {}
    if int(recorded_chunks.get("count") or 0) != chunk_count:
        reasons.append(f"chunk 数 {recorded_chunks.get('count')} != {chunk_count}")

    recorded_sources = {
        str(item.get("source")): str(item.get("sha256"))
        for item in recorded_chunks.get("sources") or []
    }
    for source, digest in source_hashes.items():
        if recorded_sources.get(source) != digest:
            reasons.append(f"源文件 {source} 内容已变化（sha256 不一致）")
    return reasons


def write_manifest(
    path: Path,
    *,
    embeddings: ZhipuEmbeddings,
    chunk_size: int,
    chunk_overlap: int,
    collection: str,
    host: str,
    port: int,
    chunks: list[dict[str, Any]],
    source_hashes: dict[str, str],
) -> dict[str, Any]:
    prompt_tokens = embeddings.token_usage.get("prompt_tokens", 0)
    manifest: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "embedding": {
            "provider": "zhipu",
            "model": embeddings.model,
            "dimensions": embeddings.dimensions,
            "batch_size": embeddings.batch_size,
        },
        "chunking": {"chunk_size": chunk_size, "chunk_overlap": chunk_overlap},
        "collection": {
            "name": collection,
            "space": COSINE_SPACE,
            "server": f"{host}:{port}",
        },
        "chunks": {
            "count": len(chunks),
            "sources": [
                {
                    "source": source,
                    "sha256": digest,
                    "chunks": sum(1 for chunk in chunks if chunk.get("source") == source),
                }
                for source, digest in sorted(source_hashes.items())
            ],
        },
        "token_usage": {
            "prompt_tokens": prompt_tokens,
            "total_tokens": embeddings.token_usage.get("total_tokens", 0),
            "requests": embeddings.request_count,
            "estimated_cost_cny": round(
                prompt_tokens / 1_000_000 * PRICE_CNY_PER_MILLION_TOKENS, 6
            ),
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return manifest


# --------------------------------------------------------------- 索引构建
def build_index(
    chunks: list[dict[str, Any]],
    *,
    embeddings: ZhipuEmbeddings,
    client: Any,
    collection: str,
    chunk_size: int,
    chunk_overlap: int,
    manifest_path: Path,
    source_hashes: dict[str, str],
    host: str = DEFAULT_CHROMA_HOST,
    port: int = DEFAULT_CHROMA_PORT,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> dict[str, Any]:
    """全量重建 Chroma 索引并写 manifest。"""
    from langchain_chroma import Chroma

    delete_collection(client, collection)
    vector_store = Chroma(
        client=client,
        collection_name=collection,
        embedding_function=embeddings,
        collection_metadata={"hnsw:space": COSINE_SPACE},
    )

    written = 0
    for start in range(0, len(chunks), batch_size):
        batch = chunks[start : start + batch_size]
        documents = [to_document(chunk) for chunk in batch]
        ids = [str(chunk["id"]) for chunk in batch]
        vector_store.add_documents(documents, ids=ids)
        written += len(batch)
        print(f"  写入 {written}/{len(chunks)} 条向量")

    stored = collection_count(client, collection)
    if stored != len(chunks):
        raise RuntimeError(
            f"Chroma 中向量条数 {stored} 与 chunk 数 {len(chunks)} 不一致，请检查写入过程。"
        )

    manifest = write_manifest(
        manifest_path,
        embeddings=embeddings,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        collection=collection,
        host=host,
        port=port,
        chunks=chunks,
        source_hashes=source_hashes,
    )
    return {"count": stored, "manifest": manifest, "status": "built"}


# --------------------------------------------------------------------- CLI
def build_parser() -> Any:
    import argparse

    parser = argparse.ArgumentParser(
        description="构建 RAG 向量索引（智谱 embedding + Chroma 服务）"
    )
    parser.add_argument("--chunks-dir", default=None, help="chunk 目录，默认 .rag/chunks")
    parser.add_argument("--manifest", default=None, help="manifest 路径，默认 .rag/index_manifest.json")
    parser.add_argument("--collection", default=None, help="collection 名，默认取 RAG_COLLECTION 或 knowledge")
    parser.add_argument("--chroma-host", default=None, help="Chroma 主机，默认取 RAG_CHROMA_HOST 或 127.0.0.1")
    parser.add_argument("--chroma-port", type=int, default=None, help="Chroma 端口，默认取 RAG_CHROMA_PORT 或 8100")
    parser.add_argument("--batch-size", type=int, default=None, help="每批写入条数，默认 64")
    parser.add_argument("--force", action="store_true", help="忽略 manifest，强制全量重建")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    chunks_dir = resolve_path(args.chunks_dir, "RAG_CHUNKS_DIR", DEFAULT_CHUNKS_DIR)
    manifest_path = resolve_path(args.manifest, "RAG_INDEX_MANIFEST", DEFAULT_MANIFEST)
    collection = args.collection or os.getenv("RAG_COLLECTION") or DEFAULT_COLLECTION
    model = os.getenv("RAG_EMBEDDING_MODEL") or DEFAULT_MODEL
    dimensions = int(os.getenv("RAG_EMBEDDING_DIMENSIONS") or DEFAULT_DIMENSIONS)
    batch_size = args.batch_size or int(
        os.getenv("RAG_EMBEDDING_BATCH_SIZE") or DEFAULT_BATCH_SIZE
    )
    env_host, env_port, use_ssl = chroma_settings()
    host = args.chroma_host or env_host
    port = args.chroma_port or env_port

    print(f"chunk 目录：{chunks_dir}")
    print(f"Chroma 服务：{host}:{port}（collection={collection}）")
    try:
        chunks, source_hashes = load_chunks(chunks_dir)
    except (FileNotFoundError, ValueError) as exc:
        print(f"[FAILED] {exc}")
        return 1

    try:
        client = create_chroma_client(host, port, use_ssl)
    except RuntimeError as exc:
        print(f"[FAILED] {exc}")
        return 1

    chunk_size, chunk_overlap = load_chunking_params(chunks_dir)
    manifest = load_manifest(manifest_path)
    reasons = manifest_stale_reasons(
        manifest,
        model=model,
        dimensions=dimensions,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        collection=collection,
        chunk_count=len(chunks),
        source_hashes=source_hashes,
    )
    if not args.force and not reasons and collection_count(client, collection) == len(chunks):
        print(
            f"[OK] 索引已是最新（{len(chunks)} 条向量，模型 {model}/{dimensions} 维），无需调用 embedding。"
            "如需强制重建请加 --force。"
        )
        return 0
    if reasons:
        print("需要重建，原因：")
        for reason in reasons:
            print(f"  - {reason}")

    try:
        embeddings = ZhipuEmbeddings.from_env()
    except (ZhipuEmbeddingsError, ValueError) as exc:
        print(f"[FAILED] {exc}")
        return 1

    print(f"调用智谱 embedding：模型={embeddings.model} 维度={embeddings.dimensions} 批量={batch_size}")
    try:
        result = build_index(
            chunks,
            embeddings=embeddings,
            client=client,
            collection=collection,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            manifest_path=manifest_path,
            source_hashes=source_hashes,
            host=host,
            port=port,
            batch_size=batch_size,
        )
    except (ZhipuEmbeddingsError, RuntimeError, ValueError) as exc:
        print(f"[FAILED] {exc}")
        return 1
    finally:
        embeddings.close()

    usage = result["manifest"]["token_usage"]
    print(
        f"[OK] 写入 {result['count']} 条向量到 Chroma {host}:{port}（collection={collection}，cosine）"
    )
    print(
        f"     token 用量：prompt={usage['prompt_tokens']} total={usage['total_tokens']} "
        f"请求 {usage['requests']} 次，估算成本 {usage['estimated_cost_cny']} 元"
    )
    print(f"     manifest：{manifest_path}")
    print(
        "可以进入下一步（检索验证）：uv run python scripts/search_docs.py "
        '--query "退货要多久才能退款？"'
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
