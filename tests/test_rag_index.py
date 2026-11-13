"""RAG 索引构建 / 检索 / 跳过逻辑测试。

全部离线：用内存版 Chroma（``EphemeralClient``）+ 假 embedding，不联网、不花钱。
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import chromadb
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import build_rag_index as idxmod  # noqa: E402
from build_rag_index import (  # noqa: E402
    build_index,
    collection_count,
    load_chunks,
    main,
    manifest_stale_reasons,
)

COLLECTION = "test-knowledge"
CHUNKS = [
    {
        "id": "faq-c000",
        "text": "退货流程：签收后 7 天内可申请退货，商品需保持完好。",
        "source": "product/faq.md",
        "title": "退换货政策常见问题",
        "chunk_index": 0,
        "section": "退货流程",
        "page_start": None,
        "metadata": {"title": "退换货政策常见问题", "owner": "客服团队", "tags": ["售后", "退货"]},
    },
    {
        "id": "faq-c001",
        "text": "退款到账时间：审核通过后 3-5 个工作日原路退回。",
        "source": "product/faq.md",
        "title": "退换货政策常见问题",
        "chunk_index": 1,
        "section": "退款到账",
        "page_start": None,
        "metadata": {"title": "退换货政策常见问题", "owner": "客服团队", "tags": ["售后"]},
    },
    {
        "id": "auth-c000",
        "text": "令牌有效期：访问令牌 2 小时后过期，刷新令牌 30 天。",
        "source": "api/auth.md",
        "title": "API 鉴权说明",
        "chunk_index": 0,
        "section": "令牌有效期",
        "page_start": None,
        "metadata": {"title": "API 鉴权说明", "owner": "平台组", "tags": ["鉴权"]},
    },
    {
        "id": "auth-c001",
        "text": "错误码 40102 表示令牌已过期，需要重新获取。",
        "source": "api/auth.md",
        "title": "API 鉴权说明",
        "chunk_index": 1,
        "section": "常见错误码",
        "page_start": None,
        "metadata": {"title": "API 鉴权说明", "owner": "平台组", "tags": ["鉴权"]},
    },
]


class FakeEmbeddings:
    """确定性假 embedding（sha256 前 8 字节），实现 LangChain Embeddings 接口。"""

    model = "fake-embedding"
    dimensions = 8
    batch_size = 64

    def __init__(self) -> None:
        self.token_usage = {"prompt_tokens": 0, "total_tokens": 0}
        self.request_count = 0

    def _vector(self, text: str) -> list[float]:
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        values = [byte / 255.0 for byte in digest[: self.dimensions]]
        norm = sum(value * value for value in values) ** 0.5 or 1.0
        self.request_count += 1
        return [value / norm for value in values]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._vector(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vector(text)

    def close(self) -> None:  # 与 ZhipuEmbeddings 接口保持一致
        pass


def write_chunks(chunks_dir: Path) -> None:
    chunks_dir.mkdir(parents=True, exist_ok=True)
    by_source: dict[str, list[dict]] = {}
    for chunk in CHUNKS:
        by_source.setdefault(chunk["source"], []).append(chunk)
    for source, rows in by_source.items():
        file_path = chunks_dir / (source.replace("/", "__") + ".jsonl")
        lines = [json.dumps(row, ensure_ascii=False) for row in rows]
        file_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def sha256_of(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture()
def chunks_fixture(tmp_path: Path) -> tuple[Path, dict[str, str]]:
    chunks_dir = tmp_path / ".rag" / "chunks"
    write_chunks(chunks_dir)
    hashes = {
        "product/faq.md": sha256_of(chunks_dir / "product__faq.md.jsonl"),
        "api/auth.md": sha256_of(chunks_dir / "api__auth.md.jsonl"),
    }
    return chunks_dir, hashes


def test_load_chunks_collects_rows_and_hashes(chunks_fixture):
    chunks_dir, hashes = chunks_fixture
    chunks, computed = load_chunks(chunks_dir)
    assert len(chunks) == 4
    assert computed == hashes
    assert all(len(digest) == 64 for digest in computed.values())


def test_load_chunks_rejects_invalid_json(tmp_path: Path):
    chunks_dir = tmp_path / "chunks"
    chunks_dir.mkdir()
    (chunks_dir / "bad.jsonl").write_text("{not-json}\n", encoding="utf-8")
    with pytest.raises(ValueError):
        load_chunks(chunks_dir)


def test_to_document_flattens_list_metadata():
    document = idxmod.to_document(CHUNKS[0])
    assert document.metadata["source"] == "product/faq.md"
    assert document.metadata["section"] == "退货流程"
    assert document.metadata["tags"] == "售后, 退货"
    assert document.metadata["chunk_index"] == 0


def test_build_index_writes_vectors_and_manifest(tmp_path, chunks_fixture):
    chunks_dir, hashes = chunks_fixture
    chunks, _ = load_chunks(chunks_dir)
    client = chromadb.EphemeralClient()
    manifest_path = tmp_path / "index_manifest.json"

    result = build_index(
        chunks,
        embeddings=FakeEmbeddings(),
        client=client,
        collection=COLLECTION,
        chunk_size=500,
        chunk_overlap=100,
        manifest_path=manifest_path,
        source_hashes=hashes,
    )

    assert result["count"] == 4
    assert collection_count(client, COLLECTION) == 4
    collection = client.get_collection(COLLECTION)
    assert collection.metadata["hnsw:space"] == "cosine"

    stored = collection.get(ids=["faq-c000"], include=["metadatas", "documents"])
    assert stored["metadatas"][0]["source"] == "product/faq.md"
    assert stored["documents"][0] == CHUNKS[0]["text"]

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["schema_version"] == 1
    assert manifest["embedding"]["model"] == "fake-embedding"
    assert manifest["embedding"]["dimensions"] == 8
    assert manifest["chunking"] == {"chunk_size": 500, "chunk_overlap": 100}
    assert manifest["chunks"]["count"] == 4
    assert {item["source"] for item in manifest["chunks"]["sources"]} == {
        "product/faq.md",
        "api/auth.md",
    }


def test_build_index_replaces_old_collection(tmp_path, chunks_fixture):
    chunks_dir, hashes = chunks_fixture
    chunks, _ = load_chunks(chunks_dir)
    client = chromadb.EphemeralClient()
    manifest_path = tmp_path / "index_manifest.json"

    for _ in range(2):
        build_index(
            chunks,
            embeddings=FakeEmbeddings(),
            client=client,
            collection=COLLECTION,
            chunk_size=500,
            chunk_overlap=100,
            manifest_path=manifest_path,
            source_hashes=hashes,
        )
    assert collection_count(client, COLLECTION) == 4  # 重建而不是叠加


def test_search_finds_identical_chunk(tmp_path, chunks_fixture):
    from langchain_chroma import Chroma

    chunks_dir, hashes = chunks_fixture
    chunks, _ = load_chunks(chunks_dir)
    client = chromadb.EphemeralClient()
    embeddings = FakeEmbeddings()
    build_index(
        chunks,
        embeddings=embeddings,
        client=client,
        collection=COLLECTION,
        chunk_size=500,
        chunk_overlap=100,
        manifest_path=tmp_path / "index_manifest.json",
        source_hashes=hashes,
    )
    store = Chroma(
        client=client,
        collection_name=COLLECTION,
        embedding_function=embeddings,
    )
    results = store.similarity_search_with_score(CHUNKS[2]["text"], k=2)
    assert results[0][0].page_content == CHUNKS[2]["text"]
    assert results[0][1] < 0.01  # cosine 距离接近 0


def test_manifest_stale_reasons_unit():
    sources = {"product/faq.md": "a" * 64, "api/auth.md": "b" * 64}
    manifest = {
        "schema_version": 1,
        "embedding": {"model": "embedding-3", "dimensions": 1024},
        "chunking": {"chunk_size": 500, "chunk_overlap": 100},
        "collection": {"name": "knowledge"},
        "chunks": {
            "count": 8,
            "sources": [
                {"source": "product/faq.md", "sha256": "a" * 64, "chunks": 4},
                {"source": "api/auth.md", "sha256": "b" * 64, "chunks": 4},
            ],
        },
    }
    kwargs = dict(
        model="embedding-3",
        dimensions=1024,
        chunk_size=500,
        chunk_overlap=100,
        collection="knowledge",
        chunk_count=8,
        source_hashes=sources,
    )
    assert manifest_stale_reasons(manifest, **kwargs) == []
    assert manifest_stale_reasons(None, **kwargs) == ["manifest 不存在"]

    assert any("模型" in reason for reason in manifest_stale_reasons(manifest, **{**kwargs, "model": "other"}))
    assert any("维度" in reason for reason in manifest_stale_reasons(manifest, **{**kwargs, "dimensions": 512}))
    assert any("chunk_size" in reason for reason in manifest_stale_reasons(manifest, **{**kwargs, "chunk_size": 300}))
    changed = {**sources, "api/auth.md": "c" * 64}
    assert any("api/auth.md" in reason for reason in manifest_stale_reasons(manifest, **{**kwargs, "source_hashes": changed}))


def test_main_skips_when_index_is_fresh(tmp_path, chunks_fixture, monkeypatch, capsys):
    chunks_dir, _ = chunks_fixture
    client = chromadb.EphemeralClient()
    manifest_path = tmp_path / "index_manifest.json"

    class FakeFactory:
        calls = 0

        @classmethod
        def from_env(cls):
            FakeFactory.calls += 1
            return FakeEmbeddings()

    # main 的“是否需要重建”用的是环境变量里的模型/维度，必须与假 embedding 对齐
    monkeypatch.setenv("RAG_EMBEDDING_MODEL", FakeEmbeddings.model)
    monkeypatch.setenv("RAG_EMBEDDING_DIMENSIONS", str(FakeEmbeddings.dimensions))
    monkeypatch.setattr(idxmod, "create_chroma_client", lambda *args, **kwargs: client)
    monkeypatch.setattr(idxmod, "ZhipuEmbeddings", FakeFactory)

    argv = [
        "--chunks-dir",
        str(chunks_dir),
        "--manifest",
        str(manifest_path),
        "--collection",
        COLLECTION,
        "--chroma-host",
        "test-host",
        "--chroma-port",
        "1234",
    ]
    assert main(argv) == 0
    first_output = capsys.readouterr().out
    assert "写入 4 条向量" in first_output
    assert FakeFactory.calls == 1

    # 第二次：参数与 chunk hash 都没变 → 跳过，不再调用 embedding
    assert main(argv) == 0
    second_output = capsys.readouterr().out
    assert "索引已是最新" in second_output
    assert FakeFactory.calls == 1

    # 强制重建 → 重新调用 embedding
    assert main(argv + ["--force"]) == 0
    assert "写入 4 条向量" in capsys.readouterr().out
    assert FakeFactory.calls == 2


def test_main_reports_chroma_connection_failure(chunks_fixture, monkeypatch, capsys):
    chunks_dir, _ = chunks_fixture

    def broken_client(*args, **kwargs):
        raise RuntimeError("无法连接 Chroma 服务（测试）")

    monkeypatch.setattr(idxmod, "create_chroma_client", broken_client)
    code = main(["--chunks-dir", str(chunks_dir), "--collection", COLLECTION])
    assert code == 1
    assert "无法连接 Chroma 服务" in capsys.readouterr().out
