"""智谱 embedding 适配器测试。

全部离线：用 ``httpx.MockTransport`` 模拟智谱接口，不联网、不花钱。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import httpx
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import rag_embeddings  # noqa: E402
from rag_embeddings import ZhipuEmbeddings, ZhipuEmbeddingsError  # noqa: E402

DIMENSIONS = 256


def make_embeddings(handler, *, max_retries: int = 3, batch_size: int = 64) -> ZhipuEmbeddings:
    client = httpx.Client(transport=httpx.MockTransport(handler))
    return ZhipuEmbeddings(
        api_key="test-key",
        dimensions=DIMENSIONS,
        batch_size=batch_size,
        max_retries=max_retries,
        client=client,
    )


def ok_response(size: int) -> dict:
    return {
        "data": [{"index": i, "embedding": [float(i + 1)] * DIMENSIONS} for i in range(size)],
        "usage": {"prompt_tokens": size, "total_tokens": size},
    }


def test_embed_documents_batches_at_64():
    calls: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        calls.append(payload)
        return httpx.Response(200, json=ok_response(len(payload["input"])))

    embeddings = make_embeddings(handler)
    texts = [f"文本{i}" for i in range(65)]
    vectors = embeddings.embed_documents(texts)

    assert len(vectors) == 65
    assert [len(call["input"]) for call in calls] == [64, 1]
    assert calls[0]["model"] == "embedding-3"
    assert calls[0]["dimensions"] == DIMENSIONS
    assert embeddings.request_count == 2
    assert embeddings.token_usage["prompt_tokens"] == 65
    # 每个 batch 内 index 从 0 开始，返回顺序与输入顺序一致
    assert vectors[0][0] == 1.0
    assert vectors[63][0] == 64.0
    assert vectors[64][0] == 1.0


def test_embed_query_returns_single_vector():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=ok_response(1))

    vector = make_embeddings(handler).embed_query("退货要多久？")
    assert isinstance(vector, list)
    assert len(vector) == DIMENSIONS


def test_empty_input_returns_empty_list():
    def handler(request: httpx.Request) -> httpx.Response:  # pragma: no cover
        raise AssertionError("空输入不应发请求")

    assert make_embeddings(handler).embed_documents([]) == []


def test_401_raises_without_retry():
    calls: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(1)
        return httpx.Response(401, json={"error": {"message": "bad key"}})

    with pytest.raises(ZhipuEmbeddingsError) as exc_info:
        make_embeddings(handler).embed_query("x")
    message = str(exc_info.value)
    assert "401" in message
    assert "ZHIPU_API_KEY" in message
    assert len(calls) == 1


def test_429_retries_then_succeeds(monkeypatch):
    monkeypatch.setattr(rag_embeddings.time, "sleep", lambda _seconds: None)
    calls: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(1)
        if len(calls) == 1:
            return httpx.Response(429, json={"error": {"message": "rate limited"}})
        return httpx.Response(200, json=ok_response(1))

    vector = make_embeddings(handler, max_retries=2).embed_query("x")
    assert len(vector) == DIMENSIONS
    assert len(calls) == 2


def test_retries_exhausted_raises(monkeypatch):
    monkeypatch.setattr(rag_embeddings.time, "sleep", lambda _seconds: None)
    calls: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(1)
        return httpx.Response(500, json={"error": {"message": "boom"}})

    with pytest.raises(ZhipuEmbeddingsError) as exc_info:
        make_embeddings(handler, max_retries=2).embed_query("x")
    assert "重试 2 次" in str(exc_info.value)
    assert len(calls) == 3


def test_dimension_mismatch_raises():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200, json={"data": [{"index": 0, "embedding": [0.1] * 8}], "usage": {}}
        )

    with pytest.raises(ZhipuEmbeddingsError) as exc_info:
        make_embeddings(handler).embed_query("x")
    assert "维度" in str(exc_info.value)


def test_missing_api_key_raises(monkeypatch):
    monkeypatch.delenv("ZHIPU_API_KEY", raising=False)
    with pytest.raises(ZhipuEmbeddingsError) as exc_info:
        ZhipuEmbeddings(api_key=None)
    assert "ZHIPU_API_KEY" in str(exc_info.value)


def test_invalid_batch_size_and_dimensions_rejected():
    with pytest.raises(ValueError):
        ZhipuEmbeddings(api_key="k", batch_size=65)
    with pytest.raises(ValueError):
        ZhipuEmbeddings(api_key="k", dimensions=100)


def test_from_env_reads_config(monkeypatch):
    monkeypatch.setenv("ZHIPU_API_KEY", "env-key")
    monkeypatch.setenv("RAG_EMBEDDING_MODEL", "embedding-3")
    monkeypatch.setenv("RAG_EMBEDDING_DIMENSIONS", "512")
    monkeypatch.setenv("RAG_EMBEDDING_BATCH_SIZE", "32")
    embeddings = ZhipuEmbeddings.from_env()
    assert embeddings.api_key == "env-key"
    assert embeddings.dimensions == 512
    assert embeddings.batch_size == 32
