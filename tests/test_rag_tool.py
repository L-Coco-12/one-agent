"""search_docs（RAG 检索工具）测试。

不连 Chroma、不调智谱：用假 store 替换 `_get_rag_store`。
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from langchain_skills import tools as tools_module  # noqa: E402
from langchain_skills.tools import ALL_TOOLS, search_docs  # noqa: E402


class FakeStore:
    def __init__(self, pairs, *, error: Exception | None = None) -> None:
        self.pairs = pairs
        self.error = error
        self.calls: list[tuple[str, int]] = []

    def similarity_search_with_score(self, query: str, k: int):
        self.calls.append((query, k))
        if self.error is not None:
            raise self.error
        return self.pairs[:k]


def make_pairs() -> list:
    return [
        (
            SimpleNamespace(
                page_content="退货流程：签收后 7 天内可申请退货，退款 3–7 个工作日到账。",
                metadata={"source": "product/faq.md", "section": "退换货政策 > 退货流程"},
            ),
            0.25,
        ),
        (
            SimpleNamespace(
                page_content="令牌有效期：访问令牌 2 小时，刷新令牌 30 天。",
                metadata={"source": "api/auth.md", "section": "令牌有效期", "page_start": 2},
            ),
            0.40,
        ),
    ]


def test_search_docs_is_registered():
    names = {tool.name for tool in ALL_TOOLS}
    assert "search_docs" in names
    assert "knowledge base" in (search_docs.description or "").lower()


def test_search_docs_formats_results_with_source(monkeypatch):
    store = FakeStore(make_pairs())
    monkeypatch.setattr(tools_module, "_get_rag_store", lambda: store)

    result = search_docs.func("退货要多久才能退款？", None)

    assert result.startswith("[OK] Found 2 relevant chunk(s)")
    assert "score=0.7500" in result  # 1 - 0.25
    assert "source=product/faq.md" in result
    assert "section=退换货政策 > 退货流程" in result
    assert "source=api/auth.md" in result
    assert "page=2" in result
    assert "Cite the source" in result
    assert store.calls == [("退货要多久才能退款？", 4)]


def test_search_docs_empty_result(monkeypatch):
    monkeypatch.setattr(tools_module, "_get_rag_store", lambda: FakeStore([]))
    result = search_docs.func("知识库里没有的问题", None)
    assert result == "[OK] No relevant documents found in the knowledge base."


def test_search_docs_clamps_top_k(monkeypatch):
    store = FakeStore(make_pairs())
    monkeypatch.setattr(tools_module, "_get_rag_store", lambda: store)

    search_docs.func("q", None, top_k=99)
    search_docs.func("q", None, top_k=0)
    search_docs.func("q", None, top_k=-3)

    assert [call[1] for call in store.calls] == [10, 1, 1]


def test_search_docs_reports_connection_failure(monkeypatch):
    def broken_store():
        raise RuntimeError("无法连接 Chroma 服务（测试）")

    monkeypatch.setattr(tools_module, "_get_rag_store", broken_store)
    result = search_docs.func("q", None)

    assert result.startswith("[FAILED] Knowledge base unavailable")
    assert "docker compose up -d chroma" in result
    assert "ZHIPU_API_KEY" in result


def test_search_docs_reports_query_failure(monkeypatch):
    store = FakeStore([], error=RuntimeError("collection 不存在"))
    monkeypatch.setattr(tools_module, "_get_rag_store", lambda: store)
    result = search_docs.func("q", None)
    assert result.startswith("[FAILED] Knowledge base search failed")
    assert "collection 不存在" in result
