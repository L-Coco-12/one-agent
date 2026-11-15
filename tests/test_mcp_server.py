"""scripts/mcp_server.py 的离线测试。

不依赖真实 Chroma / 智谱 API：
- 用 FakeStore 替换 `mcp_server._get_store`（模拟 similarity_search_with_score）；
- 用 FastMCP 的内存客户端走完整 MCP 协议（tools/list、tools/call）。

不引入 pytest-asyncio：每个用例用 asyncio.run 包裹协程。
"""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from fastmcp import Client  # noqa: E402

import mcp_server  # noqa: E402


class FakeDocument:
    def __init__(self, page_content: str, metadata: dict[str, Any] | None = None) -> None:
        self.page_content = page_content
        self.metadata = metadata or {}


class FakeStore:
    """模拟 LangChain Chroma 向量库的 similarity_search_with_score 接口。"""

    def __init__(self, pairs: list[tuple[FakeDocument, float]] | None = None) -> None:
        self.pairs = pairs if pairs is not None else []
        self.calls: list[dict[str, Any]] = []
        self.error: Exception | None = None

    def similarity_search_with_score(
        self, query: str, *, k: int = 4
    ) -> list[tuple[FakeDocument, float]]:
        self.calls.append({"query": query, "k": k})
        if self.error is not None:
            raise self.error
        return self.pairs[:k]


DEFAULT_PAIRS = [
    (
        FakeDocument(
            "退货申请提交后，平台会在 1–3 个工作日内审核。",
            {
                "source": "product/faq.md",
                "section": "退换货政策常见问题 > 退货流程",
            },
        ),
        0.3097,
    ),
    (
        FakeDocument(
            "access_token 有效期为 2 小时，refresh_token 为 30 天。",
            {
                "source": "api/auth.md",
                "section": "API 鉴权说明 > 令牌有效期",
                "page_start": 2,
            },
        ),
        0.4879,
    ),
]


async def _list_tools() -> list[Any]:
    async with Client(mcp_server.mcp) as client:
        return await client.list_tools()


async def _call_tool(name: str, arguments: dict[str, Any]) -> Any:
    async with Client(mcp_server.mcp) as client:
        return await client.call_tool(name, arguments)


def _monkeypatch_store(monkeypatch: Any, store: FakeStore) -> None:
    monkeypatch.setattr(mcp_server, "_get_store", lambda: store)


def test_mcp_server_lists_search_docs_with_schema(monkeypatch: Any) -> None:
    _monkeypatch_store(monkeypatch, FakeStore(DEFAULT_PAIRS))

    tools = asyncio.run(_list_tools())
    by_name = {tool.name: tool for tool in tools}
    assert "search_docs" in by_name

    tool = by_name["search_docs"]
    assert "知识库" in (tool.description or "")
    properties = (tool.inputSchema or {}).get("properties", {})
    assert {"query", "top_k"} <= set(properties)
    assert (tool.inputSchema or {}).get("required") == ["query"]


def test_mcp_server_call_returns_formatted_results(monkeypatch: Any) -> None:
    store = FakeStore(DEFAULT_PAIRS)
    _monkeypatch_store(monkeypatch, store)

    result = asyncio.run(_call_tool("search_docs", {"query": "退货要多久才能退款", "top_k": 2}))
    text = result.content[0].text

    assert result.is_error is False
    assert "[OK] Found 2 relevant chunk(s)" in text
    assert "score=0.6903 source=product/faq.md section=退换货政策常见问题 > 退货流程" in text
    assert "score=0.5121 source=api/auth.md section=API 鉴权说明 > 令牌有效期 page=2" in text
    assert "Cite the source" in text
    assert store.calls == [{"query": "退货要多久才能退款", "k": 2}]


def test_mcp_server_clamps_top_k(monkeypatch: Any) -> None:
    store = FakeStore(DEFAULT_PAIRS)
    _monkeypatch_store(monkeypatch, store)

    asyncio.run(_call_tool("search_docs", {"query": "退货", "top_k": 99}))
    asyncio.run(_call_tool("search_docs", {"query": "退货", "top_k": 0}))

    assert [call["k"] for call in store.calls] == [10, 1]


def test_mcp_server_reports_empty_results(monkeypatch: Any) -> None:
    _monkeypatch_store(monkeypatch, FakeStore([]))

    result = asyncio.run(_call_tool("search_docs", {"query": "不存在的问题"}))
    assert result.content[0].text == "[OK] No relevant documents found in the knowledge base."


def test_mcp_server_returns_failed_text_when_store_is_unavailable(monkeypatch: Any) -> None:
    def broken_store() -> FakeStore:
        raise RuntimeError("chroma is down")

    monkeypatch.setattr(mcp_server, "_get_store", broken_store)

    result = asyncio.run(_call_tool("search_docs", {"query": "退货"}))
    text = result.content[0].text

    assert result.is_error is False  # 工具把错误转成文本，而不是协议错误
    assert text.startswith("[FAILED]")
    assert "docker compose up -d chroma" in text
    assert "ZHIPU_API_KEY" in text


def test_mcp_server_returns_failed_text_when_search_raises(monkeypatch: Any) -> None:
    store = FakeStore(DEFAULT_PAIRS)
    store.error = ValueError("bad query")
    _monkeypatch_store(monkeypatch, store)

    result = asyncio.run(_call_tool("search_docs", {"query": "退货"}))
    text = result.content[0].text

    assert result.is_error is False
    assert text.startswith("[FAILED] Knowledge base search failed")
    assert "bad query" in text


def test_mcp_server_adds_loopback_to_no_proxy(monkeypatch: Any) -> None:
    """客户端常会过滤掉 NO_PROXY，server 必须在连 Chroma 前自己补齐本地回环。"""
    monkeypatch.delenv("NO_PROXY", raising=False)
    monkeypatch.delenv("no_proxy", raising=False)

    mcp_server._ensure_local_proxy_bypass()

    for name in ("NO_PROXY", "no_proxy"):
        parts = os.environ[name].split(",")
        assert "localhost" in parts
        assert "127.0.0.1" in parts
        assert "::1" in parts

