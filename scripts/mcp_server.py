"""把知识库检索（search_docs）暴露为 MCP server。

外部 MCP 客户端（Claude Code / Cursor / Codex 等）通过 stdio 或
Streamable HTTP 调用本 server，即可检索 knowledge/ 下的知识库文档。

用法：

    # stdio 模式（由 MCP 客户端作为子进程拉起；stdout 只能走协议，日志全部写 stderr）
    uv run python scripts/mcp_server.py

    # Streamable HTTP 模式（常驻服务，默认 127.0.0.1:8101/mcp）
    uv run python scripts/mcp_server.py --http --port 8101

这是“适配壳”：业务逻辑全部复用 scripts/search_docs.py 与 scripts/build_rag_index.py，
本文件只负责把 Python 函数翻译成 MCP 工具（签名 → JSON Schema，返回值 → 文本），
并处理进程边界上的细节（.env 加载、惰性建连接、错误转成可读文本、日志走 stderr）。
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from dotenv import load_dotenv

# 客户端拉起子进程时环境变量通常不完整（尤其没有 ZHIPU_API_KEY），
# 因此在导入检索模块前先读取项目根目录的 .env。
load_dotenv(PROJECT_ROOT / ".env")

from fastmcp import FastMCP  # noqa: E402

from build_rag_index import (  # noqa: E402
    DEFAULT_COLLECTION,
    chroma_settings,
    create_chroma_client,
)
from rag_embeddings import ZhipuEmbeddings  # noqa: E402
from search_docs import open_vector_store  # noqa: E402

# 预热重型依赖：实测在 FastMCP 的工作线程里首次 import chromadb 会长时间阻塞
# （本机 >90s，疑似其 Rust 扩展初始化与事件循环/线程交互所致），放到主线程启动时完成。
try:
    import chromadb  # noqa: F401,E402
    from langchain_chroma import Chroma  # noqa: F401,E402
except Exception as exc:  # noqa: BLE001 - 预热失败不阻止启动，真正调用时会再次报错
    print(f"[mcp_server] 预热依赖失败：{exc}", file=sys.stderr, flush=True)

DEFAULT_TOP_K = 4
MAX_TOP_K = 10
PREVIEW_CHARS = 500
DEFAULT_HTTP_HOST = "127.0.0.1"
DEFAULT_HTTP_PORT = 8101
DEFAULT_HTTP_PATH = "/mcp"

INSTRUCTIONS = """本 server 提供 knowledge/ 目录下的知识库检索能力。

使用建议：
- 当用户询问政策、退款、令牌、错误码，或 FastAPI/Kubernetes 等知识库已收录的主题时，
  调用 search_docs 检索，再基于返回片段回答；
- 回答时请引用返回的 source（以及可用的 section/page）；
- 检索结果为空或相关度低时，明确说明“知识库未覆盖”，不要凭空猜测。
"""

# 连接与向量库是惰性创建的：第一次调用工具时才连接 Chroma / 初始化 embedding 客户端。
_store: Any = None

mcp = FastMCP("skills-kb", instructions=INSTRUCTIONS)


def _log(message: str) -> None:
    """调试日志一律写 stderr（stdio 模式下 stdout 只能走协议）。"""
    if os.getenv("MCP_DEBUG", "").strip().lower() in {"1", "true", "yes", "on"}:
        print(f"[mcp_server] {message}", file=sys.stderr, flush=True)


def _error(message: str) -> None:
    """真正的错误信息总是写 stderr，方便客户端排查。"""
    print(f"[mcp_server] ERROR: {message}", file=sys.stderr, flush=True)


def _ensure_local_proxy_bypass() -> None:
    """为本地回环补上 NO_PROXY。

    很多 MCP 客户端拉起子进程时只传最小环境变量（HTTP_PROXY/NO_PROXY 会丢），
    而 httpx 在 Windows 上会回退到系统代理，访问 127.0.0.1:8100 时报 502。
    这里在连接 Chroma 前显式补上本地地址，避免“客户端环境不同导致连不上”。
    """
    required = ["localhost", "127.0.0.1", "::1"]
    for name in ("NO_PROXY", "no_proxy"):
        parts = [part.strip() for part in os.environ.get(name, "").split(",") if part.strip()]
        missing = [host for host in required if host not in parts]
        if missing:
            os.environ[name] = ",".join(parts + missing)


def _get_store() -> Any:
    """惰性创建 LangChain Chroma 向量库对象（同一进程内复用）。"""
    global _store
    if _store is None:
        _ensure_local_proxy_bypass()
        host, port, use_ssl = chroma_settings()
        _log(f"连接 Chroma {host}:{port}（ssl={use_ssl}）")
        client = create_chroma_client(host, port, use_ssl)
        _log("Chroma 连接成功，初始化 embedding 客户端")
        embeddings = ZhipuEmbeddings.from_env()
        _log("embedding 客户端就绪，打开 collection")
        _store = open_vector_store(client, DEFAULT_COLLECTION, embeddings)
        _log("collection 就绪")
    return _store


def _clamp_top_k(top_k: int) -> int:
    try:
        value = int(top_k)
    except (TypeError, ValueError):
        value = DEFAULT_TOP_K
    return max(1, min(MAX_TOP_K, value))


def _format_pairs(pairs: list[tuple[Any, float]]) -> str:
    """把 (Document, distance) 列表格式化成模型友好的文本。

    输出与 agent 内置的 search_docs 工具保持一致：score = 1 - cosine 距离，越大越相似。
    """
    if not pairs:
        return "[OK] No relevant documents found in the knowledge base."

    lines = [f"[OK] Found {len(pairs)} relevant chunk(s) (higher score = more similar):"]
    for index, (document, distance) in enumerate(pairs, start=1):
        metadata = getattr(document, "metadata", None) or {}
        text = (getattr(document, "page_content", "") or "").strip()
        if len(text) > PREVIEW_CHARS:
            text = text[:PREVIEW_CHARS] + "…"
        location = f"source={metadata.get('source') or 'unknown'}"
        section = metadata.get("section") or ""
        if section:
            location += f" section={section}"
        page = metadata.get("page_start")
        if page is not None:
            location += f" page={page}"
        lines.append(f"[{index}] score={1.0 - float(distance):.4f} {location}")
        lines.append(f"    {text}")
    lines.append("Cite the source (and section/page when available) in your answer.")
    return "\n".join(lines)


@mcp.tool
def search_docs(query: str, top_k: int = DEFAULT_TOP_K) -> str:
    """在知识库中检索与问题相关的文档片段。

    何时使用：用户询问政策、退款、令牌、错误码，或 FastAPI/Kubernetes 等
    知识库已收录的主题时，优先调用本工具，再基于返回片段回答。

    何时不用：与知识库无关的寒暄、代码操作类任务。

    参数：
        query: 检索语句（用完整问句比关键词更好）。
        top_k: 返回条数，默认 4，范围 1–10。

    返回：形如 “[OK] Found N relevant chunk(s)...” 的文本，含 score/source/section；
    回答时请引用 source。知识库不可用或检索失败时返回 “[FAILED] ...” 说明原因。
    """
    clamped = _clamp_top_k(top_k)

    try:
        store = _get_store()
    except Exception as exc:  # noqa: BLE001 - 连接类错误统一转成可读文本
        message = (
            f"[FAILED] Knowledge base unavailable: {type(exc).__name__}: {exc}. "
            "Start Chroma first (docker compose up -d chroma), build the index "
            "(uv run python scripts/build_rag_index.py), and make sure "
            "ZHIPU_API_KEY is set (in the project .env or the client's env)."
        )
        _error(message)
        return message

    try:
        _log(f"search_docs 查询：{query!r}（top_k={clamped}）")
        pairs = store.similarity_search_with_score(query, k=clamped)
        _log(f"search_docs 返回 {len(pairs)} 条结果")
    except Exception as exc:  # noqa: BLE001 - 检索失败同样返回文本，让模型知道原因
        message = f"[FAILED] Knowledge base search failed: {type(exc).__name__}: {exc}"
        _error(message)
        return message

    return _format_pairs(pairs)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="skills-kb-mcp",
        description="把知识库检索（search_docs）暴露为 MCP server。",
    )
    parser.add_argument(
        "--http",
        action="store_true",
        help="以 Streamable HTTP 方式运行（默认 stdio）",
    )
    parser.add_argument(
        "--host",
        default=os.getenv("MCP_HTTP_HOST", DEFAULT_HTTP_HOST),
        help=f"HTTP 监听地址（默认 {DEFAULT_HTTP_HOST}，可用 MCP_HTTP_HOST 覆盖）",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.getenv("MCP_HTTP_PORT", str(DEFAULT_HTTP_PORT))),
        help=f"HTTP 监听端口（默认 {DEFAULT_HTTP_PORT}，可用 MCP_HTTP_PORT 覆盖）",
    )
    parser.add_argument(
        "--path",
        default=os.getenv("MCP_HTTP_PATH", DEFAULT_HTTP_PATH),
        help=f"HTTP 挂载路径（默认 {DEFAULT_HTTP_PATH}，可用 MCP_HTTP_PATH 覆盖）",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="把调试日志输出到 stderr（等价于 MCP_DEBUG=1）",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.debug:
        os.environ["MCP_DEBUG"] = "1"

    if args.http:
        _log(f"以 Streamable HTTP 启动：http://{args.host}:{args.port}{args.path}")
        mcp.run(
            transport="http",
            host=args.host,
            port=args.port,
            path=args.path,
            show_banner=False,
        )
    else:
        _log("以 stdio 启动，等待 MCP 客户端连接")
        mcp.run(transport="stdio", show_banner=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
