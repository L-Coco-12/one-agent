"""智谱 AI（BigModel）embedding 适配器。

把智谱的 ``POST /api/paas/v4/embeddings`` 接口包装成 LangChain 的
:class:`langchain_core.embeddings.Embeddings` 接口，供 Chroma 建索引
（``embed_documents``）和检索（``embed_query``）时调用。

设计要点
--------
1. 只依赖 httpx（项目已有），不引入 langchain-community / zhipuai；
2. 智谱单次请求最多 64 条文本（``embedding-3`` 限制），内部自动分批；
3. 401 表示 Key 无效或欠费，重试没有意义，直接给出人话错误；
   429 / 5xx / 超时属于临时故障，按指数退避重试；
4. 记录累计 token 用量，建索引时写进 manifest，方便估算成本。

环境变量（也可显式传参）
------------------------
- ``ZHIPU_API_KEY``：智谱 API Key（必填）
- ``RAG_EMBEDDING_MODEL``：默认 ``embedding-3``
- ``RAG_EMBEDDING_DIMENSIONS``：默认 ``1024``（可选 256/512/1024/2048）
- ``RAG_EMBEDDING_BATCH_SIZE``：默认 ``64``（不能超过 64）
"""

from __future__ import annotations

import os
import time
from typing import Any

import httpx
from langchain_core.embeddings import Embeddings

DEFAULT_BASE_URL = "https://open.bigmodel.cn/api/paas/v4"
DEFAULT_MODEL = "embedding-3"
DEFAULT_DIMENSIONS = 1024
SUPPORTED_DIMENSIONS = (256, 512, 1024, 2048)
MAX_BATCH_SIZE = 64
DEFAULT_BATCH_SIZE = 64
DEFAULT_TIMEOUT_SECONDS = 30.0
DEFAULT_MAX_RETRIES = 3
RETRY_INITIAL_DELAY = 1.0
RETRY_MAX_DELAY = 8.0
RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}


class ZhipuEmbeddingsError(RuntimeError):
    """智谱 embedding 调用失败（带人话原因）。"""


class ZhipuEmbeddings(Embeddings):
    """把智谱 embedding 接口适配成 LangChain ``Embeddings``。

    用法::

        embeddings = ZhipuEmbeddings.from_env()
        vectors = embeddings.embed_documents(["第一段文本", "第二段文本"])
        query_vector = embeddings.embed_query("退货要多久？")
    """

    def __init__(
        self,
        api_key: str | None = None,
        *,
        model: str | None = None,
        dimensions: int | None = None,
        batch_size: int = DEFAULT_BATCH_SIZE,
        base_url: str = DEFAULT_BASE_URL,
        timeout: float = DEFAULT_TIMEOUT_SECONDS,
        max_retries: int = DEFAULT_MAX_RETRIES,
        client: httpx.Client | None = None,
    ) -> None:
        resolved_key = (api_key or os.getenv("ZHIPU_API_KEY") or "").strip()
        if not resolved_key:
            raise ZhipuEmbeddingsError(
                "缺少智谱 API Key：请在 .env 中设置 ZHIPU_API_KEY（智谱开放平台申请）。"
            )
        if not 1 <= batch_size <= MAX_BATCH_SIZE:
            raise ValueError(f"batch_size 必须在 1–{MAX_BATCH_SIZE} 之间，当前为 {batch_size}。")
        resolved_dimensions = DEFAULT_DIMENSIONS if dimensions is None else int(dimensions)
        if resolved_dimensions not in SUPPORTED_DIMENSIONS:
            raise ValueError(
                f"dimensions 只能取 {SUPPORTED_DIMENSIONS}，当前为 {resolved_dimensions}。"
            )

        self.api_key = resolved_key
        self.model = (model or DEFAULT_MODEL).strip()
        self.dimensions = resolved_dimensions
        self.batch_size = int(batch_size)
        self.base_url = base_url.rstrip("/")
        self.timeout = float(timeout)
        self.max_retries = max(0, int(max_retries))

        # 累计用量，便于建索引后写入 manifest
        self.token_usage: dict[str, int] = {"prompt_tokens": 0, "total_tokens": 0}
        self.request_count = 0
        self._client = client or httpx.Client(timeout=self.timeout)
        self._owns_client = client is None

    # ------------------------------------------------------------------ 构造
    @classmethod
    def from_env(cls) -> "ZhipuEmbeddings":
        """从环境变量读取配置，缺 ZHIPU_API_KEY 时抛 ZhipuEmbeddingsError。"""
        return cls(
            api_key=os.getenv("ZHIPU_API_KEY"),
            model=os.getenv("RAG_EMBEDDING_MODEL") or DEFAULT_MODEL,
            dimensions=int(os.getenv("RAG_EMBEDDING_DIMENSIONS") or DEFAULT_DIMENSIONS),
            batch_size=int(os.getenv("RAG_EMBEDDING_BATCH_SIZE") or DEFAULT_BATCH_SIZE),
            base_url=os.getenv("RAG_EMBEDDING_BASE_URL") or DEFAULT_BASE_URL,
            timeout=float(os.getenv("RAG_EMBEDDING_TIMEOUT") or DEFAULT_TIMEOUT_SECONDS),
            max_retries=int(os.getenv("RAG_EMBEDDING_MAX_RETRIES") or DEFAULT_MAX_RETRIES),
        )

    # -------------------------------------------------------- LangChain 接口
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """把多段文本转成向量（内部按 batch_size 分批）。"""
        return self._embed_texts([str(text) for text in texts])

    def embed_query(self, text: str) -> list[float]:
        """把一条查询转成向量（与文档向量必须同一模型/维度）。"""
        vectors = self._embed_texts([str(text)])
        if not vectors:
            raise ZhipuEmbeddingsError("查询文本为空，无法生成向量。")
        return vectors[0]

    # ------------------------------------------------------------------ 内部
    def _embed_texts(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        vectors: list[list[float]] = []
        for start in range(0, len(texts), self.batch_size):
            batch = texts[start : start + self.batch_size]
            vectors.extend(self._embed_batch(batch))
        return vectors

    def _embed_batch(self, batch: list[str]) -> list[list[float]]:
        payload: dict[str, Any] = {
            "model": self.model,
            "input": batch,
            "dimensions": self.dimensions,
        }
        url = f"{self.base_url}/embeddings"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        last_error = "未知错误"
        for attempt in range(self.max_retries + 1):
            self.request_count += 1
            try:
                response = self._client.post(url, headers=headers, json=payload)
            except httpx.TimeoutException:
                last_error = f"请求超时（{self.timeout:g} 秒）"
            except httpx.HTTPError as exc:
                last_error = f"网络错误：{type(exc).__name__}: {exc}"
            else:
                status = response.status_code
                if status == 200:
                    return self._parse_batch(response.json(), expected=len(batch))
                if status == 401:
                    raise ZhipuEmbeddingsError(
                        "智谱返回 401 Unauthorized：API Key 无效或账户欠费。"
                        "请检查 .env 中的 ZHIPU_API_KEY 与账户余额。"
                    )
                if status in (400, 403, 404):
                    raise ZhipuEmbeddingsError(
                        f"智谱返回 {status}：{self._extract_error(response)}"
                    )
                if status in RETRYABLE_STATUS_CODES:
                    last_error = f"HTTP {status}：{self._extract_error(response)}"
                else:
                    raise ZhipuEmbeddingsError(
                        f"智谱返回 HTTP {status}：{self._extract_error(response)}"
                    )

            if attempt < self.max_retries:
                delay = min(RETRY_INITIAL_DELAY * (2**attempt), RETRY_MAX_DELAY)
                time.sleep(delay)

        raise ZhipuEmbeddingsError(
            f"调用智谱 embedding 失败（已重试 {self.max_retries} 次）：{last_error}"
        )

    def _parse_batch(self, payload: dict[str, Any], *, expected: int) -> list[list[float]]:
        data = payload.get("data")
        if not isinstance(data, list) or len(data) != expected:
            raise ZhipuEmbeddingsError(
                f"智谱返回的向量条数异常：期望 {expected} 条，实际 {len(data) if isinstance(data, list) else '非列表'}。"
            )
        ordered: list[list[float] | None] = [None] * expected
        for item in data:
            try:
                index = int(item["index"])
                embedding = item["embedding"]
            except (KeyError, TypeError, ValueError) as exc:
                raise ZhipuEmbeddingsError(f"智谱返回的数据格式异常：{item!r}") from exc
            if index < 0 or index >= expected:
                raise ZhipuEmbeddingsError(f"智谱返回的 index 越界：{index}")
            if len(embedding) != self.dimensions:
                raise ZhipuEmbeddingsError(
                    f"返回向量维度 {len(embedding)} 与配置 {self.dimensions} 不一致；"
                    "请检查 RAG_EMBEDDING_MODEL / RAG_EMBEDDING_DIMENSIONS。"
                )
            ordered[index] = [float(value) for value in embedding]

        if any(vector is None for vector in ordered):
            raise ZhipuEmbeddingsError("智谱返回的向量存在缺失（index 不连续）。")

        usage = payload.get("usage") or {}
        self.token_usage["prompt_tokens"] += int(usage.get("prompt_tokens") or 0)
        self.token_usage["total_tokens"] += int(usage.get("total_tokens") or 0)
        return [vector for vector in ordered if vector is not None]

    @staticmethod
    def _extract_error(response: httpx.Response) -> str:
        try:
            body = response.json()
        except ValueError:
            return response.text[:300]
        if isinstance(body, dict):
            error = body.get("error")
            if isinstance(error, dict):
                message = error.get("message")
                if message:
                    return str(message)
            if isinstance(body.get("message"), str):
                return body["message"]
        return str(body)[:300]

    def close(self) -> None:
        """关闭底层 HTTP 客户端（非必需，进程退出时也会释放）。"""
        if self._owns_client and not self._client.is_closed:
            self._client.close()


def main() -> int:
    """手动验证用：打印一条查询向量的维度与前 5 个分量。"""
    try:
        embeddings = ZhipuEmbeddings.from_env()
        vector = embeddings.embed_query("退货要多久才能退款？")
    except ZhipuEmbeddingsError as exc:
        print(f"[FAILED] {exc}")
        return 1
    print(f"[OK] 模型={embeddings.model} 维度={len(vector)} 前 5 个分量={vector[:5]}")
    print(f"     token_usage={embeddings.token_usage}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
