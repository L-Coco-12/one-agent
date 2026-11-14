"""
LangChain Skills Agent 主体

使用 LangChain 1.0 的 create_agent API 实现 Skills Agent，演示三层加载机制：
- Level 1: 启动时将 Skills 元数据注入 system_prompt
- Level 2: load_skill tool 加载详细指令
- Level 3: bash tool 执行脚本

与 claude-agent-sdk 实现的对比：
- claude-agent-sdk: setting_sources=["user", "project"] 自动处理
- LangChain 实现: 显式调用 SkillLoader，过程透明可见

流式输出支持：
- 支持 Extended Thinking 显示模型思考过程
- 事件级流式输出 (thinking / text / tool_call / tool_result)

中间件（middleware）：
- 可靠性：ModelRetry / ToolError / ToolRetry，失败可重试、异常转 ToolMessage
- 上下文：Summarization，历史超过阈值时自动摘要压缩
- 安全：HumanInTheLoop，bash / write_file 执行前需人工 approve / edit / reject

意图路由：
- 规则快判 + 小模型分类，只对"任务型"请求启用 AgentReply 结构化输出
- 聊天型请求由 @wrap_model_call 关闭 response_format，保持自然对话
"""

import os
import re
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Optional, Iterator

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.agents.middleware import (
    HumanInTheLoopMiddleware,
    ModelRetryMiddleware,
    SummarizationMiddleware,
    ToolErrorMiddleware,
    ToolRetryMiddleware,
    wrap_model_call,
)
from langchain.chat_models import init_chat_model
from langchain_core.messages import AIMessage, AIMessageChunk
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.store.memory import InMemoryStore
from langgraph.types import Command
from pydantic import BaseModel, Field

from .skill_loader import SkillLoader
from .tools import ALL_TOOLS, SkillAgentContext
from .stream import StreamEventEmitter, ToolCallTracker, is_success, DisplayLimits


# 加载环境变量（override=True 确保 .env 文件覆盖系统环境变量）
load_dotenv(override=True)


# 默认配置
DEFAULT_PROVIDER = "deepseek"
DEFAULT_ANTHROPIC_MODEL = "claude-sonnet-4-5-20250929"
DEFAULT_OPENAI_MODEL = "gpt-5-mini"
DEFAULT_DEEPSEEK_MODEL = "deepseek-flash"
DEFAULT_DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEFAULT_MAX_TOKENS = 16000
DEFAULT_TEMPERATURE = 1.0  # Extended Thinking 要求温度为 1.0
DEFAULT_THINKING_BUDGET = 10000
DEFAULT_OPENAI_REASONING_EFFORT = "medium"
DEFAULT_DEEPSEEK_REASONING_EFFORT = "high"

# Middleware 默认配置
DEFAULT_SUMMARY_TRIGGER_TOKENS = 8000  # 历史超过该 token 数触发摘要
DEFAULT_SUMMARY_KEEP_MESSAGES = 20  # 摘要后保留最近 N 条消息
DEFAULT_SUMMARY_MAX_TOKENS = 2000  # 摘要模型的最大输出
DEFAULT_SUMMARY_TEMPERATURE = 0.2  # 摘要要稳定，温度调低
DEFAULT_MAX_RETRIES = 2  # 模型/工具重试次数

# 需要人工确认的工具（HITL）
DEFAULT_HITL_INTERRUPT_ON = {
    "bash": {
        "allowed_decisions": ["approve", "edit", "reject"],
        "description": "执行 shell 命令（可能读写文件或访问网络）",
    },
    "write_file": {
        "allowed_decisions": ["approve", "edit", "reject"],
        "description": "写入文件（会覆盖磁盘内容）",
    },
}

# 意图路由默认配置
DEFAULT_INTENT_TEMPERATURE = 0.0  # 分类要稳定
DEFAULT_INTENT_MAX_TOKENS = 300  # 分类输出很短：is_task + reason


class AgentReply(BaseModel):
    """任务完成后的结构化交付报告。

    当本轮被判定为任务型时，agent 必须在最后调用本工具交卷，
    调用会被框架拦截（不会真的执行），结果写入 structured_response。
    """

    summary: str = Field(
        description=(
            "直接给用户的最终答复（不要描述你做了什么）；"
            "有知识库来源时，把来源写进 sources 字段"
        )
    )
    sources: list[str] = Field(
        default_factory=list,
        description=(
            "回答依据的知识库来源，如 product/faq.md（search_docs 返回的 source）；"
            "没有引用任何来源就留空"
        ),
    )
    needs_followup: bool = Field(
        default=False, description="是否还需要用户补充信息或确认"
    )
    followup_reason: str | None = Field(
        default=None, description="需要继续追问时的原因；不需要则留空"
    )


class Intent(BaseModel):
    """判断用户消息属于任务型还是聊天型。"""

    is_task: bool = Field(
        description="任务型=True（需要改文件/跑命令/联网查真实数据/多步操作）；聊天型=False"
    )
    reason: str = Field(description="一句话说明判断理由")


INTENT_CLASSIFIER_PROMPT = (
    "你是意图分类器，只做分类，不要执行任何操作。\n"
    "判断用户消息属于哪一类：\n"
    "- 任务型：需要读写文件、执行命令、访问网络、查询真实数据、生成或修改产物，"
    "或包含明确的多步操作。\n"
    "- 聊天型：寒暄、概念解释、知识问答、闲聊、询问身份。"
)

# 关键词快判：命中则不必再调用小模型
# 聊天/知识问句：疑问词优先，避免"提交/多久"这类词被误判成任务型
_CHAT_PATTERN = re.compile(
    r"(什么是|是什么|为什么|怎么理解|怎么用|如何使用|解释一下|讲解一下|介绍一下|"
    r"有什么区别|区别是什么|你好|您好|谢谢|再见|辛苦了|"
    r"在哪里|哪里|在哪|多久|多长时间|多少|几天|几个|什么意思|什么含义|"
    r"如何|是否|能不能|可不可以|会不会)"
)
# 明确的"请帮我做"句式优先判为任务型
_EXPLICIT_TASK_PATTERN = re.compile(
    r"(帮我|请你|给我|麻烦你?).{0,20}(改|写|创建|新建|删除|运行|执行|安装|修复|生成|跑)"
)
# 明确指定使用某个工具（"用 list_dir 工具列出..."）也是任务型
_TOOL_USE_PATTERN = re.compile(
    r"(用|使用|调用|通过)\s*[\w\u4e00-\u9fff]+\s*(工具|tool)", re.IGNORECASE
)
# 真正的动作动词；"提交/读取/列出/查找/搜索/统计"这类容易出现在知识问句里的词
# 不再单独作为任务判据，交给小模型兜底判断。
_TASK_PATTERN = re.compile(
    r"(创建|新建|修改|删除|重命名|写入|保存|运行|执行|跑|安装|部署|下载|"
    r"修复|改一下|帮我改|生成文件|写个|写一个|检查|清理)"
)

# 知识库兜底：判断回答里是否引用了文档来源、是否已经承认"知识库未覆盖"
_DOC_CITATION_PATTERN = re.compile(
    r"[A-Za-z0-9_\-./]+\.(?:md|markdown|txt|pdf)", re.IGNORECASE
)
_KNOWLEDGE_GAP_PATTERN = re.compile(
    r"(未覆盖|没有覆盖|未提及|没有提及|没有相关内容|没有找到相关|未找到相关|"
    r"无法从知识库|无法回答|知识库中没有|知识库没有|知识库未|无法确定|不清楚|不知道)"
)
_KNOWLEDGE_GAP_SUFFIX = (
    "\n\n（注：以上回答未能引用知识库来源，知识库可能未覆盖该问题，请以官方渠道为准。）"
)


def _classify_by_rules(text: str) -> Optional[bool]:
    """
    用关键词快速判断意图

    Returns:
        True=任务型，False=聊天型，None=规则拿不准（交给小模型分类）
    """
    normalized = text.strip()
    if not normalized:
        return False
    if _EXPLICIT_TASK_PATTERN.search(normalized):
        return True
    if _TOOL_USE_PATTERN.search(normalized):
        return True
    if _CHAT_PATTERN.search(normalized):
        return False
    if _TASK_PATTERN.search(normalized):
        return True
    return None


@wrap_model_call
def _route_by_intent(request, handler):
    """
    按本轮意图决定是否启用结构化输出

    任务型（task_mode=True）：保留 create_agent 声明的 response_format=AgentReply；
    聊天型：把 response_format 覆盖为 None，模型正常自由回答，不被 schema 约束。
    """
    runtime = getattr(request, "runtime", None)
    context = getattr(runtime, "context", None)
    task_mode = bool(getattr(context, "task_mode", True))
    if task_mode:
        return handler(request)
    return handler(request.override(response_format=None))


def _tool_error_message(exc: Exception, request) -> str:
    """
    把工具执行异常转成模型能读懂的文本（ToolErrorMiddleware 的 on_error）

    只暴露异常类型，不把完整堆栈/内部细节塞给模型。
    """
    tool_call = getattr(request, "tool_call", {}) or {}
    name = tool_call.get("name", "tool")
    return f"[FAILED] {name} raised {type(exc).__name__}: {exc}"


@dataclass(frozen=True)
class ModelConfig:
    """模型初始化配置"""

    provider: str
    model: str
    api_key: str | None
    base_url: str | None
    supports_extended_thinking: bool

##把 provider 名统一成一个词
def _normalize_provider(provider: str | None) -> str | None:
    """标准化 provider 名称"""
    if provider is None:
        return None

    normalized = provider.strip().lower()
    aliases = {
        "anthropic": "anthropic",
        "claude": "anthropic",
        "openai": "openai",
        "gpt": "openai",
        "deepseek": "deepseek",
        "ds": "deepseek",
    }
    return aliases.get(normalized, normalized)

##环境变量只有字符串，得翻译成 bool
def _parse_bool_env(name: str, default: bool) -> bool:
    """解析布尔环境变量"""
    raw_value = os.getenv(name)
    if raw_value is None:
        return default
    return raw_value.strip().lower() in {"1", "true", "yes", "on"}


def _normalize_openai_base_url(base_url: str | None) -> str | None:
    """规范化 OpenAI SDK base_url，避免传入完整 endpoint 后被重复拼接。"""
    if not base_url:
        return base_url

    normalized = base_url.rstrip("/")
    for endpoint_suffix in ("/chat/completions", "/responses"):
        if normalized.endswith(endpoint_suffix):
            normalized = normalized[: -len(endpoint_suffix)]
            break

    if normalized.endswith("/v1"):
        return normalized
    return f"{normalized}/v1"


def _normalize_deepseek_base_url(base_url: str | None) -> str:
    """规范化 DeepSeek 的 OpenAI 兼容 base_url。

    DeepSeek 的官方 OpenAI 兼容端点就是 https://api.deepseek.com，
    与 OpenAI SDK 不同，不需要（也不应该自动补）``/v1``。
    这里只做两件事：清理尾部斜杠、把误填的完整 endpoint 裁剪回 base 层级。
    """
    normalized = (base_url or DEFAULT_DEEPSEEK_BASE_URL).rstrip("/")
    for endpoint_suffix in ("/chat/completions", "/responses"):
        if normalized.endswith(endpoint_suffix):
            normalized = normalized[: -len(endpoint_suffix)]
            break
    return normalized.rstrip("/")


def _split_provider_prefixed_model(model: str | None) -> tuple[str | None, str | None]:
    """解析 provider:model 形式的模型字符串"""
    if not model or ":" not in model:
        return None, model

    raw_provider, raw_model = model.split(":", 1)
    provider = _normalize_provider(raw_provider)
    if provider in ("anthropic", "openai", "deepseek") and raw_model:
        return provider, raw_model
    return None, model


def _infer_provider_from_model_name(model: str | None) -> str | None:
    """根据模型名推断 provider"""
    if not model:
        return None

    model_name = model.strip().lower()
    anthropic_prefixes = ("claude-",)
    openai_prefixes = ("gpt-", "o1", "o3", "o4", "chatgpt-")
    deepseek_prefixes = ("deepseek-",)

    if model_name.startswith(anthropic_prefixes):
        return "anthropic"
    if model_name.startswith(openai_prefixes):
        return "openai"
    if model_name.startswith(deepseek_prefixes):
        return "deepseek"
    return None


def _resolve_requested_provider(model: str | None = None, model_provider: str | None = None) -> str:
    """解析当前请求应使用的 provider"""
    explicit_provider = _normalize_provider(model_provider or os.getenv("MODEL_PROVIDER"))
    prefixed_provider, stripped_model = _split_provider_prefixed_model(model)
    generic_env_provider, generic_env_model = _split_provider_prefixed_model(os.getenv("MODEL_NAME"))

    env_provider_hint = (
        generic_env_provider
        or _infer_provider_from_model_name(generic_env_model)
        or ("openai" if os.getenv("OPENAI_MODEL") else None)
        or ("anthropic" if os.getenv("ANTHROPIC_MODEL") or os.getenv("CLAUDE_MODEL") else None)
        or (
            "deepseek"
            if os.getenv("DEEPSEEK_MODEL")
            or os.getenv("DEEPSEEK_API_KEY")
            or os.getenv("DEEPSEEK_BASE_URL")
            else None
        )
    )

    provider = (
        explicit_provider
        or prefixed_provider
        or _infer_provider_from_model_name(stripped_model)
        or env_provider_hint
        or DEFAULT_PROVIDER
    )

    if provider not in ("anthropic", "openai", "deepseek"):
        raise ValueError(f"Unsupported model provider: {provider}")

    return provider


def _resolve_model_name(provider: str, requested_model: str | None = None) -> str:
    """解析模型名称，兼容旧环境变量"""
    requested_provider, stripped_model = _split_provider_prefixed_model(requested_model)
    if requested_provider and requested_provider != provider:
        raise ValueError(
            f"Model provider mismatch: requested '{requested_provider}' but configured '{provider}'"
        )
    if stripped_model:
        return stripped_model

    generic_model = os.getenv("MODEL_NAME")
    generic_provider, stripped_generic_model = _split_provider_prefixed_model(generic_model)
    if generic_provider and generic_provider != provider:
        stripped_generic_model = None
    if stripped_generic_model:
        return stripped_generic_model

    if provider == "deepseek":
        return os.getenv("DEEPSEEK_MODEL") or DEFAULT_DEEPSEEK_MODEL

    if provider == "openai":
        return os.getenv("OPENAI_MODEL", DEFAULT_OPENAI_MODEL)

    return (
        os.getenv("ANTHROPIC_MODEL")
        or os.getenv("CLAUDE_MODEL")
        or DEFAULT_ANTHROPIC_MODEL
    )


def _get_provider_credentials(provider: str) -> tuple[str | None, str | None]:
    """
    获取 provider 对应的 API 认证信息

    支持多种认证方式：
    1. 通用配置：MODEL_API_KEY / MODEL_BASE_URL
    2. Provider 专属配置
    3. OpenAI 兼容代理场景下，允许复用 ANTHROPIC_AUTH_TOKEN
       作为 OpenAI 端点的共享平台 Token

    Returns:
        (api_key, base_url) 元组
    """
    api_key = os.getenv("MODEL_API_KEY")
    base_url = os.getenv("MODEL_BASE_URL")

    if provider == "openai":
        api_key = (
            api_key
            or os.getenv("OPENAI_API_KEY")
            or os.getenv("OPENAI_AUTH_TOKEN")
            or os.getenv("ANTHROPIC_AUTH_TOKEN")
            or os.getenv("ANTHROPIC_API_KEY")
        )
        base_url = base_url or os.getenv("OPENAI_BASE_URL")
    elif provider == "deepseek":
        api_key = (
            api_key
            or os.getenv("DEEPSEEK_API_KEY")
            or os.getenv("DEEPSEEK_AUTH_TOKEN")
        )
        base_url = (
            base_url
            or os.getenv("DEEPSEEK_BASE_URL")
            or DEFAULT_DEEPSEEK_BASE_URL
        )
    else:
        api_key = (
            api_key
            or os.getenv("ANTHROPIC_API_KEY")
            or os.getenv("ANTHROPIC_AUTH_TOKEN")
        )
        base_url = base_url or os.getenv("ANTHROPIC_BASE_URL")

    return api_key, base_url


def resolve_model_config(model: str | None = None, model_provider: str | None = None) -> ModelConfig:
    """解析当前模型配置"""
    provider = _resolve_requested_provider(model=model, model_provider=model_provider)
    model_name = _resolve_model_name(provider, model)
    api_key, base_url = _get_provider_credentials(provider)

    return ModelConfig(
        provider=provider,
        model=model_name,
        api_key=api_key,
        base_url=base_url,
        supports_extended_thinking=(
            provider in {"anthropic", "openai"}
            or (provider == "deepseek" and _parse_bool_env("DEEPSEEK_THINKING", False))
        ),
    )


def check_api_credentials(model: str | None = None, model_provider: str | None = None) -> bool:
    """检查是否配置了当前 provider 的 API 认证"""
    api_key, _ = _get_provider_credentials(
        _resolve_requested_provider(model=model, model_provider=model_provider)
    )
    return api_key is not None


class LangChainSkillsAgent:
    """
    基于 LangChain 1.0 的 Skills Agent

    演示目的：展示 Skills 三层加载机制的底层原理

    使用示例：
        agent = LangChainSkillsAgent()

        # 查看 system prompt（展示 Level 1）
        print(agent.get_system_prompt())

        # 运行 agent
        for chunk in agent.stream("提取这篇公众号文章"):
            response = agent.get_last_response(chunk)
            if response:
                print(response)
    """

    def __init__(
        self,
        model: Optional[str] = None,
        model_provider: Optional[str] = None,
        skill_paths: Optional[list[Path]] = None,
        working_directory: Optional[Path] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        enable_thinking: bool = True,
        thinking_budget: int = DEFAULT_THINKING_BUDGET,
    ):
        """
        初始化 Agent

        Args:
            model: 模型名称，默认 deepseek-flash
            model_provider: 模型提供商，支持 anthropic / openai / deepseek
            skill_paths: Skills 搜索路径
            working_directory: 工作目录
            max_tokens: 最大 tokens
            temperature: 温度参数 (启用 thinking 时强制为 1.0)
            enable_thinking: 是否启用 Extended Thinking
            thinking_budget: thinking 的 token 预算
        """
        self.model_config = resolve_model_config(model=model, model_provider=model_provider)
        self.model_provider = self.model_config.provider
        self.model_name = self.model_config.model

        # thinking 配置（Anthropic 原生支持；DeepSeek 需 DEEPSEEK_THINKING=true，
        # 且开启后带 tools 的多轮请求必须回传 reasoning_content，属实验性用法）
        self.enable_thinking = enable_thinking and self.model_config.supports_extended_thinking
        self.thinking_budget = thinking_budget

        # 配置 (Anthropic 启用 thinking 时温度必须为 1.0)
        self.max_tokens = max_tokens or int(os.getenv("MAX_TOKENS", str(DEFAULT_MAX_TOKENS)))
        if self.enable_thinking:
            self.temperature = 1.0  # Anthropic 要求启用 thinking 时温度为 1.0
        else:
            self.temperature = (
                temperature
                if temperature is not None
                else float(os.getenv("MODEL_TEMPERATURE", str(DEFAULT_TEMPERATURE)))
            )
        self.working_directory = working_directory or Path.cwd()

        # 长期记忆归属的用户（namespace 隔离用；生产可换成登录态传入）
        self.memory_user_id = (
            os.getenv("SKILLS_MEMORY_USER", "").strip() or "local-user"
        )

        # 初始化 SkillLoader
        self.skill_loader = SkillLoader(skill_paths)

        # Level 1: 构建 system prompt（将 Skills 元数据注入）
        self.system_prompt = self._build_system_prompt()

        # 创建上下文（供 tools 使用）
        self.context = SkillAgentContext(
            skill_loader=self.skill_loader,
            working_directory=self.working_directory,
            user_id=self.memory_user_id,
        )

        # 每个 thread 最近一次的意图判定（resume HITL 时沿用，避免二次分类）
        self._task_modes: dict[str, bool] = {}

        # 长期记忆库（跨 thread 共享；生产环境换成 PostgresStore / RedisStore）
        self.store = self._build_store()

        # 创建 LangChain Agent
        self.agent = self._create_agent()

    def _build_system_prompt(self) -> str:
        """
        构建 system prompt

        这是 Level 1 的核心：将所有 Skills 的元数据注入到 system prompt。
        每个 skill 约 100 tokens，启动时一次性加载。
        """
        base_prompt = """You are a helpful coding assistant with access to specialized skills.

Your capabilities include:
- Loading and using specialized skills for specific tasks
- Executing bash commands and scripts
- Reading and writing files
- Following skill instructions to complete complex tasks

Long-term memory:
- Use the `remember` tool when the user shares a durable fact or preference
  worth keeping beyond this conversation (e.g. "I use Java, not Python").
- Use the `recall` tool before answering questions about the user's
  preferences, past decisions, or ongoing work from earlier sessions.

Knowledge base (RAG):
- Use the `search_docs` tool before answering questions about documented
  facts: refund/return policies, token expiry, error codes, product rules,
  or anything that may live in the knowledge/ directory.
- Knowledge questions ("where do I submit...", "how long does it take...",
  "what does error 40102 mean") should be answered from `search_docs`
  results, not by describing actions you would take.
- Cite the returned source (and section/page when available) in your answer.
  When you deliver a structured `AgentReply`, put the real answer in
  `summary` and list the sources in its `sources` field.
- If no relevant documents are found, say the knowledge base doesn't cover
  it instead of guessing; never invent a policy, date, or error code.

When a user request matches a skill's description, use the load_skill tool to get detailed instructions before proceeding."""

        return self.skill_loader.build_system_prompt(base_prompt)

    @staticmethod
    def _build_checkpointer() -> InMemorySaver:
        """
        构建内存检查点

        AgentReply 会被写进图状态并随 checkpoint 序列化；
        显式登记到 allowed_msgpack_modules，避免未来 LangGraph 严格模式
        拒绝反序列化这个自定义类型。
        """
        try:
            from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer

            serde = JsonPlusSerializer(
                allowed_msgpack_modules=[("langchain_skills.agent", "AgentReply")]
            )
            return InMemorySaver(serde=serde)
        except Exception:
            # 老版本 langgraph 没有该参数时退回默认行为
            return InMemorySaver()

    @staticmethod
    def _build_store() -> InMemoryStore:
        """
        构建长期记忆库

        LangGraph store 保存的是按 namespace + key 组织的 JSON 文档，
        与 checkpointer（会话状态）是两套存储：
        - checkpointer: 单个 thread 内的消息历史
        - store: 跨 thread / 跨会话的事实、偏好、经验

        InMemoryStore 仅用于开发演示，进程退出即清空；
        生产环境应换成 PostgresStore / RedisStore / MongoDBStore。
        """
        return InMemoryStore()

    def _create_agent(self):
        """
        创建 LangChain Agent

        使用 LangChain 1.0 的 create_agent API:
        - model: 可以是字符串 ID 或 model 实例
        - tools: 工具列表
        - system_prompt: 系统提示（Level 1 注入 Skills 元数据）
        - context_schema: 上下文类型（供 ToolRuntime 使用）
        - checkpointer: 会话记忆（thread 内）
        - store: 长期记忆（跨 thread，工具里通过 runtime.store 读写）

        Extended Thinking 支持:
        - Anthropic: 使用 thinking budget 获取思考过程
        - OpenAI-compatible: 默认走 chat/completions + reasoning_effort，
          避免部分代理不支持 Responses API
        - DeepSeek: 默认关闭思考模式（DEEPSEEK_THINKING=true 才开启），
          通过 extra_body.thinking 控制

        认证支持:
        - 默认使用 DeepSeek（https://api.deepseek.com）
        - 支持 Anthropic
        - 支持 OpenAI / OpenAI-compatible base_url
        - 支持 MODEL_* 通用变量和 provider 专属变量
        """
        # 构建初始化参数
        init_kwargs = {
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }

        # DeepSeek 走 OpenAI 兼容协议：底层仍然是 ChatOpenAI，
        # 但不需要 Responses API，base_url 也不能被补成 /v1。
        init_kwargs["model_provider"] = (
            "openai" if self.model_provider == "deepseek" else self.model_provider
        )

        base_url = self.model_config.base_url

        # 添加认证参数（支持第三方代理）
        if self.model_config.api_key:
            init_kwargs["api_key"] = self.model_config.api_key

        # Provider-specific thinking / reasoning 配置
        if self.model_provider == "anthropic" and self.enable_thinking:
            init_kwargs["thinking"] = {
                "type": "enabled",
                "budget_tokens": self.thinking_budget,
            }
        elif self.model_provider == "openai" and self.enable_thinking:
            use_responses_api = _parse_bool_env("OPENAI_USE_RESPONSES_API", True)
            reasoning_effort = (
                os.getenv("OPENAI_REASONING_EFFORT")
                or os.getenv("MODEL_REASONING_EFFORT")
                or DEFAULT_OPENAI_REASONING_EFFORT
            )
            init_kwargs["use_responses_api"] = use_responses_api
            if use_responses_api:
                init_kwargs["reasoning"] = {
                    "effort": reasoning_effort,
                    "summary": os.getenv("OPENAI_REASONING_SUMMARY", "auto"),
                }
                base_url = _normalize_openai_base_url(base_url)
            else:
                init_kwargs["reasoning_effort"] = reasoning_effort
                base_url = _normalize_openai_base_url(base_url)
        elif self.model_provider == "deepseek":
            # DeepSeek 的思考开关属于 OpenAI 兼容协议的扩展字段，必须放在 extra_body 里。
            # 这里默认显式关闭：DeepSeek 思考模式默认开启，而开启后只要请求带 tools，
            # 历史轮次的 reasoning_content 就必须原样回传，否则 API 返回 400。
            init_kwargs["extra_body"] = {
                "thinking": {"type": "enabled" if self.enable_thinking else "disabled"}
            }
            if self.enable_thinking:
                init_kwargs["reasoning_effort"] = (
                    os.getenv("DEEPSEEK_REASONING_EFFORT")
                    or DEFAULT_DEEPSEEK_REASONING_EFFORT
                )
            base_url = _normalize_deepseek_base_url(base_url)

        if base_url:
            init_kwargs["base_url"] = base_url

        # 初始化模型
        model = init_chat_model(
            self.model_name,
            **init_kwargs,
        )

        # 创建 Agent
        agent = create_agent(
            model=model,
            tools=ALL_TOOLS,
            system_prompt=self.system_prompt,
            context_schema=SkillAgentContext,
            checkpointer=self._build_checkpointer(),
            store=self.store,
            middleware=self._build_middleware(),
            # 声明任务交付 schema；聊天轮次由 _route_by_intent 覆盖为 None
            response_format=AgentReply,
        )

        return agent

    def _build_middleware(self) -> list:
        """
        组装中间件（顺序有约束）

        顺序（列表靠前者是 wrap 钩子的外层）：
        0. IntentRouter  — 按本轮意图决定是否启用 AgentReply 结构化输出（最外层）
        1. ModelRetry  — 模型调用失败时指数退避重试
        2. ToolError   — 把工具异常转成 ToolMessage，让模型看到 [FAILED] 而不是崩溃
        3. HITL        — bash / write_file 执行前人工确认（必须在 ToolRetry 外层，
                         否则重试会绕开审批）
        4. ToolRetry   — 工具临时失败时重试
        5. Summarization — 历史超阈值时自动摘要（放在最后，避免影响审批判断）

        ToolError 必须在 ToolRetry 外层：先重试，重试仍失败才兜底成 ToolMessage；
        如果顺序反了，异常会先被兜底吞掉，重试永远不会触发。
        """
        middleware: list = [
            # 意图路由：聊天轮次关闭结构化输出（必须最外层，先于其他 wrap 生效）
            _route_by_intent,
            ModelRetryMiddleware(
                max_retries=DEFAULT_MAX_RETRIES,
                backoff_factor=2.0,
            ),
        ]

        # 工具异常兜底（官方 ToolErrorMiddleware，langchain >= 1.3.14）
        middleware.append(ToolErrorMiddleware(_tool_error_message))

        # 人工审批（HITL）：只拦 bash / write_file
        if _parse_bool_env("SKILLS_HITL", True):
            middleware.append(
                HumanInTheLoopMiddleware(
                    interrupt_on=DEFAULT_HITL_INTERRUPT_ON,
                    description_prefix="工具执行需要确认",
                )
            )

        # 工具重试（on_failure="error"：重试耗尽后把异常抛给外层 ToolError 兜底）
        middleware.append(
            ToolRetryMiddleware(
                max_retries=DEFAULT_MAX_RETRIES,
                backoff_factor=2.0,
                on_failure="error",
            )
        )

        # 历史摘要：trigger 用绝对 token 数（fraction 需要 model profile，DeepSeek 没有）
        trigger_tokens = int(
            os.getenv(
                "SKILLS_SUMMARY_TRIGGER_TOKENS", str(DEFAULT_SUMMARY_TRIGGER_TOKENS)
            )
        )
        if trigger_tokens > 0:
            keep_messages = int(
                os.getenv(
                    "SKILLS_SUMMARY_KEEP_MESSAGES", str(DEFAULT_SUMMARY_KEEP_MESSAGES)
                )
            )
            middleware.append(
                SummarizationMiddleware(
                    model=self._build_summarizer_model(),
                    trigger=("tokens", trigger_tokens),
                    keep=("messages", keep_messages),
                )
            )

        return middleware

    def _build_summarizer_model(self):
        """
        构建摘要专用模型

        不能复用主模型实例：主模型开启 DeepSeek 思考模式时，摘要请求会带上
        reasoning_content，既费钱又容易触发"带 tools 必须回传 reasoning_content"的 400。
        这里单独构造一个关闭思考、低温度、小输出预算的模型实例。
        """
        init_kwargs: dict[str, Any] = {
            "temperature": DEFAULT_SUMMARY_TEMPERATURE,
            "max_tokens": DEFAULT_SUMMARY_MAX_TOKENS,
            "model_provider": (
                "openai" if self.model_provider == "deepseek" else self.model_provider
            ),
        }

        if self.model_config.api_key:
            init_kwargs["api_key"] = self.model_config.api_key

        base_url = self.model_config.base_url
        if self.model_provider == "deepseek":
            init_kwargs["extra_body"] = {"thinking": {"type": "disabled"}}
            base_url = _normalize_deepseek_base_url(base_url)
        elif self.model_provider == "openai":
            base_url = _normalize_openai_base_url(base_url)

        if base_url:
            init_kwargs["base_url"] = base_url

        return init_chat_model(self.model_name, **init_kwargs)

    def _build_classifier_model(self):
        """
        构建意图分类专用模型

        与摘要模型同理：单独构造、关闭思考、温度为 0，避免分类结果抖动。
        可用 SKILLS_CLASSIFIER_MODEL 换更便宜的模型（同一 provider）。
        """
        init_kwargs: dict[str, Any] = {
            "temperature": DEFAULT_INTENT_TEMPERATURE,
            "max_tokens": DEFAULT_INTENT_MAX_TOKENS,
            "model_provider": (
                "openai" if self.model_provider == "deepseek" else self.model_provider
            ),
        }

        if self.model_config.api_key:
            init_kwargs["api_key"] = self.model_config.api_key

        base_url = self.model_config.base_url
        if self.model_provider == "deepseek":
            init_kwargs["extra_body"] = {"thinking": {"type": "disabled"}}
            base_url = _normalize_deepseek_base_url(base_url)
        elif self.model_provider == "openai":
            base_url = _normalize_openai_base_url(base_url)

        if base_url:
            init_kwargs["base_url"] = base_url

        model_name = os.getenv("SKILLS_CLASSIFIER_MODEL") or self.model_name
        return init_chat_model(model_name, **init_kwargs)

    def _classify_intent(self, message: str) -> dict:
        """
        判断这条用户消息是任务型还是聊天型

        顺序：环境开关 → 关键词规则 → 小模型分类 → 保守兜底（按聊天处理）。
        返回 {"is_task": bool, "source": "rule|llm|fallback|disabled", "reason": str}
        """
        if not _parse_bool_env("SKILLS_INTENT_ROUTER", True):
            return {
                "is_task": False,
                "source": "disabled",
                "reason": "意图路由已关闭，按聊天型处理",
            }

        rule_result = _classify_by_rules(message)
        if rule_result is not None:
            return {
                "is_task": rule_result,
                "source": "rule",
                "reason": "关键词规则命中",
            }

        try:
            classifier = self._build_classifier_model()
            # DeepSeek 不支持 json_schema 形式的 response_format，
            # 必须走 function_calling（伪工具）方式做结构化输出。
            structured_classifier = classifier.with_structured_output(
                Intent, method="function_calling"
            )
            result = structured_classifier.invoke(
                [
                    {"role": "system", "content": INTENT_CLASSIFIER_PROMPT},
                    {"role": "user", "content": message[:4000]},
                ]
            )
            return {
                "is_task": bool(result.is_task),
                "source": "llm",
                "reason": getattr(result, "reason", ""),
            }
        except Exception as exc:
            # 分类失败时保守处理：不启用结构化输出，绝不让副作用被结构化流程带跑
            return {
                "is_task": False,
                "source": "fallback",
                "reason": f"分类失败，按聊天型处理：{exc}",
            }

    def _context_with_task_mode(self, task_mode: bool) -> SkillAgentContext:
        """复制一份带本轮 task_mode 的上下文，避免并发请求互相污染"""
        if not hasattr(self.context, "task_mode"):
            return self.context
        return replace(self.context, task_mode=task_mode)

    def _prepare_intent(
        self, message: str, thread_id: str
    ) -> tuple[dict, SkillAgentContext]:
        """判定本轮意图并返回 (intent, 带 task_mode 的上下文)"""
        intent = self._classify_intent(message)
        task_mode = bool(intent["is_task"])
        self._task_modes[thread_id] = task_mode
        return intent, self._context_with_task_mode(task_mode)

    def get_system_prompt(self) -> str:
        """
        获取当前 system prompt

        用于演示和调试，展示 Level 1 注入的内容。
        """
        return self.system_prompt

    def get_discovered_skills(self) -> list[dict]:
        """
        获取发现的 Skills 列表

        用于演示 Level 1 的 Skills 发现过程。
        """
        skills = self.skill_loader.scan_skills()
        return [
            {
                "name": s.name,
                "description": s.description,
                "path": str(s.skill_path),
            }
            for s in skills
        ]

    def invoke(self, message: str, thread_id: str = "default") -> dict:
        """
        同步调用 Agent

        Args:
            message: 用户消息
            thread_id: 会话 ID（用于多轮对话）

        Returns:
            Agent 响应
        """
        config = {"configurable": {"thread_id": thread_id}}
        intent, context = self._prepare_intent(message, thread_id)

        result = self.agent.invoke(
            {"messages": [{"role": "user", "content": message}]},
            config=config,
            context=context,
        )

        if isinstance(result, dict):
            result.setdefault("intent", intent)
        return result

    def stream(self, message: str, thread_id: str = "default") -> Iterator[dict]:
        """
        流式调用 Agent (state 级别)

        Args:
            message: 用户消息
            thread_id: 会话 ID

        Yields:
            流式响应块 (完整状态更新)
        """
        config = {"configurable": {"thread_id": thread_id}}
        _, context = self._prepare_intent(message, thread_id)

        for chunk in self.agent.stream(
            {"messages": [{"role": "user", "content": message}]},
            config=config,
            context=context,
            stream_mode="values",
        ):
            yield chunk

    def stream_events(self, message: str, thread_id: str = "default") -> Iterator[dict]:
        """
        事件级流式输出，支持 thinking 和 token 级流式

        Args:
            message: 用户消息
            thread_id: 会话 ID

        Yields:
            事件字典，格式如下:
            - {"type": "intent", "data": {...}} - 本轮意图判定（任务型/聊天型）
            - {"type": "thinking", "content": "..."} - 思考内容片段
            - {"type": "text", "content": "..."} - 响应文本片段
            - {"type": "tool_call", "name": "...", "args": {...}} - 工具调用
            - {"type": "tool_result", "name": "...", "content": "...", "success": bool} - 工具结果
            - {"type": "interrupt", "interrupts": [...]} - 需要人工审批（HITL），本轮不再发 done
            - {"type": "done", "response": "...", "structured": {...}} - 完成标记（任务型带结构化报告）
        """
        config = {"configurable": {"thread_id": thread_id}}
        intent, context = self._prepare_intent(message, thread_id)
        yield {"type": "intent", "data": intent}
        yield from self._stream_loop(
            {"messages": [{"role": "user", "content": message}]},
            config,
            context=context,
        )

    def resume_events(
        self, decisions: list[dict], thread_id: str = "default"
    ) -> Iterator[dict]:
        """
        人工审批后继续执行（HITL resume）

        Args:
            decisions: 与 action_requests 对应的决策列表，例如
                [{"type": "approve"}] 或
                [{"type": "reject", "message": "不要删除这个文件"}]
            thread_id: 会话 ID（必须与中断时相同）

        Yields:
            与 stream_events 相同的事件字典
        """
        config = {"configurable": {"thread_id": thread_id}}
        pending = self.get_pending_interrupts(thread_id)
        # 沿用中断时那一轮的意图，不再重新分类
        task_mode = self._task_modes.get(thread_id, False)
        yield from self._stream_loop(
            self._build_resume_input(decisions, pending),
            config,
            context=self._context_with_task_mode(task_mode),
        )

    def get_pending_interrupts(self, thread_id: str = "default") -> list[dict]:
        """读取当前会话是否停在人工审批点"""
        config = {"configurable": {"thread_id": thread_id}}
        return self._interrupts_from_snapshot(self._read_state_snapshot(config))

    @staticmethod
    def _build_resume_input(decisions: list[dict], pending: list[dict]) -> Command:
        """构造 resume 输入：单中断直接传 decisions，多中断按 interrupt id 分组"""
        if len(pending) > 1:
            return Command(
                resume={item["id"]: {"decisions": decisions} for item in pending}
            )
        return Command(resume={"decisions": decisions})

    def _stream_loop(
        self,
        input_payload: Any,
        config: dict,
        context: Optional[SkillAgentContext] = None,
    ) -> Iterator[dict]:
        """执行一次图循环并翻译成事件（stream_events / resume_events 共用）"""
        active_context = context or self.context
        emitter = StreamEventEmitter()
        tracker = ToolCallTracker()

        full_response = ""
        reasoning_tokens = 0
        thinking_seen = False
        used_search = False
        debug = os.getenv("SKILLS_DEBUG", "").lower() in ("1", "true", "yes")

        # 使用 messages 模式获取 token 级流式
        try:
            for event in self.agent.stream(
                input_payload,
                config=config,
                context=active_context,
                stream_mode="messages",
            ):
                # event 可能是 tuple(message, metadata) 或直接 message
                if isinstance(event, tuple) and len(event) >= 2:
                    chunk = event[0]
                else:
                    chunk = event

                if debug:
                    chunk_type = type(chunk).__name__
                    print(f"[DEBUG] Event: {chunk_type}")

                # 处理 AIMessageChunk / AIMessage
                if isinstance(chunk, (AIMessageChunk, AIMessage)):
                    reasoning_tokens += self._extract_reasoning_tokens(chunk)
                    # DeepSeek 思考模式：reasoning_content 与 content 同级返回，
                    # LangChain 会把它放进 additional_kwargs，需要单独取出。
                    reasoning_delta = self._extract_reasoning_content(chunk)
                    if reasoning_delta:
                        thinking_seen = True
                        yield emitter.thinking(reasoning_delta).data
                    # 处理 content
                    for ev in self._process_chunk_content(chunk, emitter, tracker):
                        if ev.type == "thinking":
                            thinking_seen = True
                        if ev.type == "text":
                            full_response += ev.data.get("content", "")
                        if debug:
                            print(f"[DEBUG] Yielding: {ev.type}")
                        yield ev.data

                    # 处理 tool_calls (有些情况下在 chunk.tool_calls 中)
                    if hasattr(chunk, "tool_calls") and chunk.tool_calls:
                        for ev in self._process_tool_calls(chunk.tool_calls, emitter, tracker):
                            if debug:
                                print(f"[DEBUG] Yielding from tool_calls: {ev.type}")
                            yield ev.data

                # 处理 ToolMessage (工具执行结果)
                elif hasattr(chunk, "type") and chunk.type == "tool":
                    if str(getattr(chunk, "name", "") or "") == "search_docs":
                        used_search = True
                    if debug:
                        tool_name = getattr(chunk, "name", "unknown")
                        print(f"[DEBUG] Processing tool result: {tool_name}")
                    for ev in self._process_tool_result(chunk, emitter, tracker):
                        if debug:
                            print(f"[DEBUG] Yielding: {ev.type}")
                        yield ev.data

            if debug:
                print("[DEBUG] Stream completed normally")

        except Exception as e:
            if debug:
                import traceback
                print(f"[DEBUG] Stream error: {e}")
                traceback.print_exc()
            # 发送错误事件让用户知道发生了什么
            yield emitter.error(str(e)).data
            raise

        if (
            self.enable_thinking
            and reasoning_tokens > 0
            and not thinking_seen
            and self.model_provider in {"openai", "deepseek"}
        ):
            label = "OpenAI" if self.model_provider == "openai" else "DeepSeek"
            yield emitter.thinking(
                f"[{label} reasoning enabled: used {reasoning_tokens} reasoning tokens. "
                "This endpoint does not expose reasoning summary text in the stream.]"
            ).data

        # 聊天模式兜底：检索过知识库，但回答既没引用来源、也没承认未覆盖时，
        # 追加一句确定性提示（不依赖模型自觉），避免用户把猜测当有出处的答案。
        if used_search and not active_context.task_mode:
            suffix = self._knowledge_gap_suffix(full_response)
            if suffix:
                full_response += suffix
                yield emitter.text(suffix).data

        # 流结束：若停在人工审批点，先发 interrupt（不发 done），否则发 done
        snapshot = self._read_state_snapshot(config)
        pending = self._interrupts_from_snapshot(snapshot)
        if pending:
            yield emitter.interrupt(pending).data
            return
        yield emitter.done(
            full_response,
            structured=self._read_structured_response(config, snapshot=snapshot),
        ).data

    @staticmethod
    def _knowledge_gap_suffix(text: str) -> str:
        """聊天模式下判断是否要给回答补一句"知识库可能未覆盖"的提示

        Returns:
            需要提示时返回可追加的文本，否则返回空字符串。
        """
        if not text.strip():
            return ""
        if _DOC_CITATION_PATTERN.search(text):
            return ""
        if _KNOWLEDGE_GAP_PATTERN.search(text):
            return ""
        return _KNOWLEDGE_GAP_SUFFIX

    def _read_state_snapshot(self, config: dict):
        """读取图状态快照（失败返回 None）"""
        try:
            return self.agent.get_state(config)
        except Exception:
            return None

    @staticmethod
    def _interrupts_from_snapshot(snapshot) -> list[dict]:
        """从状态快照里提取待审批中断"""
        return [
            {"id": interrupt.id, "value": interrupt.value}
            for interrupt in (getattr(snapshot, "interrupts", None) or [])
        ]

    def _read_structured_response(self, config: dict, snapshot=None) -> dict | None:
        """从图状态里读取结构化输出，转成可 JSON 序列化的 dict

        只有本轮是任务型才返回，避免同 thread 上一轮的结构化结果串到聊天轮。
        """
        thread_id = config.get("configurable", {}).get("thread_id", "default")
        if not self._task_modes.get(thread_id, False):
            return None
        if snapshot is None:
            snapshot = self._read_state_snapshot(config)
        values = getattr(snapshot, "values", None) or {}
        structured = values.get("structured_response")
        if structured is None:
            return None
        if hasattr(structured, "model_dump"):
            return structured.model_dump()
        return structured if isinstance(structured, dict) else None

    def _process_chunk_content(self, chunk, emitter: StreamEventEmitter, tracker: ToolCallTracker):
        """处理 chunk 的 content"""
        content = chunk.content

        if isinstance(content, str):
            if content:
                yield emitter.text(content)
                return

        blocks = None
        if hasattr(chunk, "content_blocks"):
            try:
                blocks = chunk.content_blocks
            except Exception:
                blocks = None

        if blocks is None:
            if isinstance(content, dict):
                blocks = [content]
            elif isinstance(content, list):
                blocks = content
            else:
                return

        for raw_block in blocks:
            block = raw_block
            if not isinstance(block, dict):
                if hasattr(block, "model_dump"):
                    block = block.model_dump()
                elif hasattr(block, "dict"):
                    block = block.dict()
                else:
                    continue

            block_type = block.get("type")

            if block_type in ("thinking", "reasoning"):
                thinking_text = block.get("thinking") or block.get("reasoning") or ""
                if thinking_text:
                    yield emitter.thinking(thinking_text)

            elif block_type == "text":
                text = block.get("text") or block.get("content") or ""
                if text:
                    yield emitter.text(text)

            elif block_type in ("tool_use", "tool_call"):
                tool_id = block.get("id", "")
                name = block.get("name", "")
                args = block.get("input") if block_type == "tool_use" else block.get("args")
                args_payload = args if isinstance(args, dict) else {}

                if tool_id:
                    tracker.update(tool_id, name=name, args=args_payload)
                    # 立即发送（显示"执行中"状态），参数可能尚不完整
                    if tracker.is_ready(tool_id):
                        tracker.mark_emitted(tool_id)
                        yield emitter.tool_call(name, args_payload, tool_id)

            elif block_type == "input_json_delta":
                # 累积 JSON 片段（args 分批到达）
                partial_json = block.get("partial_json", "")
                if partial_json:
                    tracker.append_json_delta(partial_json, block.get("index", 0))

            elif block_type == "tool_call_chunk":
                tool_id = block.get("id", "")
                name = block.get("name", "")
                if tool_id:
                    tracker.update(tool_id, name=name)
                partial_args = block.get("args", "")
                if isinstance(partial_args, str) and partial_args:
                    tracker.append_json_delta(partial_args, block.get("index", 0))

    def _handle_tool_use_block(self, block: dict, emitter: StreamEventEmitter, tracker: ToolCallTracker):
        """处理 tool_use 块 - 立即发送 tool_call 事件

        在收到 tool_use 时立即发送，让 CLI 可以显示"正在执行"状态。
        避免重复发送（同一 tool 可能通过多个路径到达）。
        """
        tool_id = block.get("id", "")
        if tool_id:
            name = block.get("name", "")
            args = block.get("input", {})
            args_payload = args if isinstance(args, dict) else {}

            tracker.update(tool_id, name=name, args=args_payload)
            if tracker.is_ready(tool_id):
                tracker.mark_emitted(tool_id)
                yield emitter.tool_call(name, args_payload, tool_id)

    def _process_tool_calls(self, tool_calls: list, emitter: StreamEventEmitter, tracker: ToolCallTracker):
        """处理 chunk.tool_calls - 立即发送 tool_call 事件

        避免重复发送（同一 tool 可能通过 tool_use block 已发送）。
        """
        for tc in tool_calls:
            tool_id = tc.get("id", "")
            if tool_id:
                name = tc.get("name", "")
                args = tc.get("args", {})
                args_payload = args if isinstance(args, dict) else {}

                tracker.update(tool_id, name=name, args=args_payload)
                if tracker.is_ready(tool_id):
                    tracker.mark_emitted(tool_id)
                    yield emitter.tool_call(name, args_payload, tool_id)

    def _process_tool_result(self, chunk, emitter: StreamEventEmitter, tracker: ToolCallTracker):
        """处理工具结果"""
        # 最终化：解析累积的 JSON 片段为 args
        tracker.finalize_all()

        # 发送所有工具调用的更新（参数现在是完整的）
        # CLI 会用 tool_id 去重和更新
        for info in tracker.get_all():
            yield emitter.tool_call(info.name, info.args, info.id)

        # 发送结果
        name = getattr(chunk, "name", "unknown")
        raw_content = str(getattr(chunk, "content", ""))
        content = raw_content[:DisplayLimits.TOOL_RESULT_MAX]
        if len(raw_content) > DisplayLimits.TOOL_RESULT_MAX:
            content += "\n... (truncated)"

        # 基于内容判断是否成功（统一使用 is_success）
        success = is_success(content)

        yield emitter.tool_result(name, content, success)

    def _extract_reasoning_tokens(self, chunk) -> int:
        """从 OpenAI chunk 的 usage_metadata 中提取 reasoning token 数量"""
        usage_metadata = getattr(chunk, "usage_metadata", None) or {}
        output_details = usage_metadata.get("output_token_details") or {}
        reasoning_tokens = output_details.get("reasoning", 0)
        return reasoning_tokens if isinstance(reasoning_tokens, int) else 0

    def _extract_reasoning_content(self, chunk) -> str:
        """提取 DeepSeek 思考模式返回的 reasoning_content 文本片段

        DeepSeek 的 OpenAI 兼容接口把思考内容放在与 content 同级的
        ``reasoning_content`` 字段里，LangChain 会转存到
        ``additional_kwargs["reasoning_content"]``；流式下是增量文本。
        """
        additional_kwargs = getattr(chunk, "additional_kwargs", None) or {}
        reasoning = additional_kwargs.get("reasoning_content")
        if isinstance(reasoning, str):
            return reasoning
        return ""

    def get_last_response(self, result: dict) -> str:
        """
        从结果中提取最后的 AI 响应文本

        Args:
            result: invoke 或 stream 的结果

        Returns:
            AI 响应文本
        """
        messages = result.get("messages", [])
        for msg in reversed(messages):
            if isinstance(msg, AIMessage) and msg.content:
                if isinstance(msg.content, str):
                    return msg.content
                elif isinstance(msg.content, list):
                    # 处理多部分内容
                    text_parts = []
                    for part in msg.content:
                        if isinstance(part, dict) and part.get("type") == "text":
                            text_parts.append(part.get("text", ""))
                        elif isinstance(part, str):
                            text_parts.append(part)
                    return "\n".join(text_parts)
        return ""


def create_skills_agent(
    model: Optional[str] = None,
    model_provider: Optional[str] = None,
    skill_paths: Optional[list[Path]] = None,
    working_directory: Optional[Path] = None,
    enable_thinking: bool = True,
    thinking_budget: int = DEFAULT_THINKING_BUDGET,
) -> LangChainSkillsAgent:
    """
    便捷函数：创建 Skills Agent

    Args:
        model: 模型名称
        model_provider: 模型提供商
        skill_paths: Skills 搜索路径
        working_directory: 工作目录
        enable_thinking: 是否启用 Extended Thinking
        thinking_budget: thinking 的 token 预算

    Returns:
        配置好的 LangChainSkillsAgent 实例
    """
    return LangChainSkillsAgent(
        model=model,
        model_provider=model_provider,
        skill_paths=skill_paths,
        working_directory=working_directory,
        enable_thinking=enable_thinking,
        thinking_budget=thinking_budget,
    )
