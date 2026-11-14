"""Agent provider routing tests."""

from __future__ import annotations

import json
import os
import threading
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, HTTPServer
from types import SimpleNamespace
from unittest.mock import Mock, patch

from langchain.agents.middleware import (
    HumanInTheLoopMiddleware,
    ModelRetryMiddleware,
    ToolErrorMiddleware,
    ToolRetryMiddleware,
)
from langchain.agents.middleware.types import ModelRequest
from langgraph.types import Command

from langchain_skills.agent import (
    AgentReply,
    Intent,
    LangChainSkillsAgent,
    _classify_by_rules,
    resolve_model_config,
)
from langchain_skills.tools import ALL_TOOLS, _recall_facts, _remember_fact


# Windows 下 OpenSSL 初始化依赖 SYSTEMROOT/WINDIR 环境变量；
# NO_PROXY 必须保留，否则 httpx 会走系统代理（注册表里的 Internet 设置），
# 本机 mock server 的 127.0.0.1 请求会被代理吞掉（返回 502）。
_SYSTEM_ENV_KEYS = ("SYSTEMROOT", "WINDIR", "TEMP", "TMP", "NO_PROXY", "no_proxy")


def _patched_env(overrides: dict[str, str]):
    env = {key: os.environ[key] for key in _SYSTEM_ENV_KEYS if key in os.environ}
    env.update(overrides)
    return patch.dict(os.environ, env, clear=True)


@contextmanager
def openai_compatible_chat_server():
    requests: list[dict[str, object]] = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):  # noqa: N802 - stdlib callback name
            body = self.rfile.read(int(self.headers.get("Content-Length", "0") or 0))
            payload = json.loads(body.decode("utf-8"))
            requests.append({"path": self.path, "payload": payload})

            if payload.get("stream"):
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.end_headers()
                chunks = [
                    {
                        "id": "chatcmpl-test",
                        "object": "chat.completion.chunk",
                        "created": 0,
                        "model": payload.get("model", "deepseek-r1"),
                        "choices": [
                            {
                                "index": 0,
                                "delta": {"role": "assistant", "content": "PING"},
                                "finish_reason": None,
                            }
                        ],
                    },
                    {
                        "id": "chatcmpl-test",
                        "object": "chat.completion.chunk",
                        "created": 0,
                        "model": payload.get("model", "deepseek-r1"),
                        "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                    },
                ]
                for chunk in chunks:
                    self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode("utf-8"))
                self.wfile.write(b"data: [DONE]\n\n")
                return

            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            response = {
                "id": "chatcmpl-test",
                "object": "chat.completion",
                "created": 0,
                "model": payload.get("model", "deepseek-r1"),
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "PING"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            }
            self.wfile.write(json.dumps(response).encode("utf-8"))

        def log_message(self, format, *args):  # noqa: A002 - stdlib callback signature
            return

    server = HTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}", requests
    finally:
        server.shutdown()
        thread.join(timeout=5)
        server.server_close()


def test_resolve_model_config_defaults_to_anthropic_legacy_env():
    with patch.dict(
        os.environ,
        {
            "ANTHROPIC_AUTH_TOKEN": "anthropic-token",
            "ANTHROPIC_BASE_URL": "https://api.jiekou.ai/anthropic",
            "CLAUDE_MODEL": "claude-sonnet-4-5-20250929",
        },
        clear=True,
    ):
        config = resolve_model_config()

    assert config.provider == "anthropic"
    assert config.model == "claude-sonnet-4-5-20250929"
    assert config.api_key == "anthropic-token"
    assert config.base_url == "https://api.jiekou.ai/anthropic"
    assert config.supports_extended_thinking is True


def test_resolve_model_config_supports_openai_generic_env():
    with patch.dict(
        os.environ,
        {
            "MODEL_PROVIDER": "openai",
            "MODEL_NAME": "gpt-5.4",
            "MODEL_API_KEY": "shared-token",
            "MODEL_BASE_URL": "https://api.jiekou.ai/openai",
        },
        clear=True,
    ):
        config = resolve_model_config()

    assert config.provider == "openai"
    assert config.model == "gpt-5.4"
    assert config.api_key == "shared-token"
    assert config.base_url == "https://api.jiekou.ai/openai"
    assert config.supports_extended_thinking is True


def test_resolve_model_config_supports_openai_provider_specific_env():
    with patch.dict(
        os.environ,
        {
            "OPENAI_MODEL": "gpt-5.4",
            "OPENAI_API_KEY": "openai-token",
            "OPENAI_BASE_URL": "https://api.jiekou.ai/openai",
        },
        clear=True,
    ):
        config = resolve_model_config()

    assert config.provider == "openai"
    assert config.model == "gpt-5.4"
    assert config.api_key == "openai-token"
    assert config.base_url == "https://api.jiekou.ai/openai"


def test_openai_agent_enables_reasoning_effort_kwargs():
    fake_loader = Mock()
    fake_loader.build_system_prompt.return_value = "system prompt"

    with patch.dict(
        os.environ,
        {
            "MODEL_PROVIDER": "openai",
            "MODEL_NAME": "gpt-5.4",
            "MODEL_API_KEY": "shared-token",
            "MODEL_BASE_URL": "https://api.jiekou.ai/openai",
        },
        clear=True,
    ), patch("langchain_skills.agent.SkillLoader", return_value=fake_loader), patch(
        "langchain_skills.agent.init_chat_model"
    ) as init_chat_model, patch("langchain_skills.agent.create_agent", return_value=object()), patch("langchain_skills.agent.SummarizationMiddleware"), patch.object(
        LangChainSkillsAgent, "_build_summarizer_model", return_value=Mock()
    ):
        init_chat_model.return_value = object()
        agent = LangChainSkillsAgent(enable_thinking=True)

    kwargs = init_chat_model.call_args.kwargs
    assert agent.model_provider == "openai"
    assert agent.enable_thinking is True
    assert kwargs["model_provider"] == "openai"
    assert kwargs["api_key"] == "shared-token"
    assert kwargs["base_url"] == "https://api.jiekou.ai/openai/v1"
    assert kwargs["use_responses_api"] is True
    assert kwargs["reasoning"] == {"effort": "medium", "summary": "auto"}
    assert "thinking" not in kwargs


def test_openai_chat_completions_adds_v1_to_proxy_base_url():
    fake_loader = Mock()
    fake_loader.build_system_prompt.return_value = "system prompt"

    with patch.dict(
        os.environ,
        {
            "MODEL_PROVIDER": "openai",
            "MODEL_NAME": "deepseek-r1",
            "MODEL_API_KEY": "shared-token",
            "MODEL_BASE_URL": "https://api.example.test/openai",
            "OPENAI_USE_RESPONSES_API": "false",
        },
        clear=True,
    ), patch("langchain_skills.agent.SkillLoader", return_value=fake_loader), patch(
        "langchain_skills.agent.init_chat_model"
    ) as init_chat_model, patch("langchain_skills.agent.create_agent", return_value=object()), patch("langchain_skills.agent.SummarizationMiddleware"), patch.object(
        LangChainSkillsAgent, "_build_summarizer_model", return_value=Mock()
    ):
        init_chat_model.return_value = object()
        LangChainSkillsAgent(enable_thinking=True)

    kwargs = init_chat_model.call_args.kwargs
    assert kwargs["base_url"] == "https://api.example.test/openai/v1"
    assert kwargs["use_responses_api"] is False
    assert kwargs["reasoning_effort"] == "medium"


def test_openai_chat_completions_accepts_full_endpoint_base_url():
    fake_loader = Mock()
    fake_loader.build_system_prompt.return_value = "system prompt"

    with patch.dict(
        os.environ,
        {
            "MODEL_PROVIDER": "openai",
            "MODEL_NAME": "deepseek-r1",
            "MODEL_API_KEY": "shared-token",
            "MODEL_BASE_URL": "https://api.example.test/openai/v1/chat/completions",
            "OPENAI_USE_RESPONSES_API": "false",
        },
        clear=True,
    ), patch("langchain_skills.agent.SkillLoader", return_value=fake_loader), patch(
        "langchain_skills.agent.init_chat_model"
    ) as init_chat_model, patch("langchain_skills.agent.create_agent", return_value=object()), patch("langchain_skills.agent.SummarizationMiddleware"), patch.object(
        LangChainSkillsAgent, "_build_summarizer_model", return_value=Mock()
    ):
        init_chat_model.return_value = object()
        LangChainSkillsAgent(enable_thinking=True)

    kwargs = init_chat_model.call_args.kwargs
    assert kwargs["base_url"] == "https://api.example.test/openai/v1"
    assert kwargs["use_responses_api"] is False


def test_openai_compatible_chat_completions_streams_end_to_end():
    with openai_compatible_chat_server() as (server_url, requests), _patched_env(
        {
            "MODEL_PROVIDER": "openai",
            "MODEL_NAME": "deepseek-r1",
            "MODEL_API_KEY": "shared-token",
            "MODEL_BASE_URL": f"{server_url}/openai",
            "OPENAI_USE_RESPONSES_API": "false",
            "SKILLS_INTENT_ROUTER": "false",
        }
    ):
        agent = LangChainSkillsAgent(skill_paths=[], enable_thinking=True)
        events = list(agent.stream_events("Return exactly PING.", thread_id="openai-compatible-e2e"))

    text = "".join(event.get("content", "") for event in events if event["type"] == "text")

    assert text == "PING"
    assert events[-1] == {"type": "done", "response": "PING"}
    assert requests[0]["path"] == "/openai/v1/chat/completions"
    assert requests[0]["payload"]["model"] == "deepseek-r1"
    assert requests[0]["payload"]["stream"] is True


def test_openai_compatible_full_endpoint_streams_without_duplicate_path():
    with openai_compatible_chat_server() as (server_url, requests), _patched_env(
        {
            "MODEL_PROVIDER": "openai",
            "MODEL_NAME": "deepseek-r1",
            "MODEL_API_KEY": "shared-token",
            "MODEL_BASE_URL": f"{server_url}/openai/v1/chat/completions",
            "OPENAI_USE_RESPONSES_API": "false",
            "SKILLS_INTENT_ROUTER": "false",
        }
    ):
        agent = LangChainSkillsAgent(skill_paths=[], enable_thinking=True)
        events = list(agent.stream_events("Return exactly PING.", thread_id="openai-full-endpoint-e2e"))

    text = "".join(event.get("content", "") for event in events if event["type"] == "text")

    assert text == "PING"
    assert requests[0]["path"] == "/openai/v1/chat/completions"


def test_anthropic_agent_keeps_thinking_kwargs():
    fake_loader = Mock()
    fake_loader.build_system_prompt.return_value = "system prompt"

    with patch.dict(
        os.environ,
        {
            "MODEL_PROVIDER": "anthropic",
            "MODEL_NAME": "claude-sonnet-4-5-20250929",
            "MODEL_API_KEY": "anthropic-token",
            "MODEL_BASE_URL": "https://api.jiekou.ai/anthropic",
        },
        clear=True,
    ), patch("langchain_skills.agent.SkillLoader", return_value=fake_loader), patch(
        "langchain_skills.agent.init_chat_model"
    ) as init_chat_model, patch("langchain_skills.agent.create_agent", return_value=object()), patch("langchain_skills.agent.SummarizationMiddleware"), patch.object(
        LangChainSkillsAgent, "_build_summarizer_model", return_value=Mock()
    ):
        init_chat_model.return_value = object()
        agent = LangChainSkillsAgent(enable_thinking=True, thinking_budget=2048)

    kwargs = init_chat_model.call_args.kwargs
    assert agent.model_provider == "anthropic"
    assert agent.enable_thinking is True
    assert kwargs["model_provider"] == "anthropic"
    assert kwargs["thinking"] == {"type": "enabled", "budget_tokens": 2048}
    assert kwargs["api_key"] == "anthropic-token"
    assert kwargs["base_url"] == "https://api.jiekou.ai/anthropic"


def test_resolve_model_config_defaults_to_deepseek():
    with patch.dict(os.environ, {}, clear=True):
        config = resolve_model_config()

    assert config.provider == "deepseek"
    assert config.model == "deepseek-flash"
    assert config.api_key is None
    assert config.base_url == "https://api.deepseek.com"
    assert config.supports_extended_thinking is False


def test_resolve_model_config_supports_deepseek_provider_specific_env():
    with patch.dict(
        os.environ,
        {
            "DEEPSEEK_MODEL": "deepseek-v4-pro",
            "DEEPSEEK_API_KEY": "deepseek-token",
            "DEEPSEEK_BASE_URL": "https://proxy.example.test/deepseek",
        },
        clear=True,
    ):
        config = resolve_model_config()

    assert config.provider == "deepseek"
    assert config.model == "deepseek-v4-pro"
    assert config.api_key == "deepseek-token"
    assert config.base_url == "https://proxy.example.test/deepseek"


def test_deepseek_agent_disables_thinking_by_default():
    fake_loader = Mock()
    fake_loader.build_system_prompt.return_value = "system prompt"

    with patch.dict(
        os.environ,
        {
            "MODEL_PROVIDER": "deepseek",
            "MODEL_NAME": "deepseek-flash",
            "MODEL_API_KEY": "deepseek-token",
        },
        clear=True,
    ), patch("langchain_skills.agent.SkillLoader", return_value=fake_loader), patch(
        "langchain_skills.agent.init_chat_model"
    ) as init_chat_model, patch("langchain_skills.agent.create_agent", return_value=object()), patch("langchain_skills.agent.SummarizationMiddleware"), patch.object(
        LangChainSkillsAgent, "_build_summarizer_model", return_value=Mock()
    ):
        init_chat_model.return_value = object()
        agent = LangChainSkillsAgent(enable_thinking=True)

    kwargs = init_chat_model.call_args.kwargs
    assert agent.model_provider == "deepseek"
    assert agent.enable_thinking is False
    assert kwargs["model_provider"] == "openai"
    assert kwargs["base_url"] == "https://api.deepseek.com"
    assert kwargs["extra_body"] == {"thinking": {"type": "disabled"}}
    assert "reasoning_effort" not in kwargs
    assert "use_responses_api" not in kwargs
    assert "thinking" not in kwargs


def test_deepseek_agent_can_enable_thinking_explicitly():
    fake_loader = Mock()
    fake_loader.build_system_prompt.return_value = "system prompt"

    with patch.dict(
        os.environ,
        {
            "MODEL_PROVIDER": "deepseek",
            "MODEL_NAME": "deepseek-flash",
            "MODEL_API_KEY": "deepseek-token",
            "DEEPSEEK_THINKING": "true",
            "DEEPSEEK_REASONING_EFFORT": "max",
        },
        clear=True,
    ), patch("langchain_skills.agent.SkillLoader", return_value=fake_loader), patch(
        "langchain_skills.agent.init_chat_model"
    ) as init_chat_model, patch("langchain_skills.agent.create_agent", return_value=object()), patch("langchain_skills.agent.SummarizationMiddleware"), patch.object(
        LangChainSkillsAgent, "_build_summarizer_model", return_value=Mock()
    ):
        init_chat_model.return_value = object()
        agent = LangChainSkillsAgent(enable_thinking=True)

    kwargs = init_chat_model.call_args.kwargs
    assert agent.enable_thinking is True
    assert kwargs["temperature"] == 1.0
    assert kwargs["extra_body"] == {"thinking": {"type": "enabled"}}
    assert kwargs["reasoning_effort"] == "max"
    assert "reasoning" not in kwargs


def test_deepseek_full_endpoint_base_url_is_trimmed():
    fake_loader = Mock()
    fake_loader.build_system_prompt.return_value = "system prompt"

    with patch.dict(
        os.environ,
        {
            "MODEL_PROVIDER": "deepseek",
            "MODEL_NAME": "deepseek-flash",
            "MODEL_API_KEY": "deepseek-token",
            "MODEL_BASE_URL": "https://api.deepseek.com/v1/chat/completions",
        },
        clear=True,
    ), patch("langchain_skills.agent.SkillLoader", return_value=fake_loader), patch(
        "langchain_skills.agent.init_chat_model"
    ) as init_chat_model, patch("langchain_skills.agent.create_agent", return_value=object()), patch("langchain_skills.agent.SummarizationMiddleware"), patch.object(
        LangChainSkillsAgent, "_build_summarizer_model", return_value=Mock()
    ):
        init_chat_model.return_value = object()
        LangChainSkillsAgent(enable_thinking=False)

    kwargs = init_chat_model.call_args.kwargs
    assert kwargs["base_url"] == "https://api.deepseek.com/v1"


def test_deepseek_chat_completions_streams_end_to_end():
    with openai_compatible_chat_server() as (server_url, requests), _patched_env(
        {
            "MODEL_PROVIDER": "deepseek",
            "MODEL_NAME": "deepseek-flash",
            "MODEL_API_KEY": "shared-token",
            "MODEL_BASE_URL": server_url,
            "SKILLS_INTENT_ROUTER": "false",
        }
    ):
        agent = LangChainSkillsAgent(skill_paths=[], enable_thinking=False)
        events = list(agent.stream_events("Return exactly PING.", thread_id="deepseek-e2e"))

    text = "".join(event.get("content", "") for event in events if event["type"] == "text")

    assert text == "PING"
    assert events[-1] == {"type": "done", "response": "PING"}
    assert requests[0]["path"] == "/chat/completions"
    assert requests[0]["payload"]["model"] == "deepseek-flash"
    assert requests[0]["payload"]["stream"] is True
    assert requests[0]["payload"]["thinking"] == {"type": "disabled"}


# ---------------------------------------------------------------------------
# 中间件（middleware）
# ---------------------------------------------------------------------------

_DEEPSEEK_ENV = {
    "MODEL_PROVIDER": "deepseek",
    "MODEL_NAME": "deepseek-flash",
    "MODEL_API_KEY": "sk-test",
    "MODEL_BASE_URL": "https://api.deepseek.com",
}


def _instantiate_agent(env_overrides, *, patch_summarizer_model=True):
    """构造 agent（mock 掉真实模型与图），返回构造过程中用到的 mock。"""
    fake_loader = Mock()
    fake_loader.build_system_prompt.return_value = "system prompt"

    with _patched_env(env_overrides), patch(
        "langchain_skills.agent.SkillLoader", return_value=fake_loader
    ), patch("langchain_skills.agent.init_chat_model") as init_chat_model, patch(
        "langchain_skills.agent.create_agent", return_value=object()
    ) as create_agent, patch(
        "langchain_skills.agent.SummarizationMiddleware"
    ) as summarization:
        if patch_summarizer_model:
            with patch.object(
                LangChainSkillsAgent, "_build_summarizer_model", return_value=Mock()
            ):
                agent = LangChainSkillsAgent()
        else:
            agent = LangChainSkillsAgent()
    return agent, init_chat_model, create_agent, summarization


class _FakeGraph:
    """只实现 stream_events 用到的 stream/get_state 两个接口。"""

    def __init__(self, interrupts=None):
        self.interrupts = list(interrupts or [])
        self.recorded_inputs: list[object] = []
        self.recorded_contexts: list[object] = []

    def get_state(self, config):
        return SimpleNamespace(interrupts=list(self.interrupts))

    def stream(self, input_payload, config=None, context=None, stream_mode=None):
        self.recorded_inputs.append(input_payload)
        self.recorded_contexts.append(context)
        if isinstance(input_payload, Command):
            self.interrupts = []
        return iter(())


def _fake_interrupt():
    return SimpleNamespace(
        id="int-1",
        value={
            "action_requests": [
                {"name": "bash", "args": {"command": "echo hi"}, "description": "执行 shell 命令"}
            ],
            "review_configs": [
                {"action_name": "bash", "allowed_decisions": ["approve", "edit", "reject"]}
            ],
        },
    )


def test_middleware_stack_is_wired_in_recommended_order():
    _, _, create_agent, summarization = _instantiate_agent(dict(_DEEPSEEK_ENV))

    middleware = create_agent.call_args.kwargs["middleware"]

    # [0] 意图路由：按轮决定是否启用 AgentReply 结构化输出
    assert "_route_by_intent" in type(middleware[0]).__name__
    assert isinstance(middleware[1], ModelRetryMiddleware)
    assert isinstance(middleware[2], ToolErrorMiddleware)
    assert isinstance(middleware[3], HumanInTheLoopMiddleware)
    assert set(middleware[3].interrupt_on) == {"bash", "write_file"}
    assert isinstance(middleware[4], ToolRetryMiddleware)
    assert middleware[4].on_failure == "error"
    assert middleware[5] is summarization.return_value


def test_middleware_can_disable_hitl_and_summarization():
    env = dict(_DEEPSEEK_ENV)
    env.update({"SKILLS_HITL": "false", "SKILLS_SUMMARY_TRIGGER_TOKENS": "0"})
    _, _, create_agent, _ = _instantiate_agent(env)

    middleware = create_agent.call_args.kwargs["middleware"]

    assert [type(item) for item in middleware[1:]] == [
        ModelRetryMiddleware,
        ToolErrorMiddleware,
        ToolRetryMiddleware,
    ]
    assert "_route_by_intent" in type(middleware[0]).__name__


def test_summarizer_model_disables_thinking_and_reuses_provider_settings():
    _, init_chat_model, _, summarization = _instantiate_agent(
        dict(_DEEPSEEK_ENV), patch_summarizer_model=False
    )

    assert init_chat_model.call_count == 2
    kwargs = init_chat_model.call_args_list[1].kwargs
    assert kwargs["temperature"] == 0.2
    assert kwargs["max_tokens"] == 2000
    assert kwargs["model_provider"] == "openai"
    assert kwargs["base_url"] == "https://api.deepseek.com"
    assert kwargs["extra_body"] == {"thinking": {"type": "disabled"}}
    summarization.assert_called_once()


def test_summarization_uses_absolute_token_trigger_and_keep_messages():
    env = dict(_DEEPSEEK_ENV)
    env.update({"SKILLS_SUMMARY_TRIGGER_TOKENS": "5000", "SKILLS_SUMMARY_KEEP_MESSAGES": "10"})
    _, _, _, summarization = _instantiate_agent(env)

    summarization.assert_called_once()
    kwargs = summarization.call_args.kwargs
    assert kwargs["trigger"] == ("tokens", 5000)
    assert kwargs["keep"] == ("messages", 10)


def test_stream_events_emits_interrupt_event():
    agent, *_ = _instantiate_agent(dict(_DEEPSEEK_ENV))
    agent.agent = _FakeGraph([_fake_interrupt()])

    events = list(agent.stream_events("跑个命令", thread_id="t1"))

    assert events[-1]["type"] == "interrupt"
    assert all(event["type"] != "done" for event in events)
    interrupts = events[-1]["interrupts"]
    assert interrupts[0]["id"] == "int-1"
    assert interrupts[0]["value"]["action_requests"][0]["name"] == "bash"


def test_get_pending_interrupts_returns_snapshot_interrupts():
    agent, *_ = _instantiate_agent(dict(_DEEPSEEK_ENV))
    interrupt = _fake_interrupt()
    agent.agent = _FakeGraph([interrupt])

    pending = agent.get_pending_interrupts(thread_id="t1")

    assert pending == [{"id": "int-1", "value": interrupt.value}]


def test_resume_events_sends_command_and_clears_interrupts():
    agent, *_ = _instantiate_agent(dict(_DEEPSEEK_ENV))
    graph = _FakeGraph([_fake_interrupt()])
    agent.agent = graph

    events = list(agent.resume_events([{"type": "approve"}], thread_id="t1"))

    command = graph.recorded_inputs[-1]
    assert isinstance(command, Command)
    assert command.resume == {"decisions": [{"type": "approve"}]}
    assert all(event["type"] != "interrupt" for event in events)
    assert agent.get_pending_interrupts(thread_id="t1") == []


# ---------------------------------------------------------------------------
# 意图路由（任务型才结构化输出）
# ---------------------------------------------------------------------------


def test_classify_by_rules_recognizes_chat_and_task():
    assert _classify_by_rules("你好") is False
    assert _classify_by_rules("解释一下 middleware 是什么") is False
    assert _classify_by_rules("帮我创建 hello.txt 并写入 hi") is True
    assert _classify_by_rules("用 bash 运行 pytest") is True
    assert _classify_by_rules("LangChain 1.5 有什么新变化") is None
    assert _classify_by_rules("   ") is False
    # 知识问句里的"提交/申请"、"列出"等词不应把题目误判成任务型
    assert _classify_by_rules("退货要在哪里提交申请？") is False
    assert _classify_by_rules("token 有效期多久？") is False
    assert _classify_by_rules("提交退货申请后客服多久审核？") is False
    # 明确"用某工具做事"仍应判任务型
    assert _classify_by_rules("用 list_dir 工具列出当前目录的文件") is True


def test_knowledge_gap_suffix_only_when_no_citation_and_no_admission():
    suffix = LangChainSkillsAgent._knowledge_gap_suffix

    assert suffix("退款一般在 1-3 个工作日内到账。").startswith("\n\n（注：")
    assert suffix("详见 product/faq.md 的退货流程。") == ""
    assert suffix("知识库未覆盖微信渠道的退款时效。") == ""
    assert suffix("") == ""


def test_agent_declares_agent_reply_response_format():
    _, _, create_agent, _ = _instantiate_agent(dict(_DEEPSEEK_ENV))

    assert create_agent.call_args.kwargs["response_format"] is AgentReply


def test_intent_router_switches_response_format_by_task_mode():
    agent, *_ = _instantiate_agent(dict(_DEEPSEEK_ENV))
    router = agent._build_middleware()[0]
    seen = {}

    def handler(request):
        seen["response_format"] = request.response_format
        return Mock()

    for task_mode in (False, True):
        request = ModelRequest(
            model=Mock(),
            messages=[],
            response_format=AgentReply,
            runtime=SimpleNamespace(context=SimpleNamespace(task_mode=task_mode)),
        )
        router.wrap_model_call(request, handler)
        assert seen["response_format"] is (AgentReply if task_mode else None)


def test_stream_events_emits_intent_event_for_chat():
    agent, *_ = _instantiate_agent(dict(_DEEPSEEK_ENV))
    agent.agent = _FakeGraph()

    events = list(agent.stream_events("你好", thread_id="intent-chat"))

    assert events[0] == {
        "type": "intent",
        "data": {"is_task": False, "source": "rule", "reason": "关键词规则命中"},
    }


def test_stream_events_uses_classifier_when_rules_are_unsure():
    agent, *_ = _instantiate_agent(dict(_DEEPSEEK_ENV))
    agent.agent = _FakeGraph()
    classifier = Mock()
    classifier.with_structured_output.return_value.invoke.return_value = Intent(
        is_task=True, reason="需要联网查询最新变化"
    )

    with patch.object(
        LangChainSkillsAgent, "_build_classifier_model", return_value=classifier
    ):
        events = list(
            agent.stream_events("LangChain 1.5 有什么新变化", thread_id="intent-llm")
        )

    assert events[0]["data"]["is_task"] is True
    assert events[0]["data"]["source"] == "llm"
    # DeepSeek 不支持 json_schema response_format，必须走 function_calling
    classifier.with_structured_output.assert_called_once_with(
        Intent, method="function_calling"
    )


def test_intent_router_can_be_disabled():
    agent, *_ = _instantiate_agent(dict(_DEEPSEEK_ENV))
    agent.agent = _FakeGraph()

    # 开关在每次分类时读取，所以要在 stream_events 期间保持环境变量
    with _patched_env({"SKILLS_INTENT_ROUTER": "false"}):
        events = list(agent.stream_events("帮我创建 hello.txt", thread_id="intent-off"))

    assert events[0]["data"] == {
        "is_task": False,
        "source": "disabled",
        "reason": "意图路由已关闭，按聊天型处理",
    }


def test_intent_router_falls_back_to_chat_on_classifier_error():
    agent, *_ = _instantiate_agent(dict(_DEEPSEEK_ENV))
    agent.agent = _FakeGraph()
    classifier = Mock()
    classifier.with_structured_output.side_effect = RuntimeError("boom")

    with patch.object(
        LangChainSkillsAgent, "_build_classifier_model", return_value=classifier
    ):
        events = list(
            agent.stream_events("LangChain 1.5 有什么新变化", thread_id="intent-error")
        )

    assert events[0]["data"]["is_task"] is False
    assert events[0]["data"]["source"] == "fallback"


def test_resume_events_reuses_previous_task_mode():
    agent, *_ = _instantiate_agent(dict(_DEEPSEEK_ENV))
    graph = _FakeGraph([_fake_interrupt()])
    agent.agent = graph
    agent._task_modes["t1"] = True

    list(agent.resume_events([{"type": "approve"}], thread_id="t1"))

    assert graph.recorded_contexts[-1].task_mode is True


# ---------------------------------------------------------------------------
# 长期记忆（store + remember/recall）
# ---------------------------------------------------------------------------


def test_long_term_memory_tools_are_registered():
    names = {tool.name for tool in ALL_TOOLS}

    assert {"remember", "recall"} <= names


def test_remember_and_recall_round_trip_within_same_user():
    from langgraph.store.memory import InMemoryStore

    store = InMemoryStore()

    saved = _remember_fact(store, "user-a", "用户偏好使用 Java，不使用 Python。")
    facts = _recall_facts(store, "user-a", "用什么语言")

    assert saved.startswith("[OK]")
    assert len(facts) == 1
    assert "Java" in facts[0]["content"]
    assert facts[0]["created_at"]


def test_recall_isolated_between_users():
    from langgraph.store.memory import InMemoryStore

    store = InMemoryStore()
    _remember_fact(store, "user-a", "用户偏好使用 Java。")

    assert _recall_facts(store, "user-b", "Java") == []


def test_agent_attaches_store_and_memory_user():
    env = dict(_DEEPSEEK_ENV)
    env["SKILLS_MEMORY_USER"] = "alice"

    agent, _, create_agent, _ = _instantiate_agent(env)

    from langgraph.store.memory import InMemoryStore

    assert isinstance(agent.store, InMemoryStore)
    assert agent.context.user_id == "alice"
    assert create_agent.call_args.kwargs["store"] is agent.store
