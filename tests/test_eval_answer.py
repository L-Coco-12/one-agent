"""RAG 生成层评估脚本的离线测试（不需要 Chroma / 智谱 / 真实模型）。"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import eval_answer as evaluator  # noqa: E402


def _search_result_event(*lines: str, name: str = "search_docs") -> dict:
    return {
        "type": "tool_result",
        "name": name,
        "success": True,
        "content": "[OK] Found 2 relevant chunk(s) (higher score = more similar):\n"
        + "\n".join(lines),
    }


def _tool_call_event(name: str = "search_docs") -> dict:
    return {"type": "tool_call", "name": name, "args": {"query": "q"}, "id": "call-1"}


def _question(
    qid: str = "q1",
    qtype: str = "single_hop",
    sources: list[str] | None = None,
    sections: list[str] | None = None,
    must_contain: list[str] | None = None,
    require_all_sections: bool = False,
) -> dict:
    return {
        "id": qid,
        "question": "退货要多久才能退款？",
        "type": qtype,
        "expected_sources": sources or [],
        "expected_sections": sections or [],
        "must_contain": must_contain or [],
        "require_all_sections": require_all_sections,
    }


def test_extract_cited_sources_dedupes_and_strips_punctuation() -> None:
    answer = "依据 product/faq.md 与 faq.md（见 api/auth.md。），以及 product/faq.md 再次。"
    assert evaluator.extract_cited_sources(answer) == ["product/faq.md", "faq.md", "api/auth.md"]


def test_answer_contains_tolerates_markdown_and_regex() -> None:
    answer = "换货后，原订单的**退货入口就会关闭**，无法再申请退货。"
    assert evaluator.answer_contains(answer, "退货入口关闭") is False
    assert evaluator.answer_contains(answer, "退货入口.{0,6}关闭") is True
    assert evaluator.answer_contains("运费由**平台承担**。", "平台承担") is True
    assert evaluator.answer_contains("退款 3–7 个工作日到账。", "3–7 个工作日") is True


def test_extract_cited_sources_ignores_plain_text() -> None:
    assert evaluator.extract_cited_sources("没有文件引用，只有一句回答。") == []


def test_parse_search_results_reads_score_section_and_page() -> None:
    event = _search_result_event(
        "[1] score=0.6903 source=product/faq.md section=退换货政策常见问题 > 退货流程 page=2",
        "[2] score=0.4120 source=api/auth.md section=API 鉴权说明 > 令牌有效期",
    )
    parsed = evaluator.parse_search_results([event])
    assert len(parsed) == 2
    assert parsed[0]["index"] == 1
    assert parsed[0]["score"] == 0.6903
    assert parsed[0]["source"] == "product/faq.md"
    assert parsed[0]["section"] == "退换货政策常见问题 > 退货流程"
    assert parsed[0]["page"] == 2
    assert parsed[1]["section"] == "API 鉴权说明 > 令牌有效期"
    assert parsed[1]["page"] is None


def test_parse_search_results_keeps_chunk_content_for_judge() -> None:
    event = _search_result_event(
        "[1] score=0.6903 source=product/faq.md section=退货流程 page=2",
        "签收后 7 天内可申请退货，质量问题由平台承担运费。",
        "",
        "[2] score=0.4120 source=api/auth.md section=令牌有效期",
        "access_token 有效期 2 小时，refresh_token 有效期 30 天。",
        "Cite the source (and section/page when available) in your answer.",
    )
    parsed = evaluator.parse_search_results([event])
    assert len(parsed) == 2
    assert "7 天内可申请退货" in parsed[0]["content"]
    assert "refresh_token 有效期 30 天" in parsed[1]["content"]
    assert "Cite the source" not in parsed[1]["content"]


def test_parse_search_results_ignores_other_tools_and_failures() -> None:
    assert evaluator.parse_search_results([_search_result_event("x", name="bash")]) == []
    failed = _search_result_event("[1] score=0.5 source=a.md section=b > c")
    failed["success"] = False
    assert evaluator.parse_search_results([failed]) == []


def test_resolve_citation_matches_basename_and_relative_path() -> None:
    known = {"product/faq.md", "api/auth.md"}
    assert evaluator.resolve_citation("product/faq.md", known) == "product/faq.md"
    assert evaluator.resolve_citation("faq.md", known) == "product/faq.md"
    assert evaluator.resolve_citation("./api/auth.md", known) == "api/auth.md"
    assert evaluator.resolve_citation("missing.md", known) is None


def test_resolve_citation_accepts_existing_repo_file() -> None:
    # 与 agent 的引用硬校验一致：仓库内真实存在的文件不算编造
    assert evaluator.resolve_citation("KNOWLEDGE_SOURCES.md", set()) == "KNOWLEDGE_SOURCES.md"


def test_answer_contains_is_whitespace_and_case_insensitive() -> None:
    assert evaluator.answer_contains("退款一般在 1-3 个  工作日", "1-3 个 工作日")
    assert evaluator.answer_contains("Token 有效期 2 小时", "token")
    assert not evaluator.answer_contains("无关回答", "退款")


@pytest.mark.parametrize(
    ("answer", "expected"),
    [
        ("知识库未覆盖这个问题。", True),
        ("该信息未提及，无法确定。", True),
        ("我不清楚，建议咨询客服。", True),
        ("退款一般 1-3 个工作日到账。", False),
    ],
)
def test_admits_no_answer(answer: str, expected: bool) -> None:
    assert evaluator.admits_no_answer(answer) is expected


def test_evaluate_question_answerable_passes_with_citation_and_facts() -> None:
    events = [
        {"type": "intent", "data": {"is_task": False, "source": "rule"}},
        _tool_call_event(),
        _search_result_event(
            "[1] score=0.6903 source=product/faq.md section=退换货政策常见问题 > 退货流程 page=1"
        ),
        {"type": "text", "content": "退款一般在 1-3 个工作日内到账。来源：product/faq.md。"},
        {"type": "done", "response": "退款一般在 1-3 个工作日内到账。来源：product/faq.md。"},
    ]
    result = evaluator.evaluate_question(
        _question(sources=["product/faq.md"], sections=["退换货政策常见问题 > 退货流程"], must_contain=["1-3"]),
        events,
        known_sources={"product/faq.md", "api/auth.md"},
    )
    assert result["citation_present"] is True
    assert result["expected_source_cited"] is True
    assert result["must_contain_ok"] is True
    assert result["retrieval_hit"] is True
    assert result["fabricated_sources"] == []
    assert result["passed"] is True


def test_evaluate_question_flags_fabricated_citation() -> None:
    events = [
        _tool_call_event(),
        _search_result_event(
            "[1] score=0.6903 source=product/faq.md section=退换货政策常见问题 > 退货流程"
        ),
        {"type": "text", "content": "见 product/faq.md 与 ops/refund_policy.md，退款 1-3 天。"},
        {"type": "done", "response": "见 product/faq.md 与 ops/refund_policy.md，退款 1-3 天。"},
    ]
    result = evaluator.evaluate_question(
        _question(sources=["product/faq.md"], sections=["退换货政策常见问题 > 退货流程"]),
        events,
        known_sources={"product/faq.md"},
    )
    assert result["fabricated_sources"] == ["ops/refund_policy.md"]
    assert result["passed"] is False


def test_structured_agent_reply_sources_count_as_citations() -> None:
    """任务型走 AgentReply 时，引用写在 structured.sources 里也要算。"""
    events = [
        {"type": "tool_call", "name": "AgentReply"},
        {
            "type": "done",
            "response": "",
            "structured": {
                "summary": "退货入口在售后页，审核一般 1-3 个工作日。",
                "sources": ["product/faq.md"],
            },
        },
    ]
    result = evaluator.evaluate_question(
        _question(
            sources=["product/faq.md"],
            sections=["退换货政策常见问题 > 退货流程"],
            must_contain=["1-3"],
        ),
        events,
        known_sources={"product/faq.md"},
    )
    assert result["structured_delivery"] is True
    assert result["citation_present"] is True
    assert result["expected_source_cited"] is True
    assert result["passed"] is True


def test_evaluate_question_no_answer_passes_when_model_admits_gap() -> None:
    events = [
        _tool_call_event(),
        _search_result_event("[1] score=0.61 source=product/faq.md section=退换货政策常见问题 > 售后范围"),
        {"type": "text", "content": "知识库未覆盖微信渠道的退款时效，无法确认。"},
        {"type": "done", "response": "知识库未覆盖微信渠道的退款时效，无法确认。"},
    ]
    result = evaluator.evaluate_question(
        _question(qid="q19", qtype="no_answer"),
        events,
        known_sources={"product/faq.md"},
    )
    assert result["admits_no_answer"] is True
    assert result["retrieval_hit"] is None
    assert result["passed"] is True


def test_evaluate_question_no_answer_fails_when_model_invents() -> None:
    events = [
        _tool_call_event(),
        _search_result_event("[1] score=0.61 source=product/faq.md section=退换货政策常见问题 > 售后范围"),
        {"type": "text", "content": "微信退款一般 1 个工作日到账。"},
        {"type": "done", "response": "微信退款一般 1 个工作日到账。"},
    ]
    result = evaluator.evaluate_question(_question(qid="q19", qtype="no_answer"), events, known_sources={"product/faq.md"})
    assert result["admits_no_answer"] is False
    assert result["passed"] is False


def test_evaluate_question_multi_hop_requires_all_expected_sections() -> None:
    events = [
        _tool_call_event(),
        _search_result_event(
            "[1] score=0.72 source=api/auth.md section=API 鉴权说明 > 令牌有效期",
            "[2] score=0.55 source=api/auth.md section=API 鉴权说明 > 常见错误码",
        ),
        {"type": "text", "content": "令牌 2 小时过期（api/auth.md）。"},
        {"type": "done", "response": "令牌 2 小时过期（api/auth.md）。"},
    ]
    question = _question(
        qid="q17",
        qtype="multi_hop",
        sources=["api/auth.md"],
        sections=["API 鉴权说明 > 令牌有效期", "API 鉴权说明 > 常见错误码"],
        require_all_sections=True,
    )
    result = evaluator.evaluate_question(question, events, known_sources={"api/auth.md"})
    assert result["retrieval_hit"] is True

    events[1] = _search_result_event("[1] score=0.72 source=api/auth.md section=API 鉴权说明 > 令牌有效期")
    result = evaluator.evaluate_question(question, events, known_sources={"api/auth.md"})
    assert result["retrieval_hit"] is False


def test_final_answer_text_falls_back_to_structured_summary() -> None:
    events = [
        {"type": "done", "response": "", "structured": {"summary": "已列出文件。", "followup_reason": "需要确认目录"}},
    ]
    assert evaluator.final_answer_text(events) == "已列出文件。\n需要确认目录"


def test_summarize_computes_rates() -> None:
    results = [
        {"type": "single_hop", "passed": True, "citation_present": True, "expected_source_cited": True, "must_contain_ok": True, "retrieval_hit": True, "fabricated_sources": [], "tool_calls": ["search_docs"]},
        {"type": "single_hop", "passed": False, "citation_present": False, "expected_source_cited": False, "must_contain_ok": False, "retrieval_hit": False, "fabricated_sources": ["fake.md"], "tool_calls": []},
        {"type": "no_answer", "passed": True, "citation_present": True, "expected_source_cited": False, "must_contain_ok": None, "retrieval_hit": None, "fabricated_sources": [], "tool_calls": ["search_docs", "search_docs"], "admits_no_answer": True},
    ]
    summary = evaluator.summarize(results)
    assert summary["questions"] == 3
    assert summary["answerable"] == 2
    assert summary["no_answer"] == 1
    assert summary["pass_rate"] == pytest.approx(2 / 3, abs=1e-4)
    assert summary["expected_source_cited_rate"] == pytest.approx(0.5)
    assert summary["no_answer_admit_rate"] == pytest.approx(1.0)
    assert summary["fabricated_source_questions"] == 1
    assert summary["avg_tool_calls"] == pytest.approx(1.0)


def test_render_markdown_contains_summary_and_rows() -> None:
    results = [
        {
            "id": "q1",
            "type": "single_hop",
            "passed": True,
            "citation_present": True,
            "expected_source_cited": True,
            "must_contain_ok": True,
            "retrieval_hit": True,
            "fabricated_sources": [],
            "admits_no_answer": None,
        }
    ]
    summary = evaluator.summarize(
        [
            {
                "type": "single_hop",
                "passed": True,
                "citation_present": True,
                "expected_source_cited": True,
                "must_contain_ok": True,
                "retrieval_hit": True,
                "fabricated_sources": [],
                "tool_calls": ["search_docs"],
            }
        ]
    )
    markdown = evaluator.render_markdown(results, summary, judge=False)
    assert "# RAG 生成层评估报告" in markdown
    assert "| 通过率 |" in markdown
    assert "| q1 | single_hop | ✅ |" in markdown


def test_main_runs_with_injected_run_one(tmp_path: Path) -> None:
    questions_path = tmp_path / "questions.jsonl"
    questions_path.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "id": "q1",
                        "question": "退货要多久才能退款？",
                        "type": "single_hop",
                        "expected_sources": ["product/faq.md"],
                        "expected_sections": ["退换货政策常见问题 > 退货流程"],
                        "must_contain": ["1-3"],
                    },
                    ensure_ascii=False,
                ),
                json.dumps(
                    {
                        "id": "q19",
                        "question": "微信支付退款多久到账？",
                        "type": "no_answer",
                    },
                    ensure_ascii=False,
                ),
            ]
        ),
        encoding="utf-8",
    )
    report_path = tmp_path / "answer_report.md"

    def fake_run_one(question: dict) -> list[dict]:
        if question["id"] == "q1":
            return [
                _tool_call_event(),
                _search_result_event(
                    "[1] score=0.6903 source=product/faq.md section=退换货政策常见问题 > 退货流程"
                ),
                {"type": "text", "content": "退款 1-3 个工作日到账。来源：product/faq.md。"},
                {"type": "done", "response": "退款 1-3 个工作日到账。来源：product/faq.md。"},
            ]
        return [
            {"type": "text", "content": "知识库未覆盖微信退款时效。"},
            {"type": "done", "response": "知识库未覆盖微信退款时效。"},
        ]

    exit_code = evaluator.main(
        ["--questions", str(questions_path), "--report", str(report_path)],
        run_one=fake_run_one,
    )
    assert exit_code == 0
    markdown = report_path.read_text(encoding="utf-8")
    assert "题目数：2" in markdown
    assert "通过率 | 100.0%" in markdown


def test_main_returns_one_when_questions_missing(tmp_path: Path) -> None:
    exit_code = evaluator.main(
        ["--questions", str(tmp_path / "missing.jsonl"), "--report", str(tmp_path / "r.md")]
    )
    assert exit_code == 1


def test_knowledge_sources_scans_dir(tmp_path: Path) -> None:
    (tmp_path / "api").mkdir()
    (tmp_path / "api" / "auth.md").write_text("x", encoding="utf-8")
    (tmp_path / "ignore.png").write_bytes(b"x")
    assert evaluator.knowledge_sources(tmp_path) == {"api/auth.md"}


def test_normalize_cited_source_strips_section_path() -> None:
    assert (
        evaluator.normalize_cited_source(
            "fastapi/tutorial-cors.md > CORS（跨域资源共享） > 使用 CORSMiddleware"
        )
        == "fastapi/tutorial-cors.md"
    )
    assert (
        evaluator.normalize_cited_source("product/faq.md — 退换货政策 > 退货流程")
        == "product/faq.md"
    )
    assert evaluator.normalize_cited_source("product/faq.md") == "product/faq.md"


def test_section_suffixed_structured_sources_resolve_to_real_files() -> None:
    known = {"fastapi/tutorial-cors.md", "fastapi/tutorial-middleware.md"}
    events = [
        {"type": "tool_call", "name": "search_docs", "args": {"query": "q"}, "id": "c1"},
        _search_result_event(
            "[1] score=0.70 source=fastapi/tutorial-cors.md section=CORS 预检请求",
            "[2] score=0.60 source=fastapi/tutorial-middleware.md section=多个中间件的执行顺序",
        ),
        {"type": "tool_call", "name": "AgentReply", "args": {}, "id": "c2"},
        {
            "type": "done",
            "response": "",
            "structured": {
                "summary": "CORS 预检先经过 CORS 中间件，再进入其它中间件。",
                "sources": [
                    "fastapi/tutorial-cors.md > CORS（跨域资源共享） > CORS 预检请求",
                    "fastapi/tutorial-middleware.md > 中间件 > 多个中间件的执行顺序",
                ],
            },
        },
    ]
    question = {
        "id": "q40",
        "type": "multi_hop",
        "expected_sources": ["fastapi/tutorial-cors.md", "fastapi/tutorial-middleware.md"],
        "expected_sections": ["CORS 预检请求", "多个中间件的执行顺序"],
        "require_all_sections": True,
        "must_contain": [],
    }

    result = evaluator.evaluate_question(question, events, known_sources=known)

    assert result["fabricated_sources"] == []
    assert result["expected_source_cited"] is True
    assert result["passed"] is True
