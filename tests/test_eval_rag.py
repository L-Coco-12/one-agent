"""RAG 检索评估脚本测试（离线：不连 Chroma、不调智谱）。"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
for path in (str(SCRIPTS_DIR), str(PROJECT_ROOT / "src")):
    if path not in sys.path:
        sys.path.insert(0, path)

import eval_rag  # noqa: E402


def item(source: str, section: str, score: float) -> dict:
    return {"source": source, "section": section, "score": score, "text": f"{source} {section}"}


def question(qid: str, *, qtype: str = "single_hop", sections: list[str] | None = None,
             sources: list[str] | None = None, require_all: bool | None = None) -> dict:
    q = {
        "id": qid,
        "question": f"问题 {qid}",
        "type": qtype,
        "expected_sources": sources if sources is not None else ["product/faq.md"],
        "expected_sections": sections if sections is not None else ["退换货政策常见问题 > 退货流程"],
        "require_all_sections": require_all if require_all is not None else qtype == "multi_hop",
        "must_contain": [],
    }
    return q


# --------------------------------------------------------------------- 加载校验


def write_questions(tmp_path: Path, lines: list[dict]) -> Path:
    path = tmp_path / "questions.jsonl"
    path.write_text("\n".join(json.dumps(line, ensure_ascii=False) for line in lines), encoding="utf-8")
    return path


def test_load_questions_ok_and_defaults(tmp_path):
    path = write_questions(
        tmp_path,
        [
            {"id": "q1", "question": "问题1", "type": "single_hop",
             "expected_sources": ["a.md"], "expected_sections": ["s1"]},
            {"id": "q2", "question": "问题2", "type": "no_answer"},
        ],
    )
    questions = eval_rag.load_questions(path)
    assert [q["id"] for q in questions] == ["q1", "q2"]
    assert questions[0]["require_all_sections"] is False
    assert questions[0]["must_contain"] == []
    assert questions[1]["expected_sections"] == []


def test_load_questions_rejects_bad_type(tmp_path):
    path = write_questions(tmp_path, [{"id": "q1", "question": "问题", "type": "weird"}])
    with pytest.raises(RuntimeError, match="type 非法"):
        eval_rag.load_questions(path)


def test_load_questions_rejects_missing_sections(tmp_path):
    path = write_questions(tmp_path, [{"id": "q1", "question": "问题", "type": "single_hop"}])
    with pytest.raises(RuntimeError, match="expected_sources"):
        eval_rag.load_questions(path)


def test_load_questions_rejects_duplicate_id(tmp_path):
    line = {"id": "q1", "question": "问题", "type": "no_answer"}
    path = write_questions(tmp_path, [line, dict(line)])
    with pytest.raises(RuntimeError, match="id 重复"):
        eval_rag.load_questions(path)


# --------------------------------------------------------------------- 匹配与指标


def test_section_matches_prefix():
    assert eval_rag.section_matches("A > B", "A > B")
    assert eval_rag.section_matches("A > B > C", "A > B")
    assert not eval_rag.section_matches("A > C", "A > B")
    assert not eval_rag.section_matches("", "A > B")


def test_evaluate_perfect_hits():
    questions = [question("q1"), question("q2", sections=["API 鉴权说明 > 令牌有效期"], sources=["api/auth.md"])]
    results = {
        "问题 q1": [item("product/faq.md", "退换货政策常见问题 > 退货流程", 0.8)],
        "问题 q2": [item("api/auth.md", "API 鉴权说明 > 令牌有效期", 0.9)],
    }
    result = eval_rag.evaluate_questions(questions, lambda q: results[q], top_k=6)
    metrics = result["metrics"]
    assert metrics["hit@1"] == 1.0
    assert metrics["hit@4"] == 1.0
    assert metrics["mrr"] == 1.0
    assert metrics["precision_at_k"] == round(1 / 6, 4)


def test_evaluate_miss_and_rank():
    questions = [question("q1"), question("q2")]
    def retrieve(q: str):
        if q == "问题 q1":
            return [
                item("product/faq.md", "退换货政策常见问题 > 换货规则", 0.7),
                item("product/faq.md", "退换货政策常见问题 > 退货流程", 0.6),
            ]
        return [item("api/auth.md", "API 鉴权说明 > 令牌有效期", 0.9)]

    result = eval_rag.evaluate_questions(questions, retrieve, top_k=6)
    metrics = result["metrics"]
    assert metrics["hit@1"] == 0.0  # q1 第一条不对，q2 完全未命中
    assert metrics["hit@4"] == 0.5  # 只有 q1 在前 4 命中
    assert metrics["mrr"] == round((1 / 2 + 0) / 2, 4)
    assert metrics["precision_at_k"] == round((1 / 6 + 0) / 2, 4)


def test_evaluate_multi_hop_requires_all_sections():
    q = question(
        "q1",
        qtype="multi_hop",
        sections=["API 鉴权说明 > 令牌有效期", "API 鉴权说明 > 常见错误码"],
        sources=["api/auth.md"],
    )
    only_one = lambda _q: [item("api/auth.md", "API 鉴权说明 > 令牌有效期", 0.9)]  # noqa: E731
    partial = eval_rag.evaluate_questions([q], only_one, top_k=6)["metrics"]
    assert partial["hit@1"] == 0.0
    assert partial["multi_hop_full_coverage"] == 0.0

    both = lambda _q: [  # noqa: E731
        item("api/auth.md", "API 鉴权说明 > 令牌有效期", 0.9),
        item("api/auth.md", "API 鉴权说明 > 常见错误码", 0.7),
    ]
    full = eval_rag.evaluate_questions([q], both, top_k=6)
    assert full["metrics"]["hit@4"] == 1.0
    assert full["metrics"]["multi_hop_full_coverage"] == 1.0
    assert full["rows"][0]["hit_rank"] == 2  # 最后一个必需 section 的排名


def test_evaluate_no_answer_threshold():
    questions = [
        question("q1", qtype="no_answer", sections=[], sources=[]),
        question("q2", qtype="no_answer", sections=[], sources=[]),
    ]
    scores = {"问题 q1": 0.2, "问题 q2": 0.8}
    retrieve = lambda q: [item("product/faq.md", "退换货政策常见问题 > 退货流程", scores[q])]  # noqa: E731
    result = eval_rag.evaluate_questions(questions, retrieve, top_k=6)
    assert result["metrics"]["no_answer_pass_rate"] == 0.5
    assert result["rows"][0]["passed"] is True
    assert result["rows"][1]["passed"] is False


def test_render_markdown_contains_metrics():
    questions = [question("q1")]
    retrieve = lambda _q: [item("product/faq.md", "退换货政策常见问题 > 退货流程", 0.8)]  # noqa: E731
    result = eval_rag.evaluate_questions(questions, retrieve, top_k=6)
    markdown = eval_rag.render_markdown(result, questions_path=Path("evals/rag_questions.jsonl"), collection="knowledge")
    assert "# RAG 检索评估报告" in markdown
    assert "| Hit@1 | 100.0% |" in markdown
    assert "| q1 | single_hop | 1 |" in markdown
