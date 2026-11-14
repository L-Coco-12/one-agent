"""RAG 生成评估：端到端跑 agent，给「回答层」打分。

检索层评估（``eval_rag.py``）只回答「资料找没找到」；本脚本回答「答得对不对」：

- **引用真实性**：回答里引用的 source 是否真的来自本轮检索结果/知识库，而不是编造。
- **期望来源**：可回答题是否引用了标注的 ``expected_sources``。
- **关键事实**：``must_contain`` 里的字符串是否出现在回答中。
- **无答案识别**：知识库没有覆盖的问题，模型是否明确说「未覆盖/不知道」，而不是硬编。
- **可选 LLM-as-judge**：``--judge`` 时用一个低温度小模型按 rubric 评忠实度与覆盖度
  （只作辅助参考，上线前建议人工抽 5–10 题校准）。

用法::

    # 前置：.env 已配置 MODEL_*/DEEPSEEK_* 与 ZHIPU_API_KEY，Chroma 已启动且索引已构建
    uv run python scripts/eval_answer.py
    uv run python scripts/eval_answer.py --limit 5 --json
    uv run python scripts/eval_answer.py --judge --report evals/answer_report.md
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
for _path in (SCRIPT_DIR, PROJECT_ROOT / "src"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(PROJECT_ROOT / ".env")

from eval_rag import (  # noqa: E402
    DEFAULT_QUESTIONS,
    is_relevant,
    load_questions,
    section_matches,
)

DEFAULT_REPORT = PROJECT_ROOT / "evals" / "answer_report.md"
DEFAULT_KNOWLEDGE_DIR = PROJECT_ROOT / "knowledge"
DEFAULT_LIMIT = 0  # 0 = 全部

# 回答里出现这些说法，视为「承认知识库未覆盖」。
ADMIT_PATTERNS = (
    "未覆盖",
    "没有覆盖",
    "未提及",
    "没有提及",
    "没有相关内容",
    "没有找到相关",
    "未找到相关",
    "知识库中没有",
    "知识库没有",
    "知识库未",
    "无法从知识库",
    "无法回答",
    "无法确定",
    "不清楚",
    "不知道",
)

# 回答中的文件引用：例如 product/faq.md、faq.md。
SOURCE_PATTERN = re.compile(r"[A-Za-z0-9_\-./]+\.(?:md|markdown|txt|pdf)", re.IGNORECASE)
# search_docs 的 tool_result 行格式：[1] score=0.6903 source=product/faq.md section=... page=3
RESULT_LINE_PATTERN = re.compile(
    r"\[(?P<index>\d+)\]\s+score=(?P<score>\d+\.\d+)\s+source=(?P<source>\S+)\s+section=(?P<section>.*?)(?:\s+page=(?P<page>\d+))?$"
)

EventList = list[dict[str, Any]]
RunOne = Callable[[dict[str, Any]], EventList]


# --------------------------------------------------------------------- 文本工具


def normalize_text(text: str | None) -> str:
    """压掉空白与大小写差异，便于做「关键事实」比对。"""
    return " ".join((text or "").lower().split())


def answer_contains(answer: str, needle: str) -> bool:
    """判断回答里是否覆盖某条关键事实。

    ``needle`` 既可以是普通字符串（忽略空白、大小写和 ``**`` 等 Markdown 装饰），
    也可以写正则（例如 ``退货入口.{0,6}关闭``），用于容忍模型改述。
    """
    compact_answer = re.sub(r"[*_`~\s]+", "", answer or "").lower()
    compact_needle = re.sub(r"[*_`~\s]+", "", needle or "").lower()
    if not compact_needle:
        return True
    if compact_needle in compact_answer:
        return True
    try:
        return re.search(compact_needle, compact_answer, re.IGNORECASE) is not None
    except re.error:
        return False


def extract_cited_sources(answer: str | None) -> list[str]:
    """从回答里抽取形如 ``product/faq.md`` 的引用，去重保持出现顺序。"""
    seen: list[str] = []
    for match in SOURCE_PATTERN.finditer(answer or ""):
        source = match.group(0).strip("。，,.;；:：()（）[]【】<>'\"")
        if source and source not in seen:
            seen.append(source)
    return seen


def parse_search_results(tool_results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """把 search_docs 的 tool_result 文本解析回结构化结果。"""
    parsed: list[dict[str, Any]] = []
    for event in tool_results:
        if event.get("name") != "search_docs" or not event.get("success", True):
            continue
        for line in str(event.get("content") or "").splitlines():
            match = RESULT_LINE_PATTERN.search(line.strip())
            if not match:
                continue
            parsed.append(
                {
                    "index": int(match.group("index")),
                    "score": float(match.group("score")),
                    "source": match.group("source").strip(),
                    "section": match.group("section").strip(),
                    "page": int(match.group("page")) if match.group("page") else None,
                }
            )
    return parsed


CITATION_SUFFIX_PATTERN = re.compile(
    r"^(.*?\.(?:md|markdown|txt|pdf))(?=$|[\s>—\-:：,，;；|])", re.IGNORECASE
)


def normalize_cited_source(cited: str) -> str:
    """把"文件路径 + 章节路径"归一成文件路径。

    模型常见的引用写法有两种：
    - ``product/faq.md``
    - ``fastapi/tutorial-cors.md > CORS（跨域资源共享） > 使用 CORSMiddleware``
    - ``product/faq.md — 退换货政策 > 退货流程``

    后两种引用是合法的，只是带了章节信息；比对前要截到文件后缀，
    否则会被误判为"编造引用"。
    """
    text = str(cited or "").strip()
    match = CITATION_SUFFIX_PATTERN.match(text)
    if match:
        text = match.group(1)
    return text.strip().strip("。，,.;；:：()（）[]【】<>'\"“”")


def resolve_citation(cited: str, known_sources: set[str]) -> str | None:
    """把回答里的引用对回知识库路径（都允许只写文件名）。"""
    cited_clean = normalize_cited_source(cited).lstrip("./")
    if cited_clean in known_sources:
        return cited_clean
    basename = cited_clean.rsplit("/", 1)[-1]
    for known in known_sources:
        if known == cited_clean:
            return known
        if known.rsplit("/", 1)[-1] == basename:
            return known
    return None


def admits_no_answer(answer: str | None) -> bool:
    text = answer or ""
    return any(pattern in text for pattern in ADMIT_PATTERNS)


# --------------------------------------------------------------------- 单题打分


def final_answer_text(events: EventList) -> str:
    """取最终回答：优先 done.response；任务型没有正文时回退 structured.summary。"""
    text_parts = [
        str(event.get("content") or "")
        for event in events
        if event.get("type") == "text"
    ]
    answer = "".join(text_parts).strip()
    done = next((event for event in reversed(events) if event.get("type") == "done"), None)
    if done:
        response = str(done.get("response") or "").strip()
        if response:
            answer = response if not answer else response
        structured = done.get("structured")
        if not answer and isinstance(structured, dict):
            answer = str(structured.get("summary") or "").strip()
            reason = structured.get("followup_reason")
            if reason:
                answer = f"{answer}\n{reason}".strip()
    return answer


def evaluate_question(
    question: dict[str, Any],
    events: EventList,
    *,
    known_sources: set[str],
) -> dict[str, Any]:
    """给一道题的回答打分（纯函数，便于离线测试）。"""
    answer = final_answer_text(events)
    tool_calls = [event for event in events if event.get("type") == "tool_call"]
    tool_results = [event for event in events if event.get("type") == "tool_result"]
    intent = next(
        (event.get("data") for event in events if event.get("type") == "intent"),
        None,
    )
    intent_data = intent if isinstance(intent, dict) else {}
    structured_delivery = any(
        str(event.get("name") or "") == "AgentReply" for event in tool_calls
    )
    done = next((event for event in reversed(events) if event.get("type") == "done"), None)
    structured = done.get("structured") if isinstance(done, dict) else None
    structured_sources = (
        [str(item).strip() for item in (structured.get("sources") or [])]
        if isinstance(structured, dict)
        else []
    )
    retrieved = parse_search_results(tool_results)
    retrieved_sources = {item["source"] for item in retrieved}

    cited = extract_cited_sources(answer)
    # 任务型走 AgentReply 时，来源写在结构化字段 sources 里，也要算作引用
    for source in structured_sources:
        if source and source not in cited:
            cited.append(source)
    allowed_sources = set(known_sources) | retrieved_sources
    resolved_cited = {resolve_citation(source, allowed_sources) for source in cited}
    fabricated = sorted(
        source for source in cited if resolve_citation(source, allowed_sources) is None
    )

    expected_sources = set(question.get("expected_sources") or [])
    expected_cited = sorted(
        {
            resolved
            for source in cited
            for resolved in [resolve_citation(source, allowed_sources)]
            if resolved in expected_sources
        }
    )

    if question.get("type") == "no_answer":
        retrieval_hit = None
        must_contain_ok = None
    else:
        if question.get("require_all_sections"):
            retrieval_hit = all(
                any(
                    item.get("source") in expected_sources
                    and section_matches(item.get("section", ""), expected_section)
                    for item in retrieved
                )
                for expected_section in (question.get("expected_sections") or [])
            )
        else:
            retrieval_hit = any(is_relevant(item, question) for item in retrieved)
        must_contain_ok = (
            all(answer_contains(answer, fact) for fact in question.get("must_contain") or [])
            if question.get("must_contain")
            else None
        )

    if question.get("type") == "no_answer":
        passed = admits_no_answer(answer) and not fabricated
    else:
        passed = bool(expected_cited) and must_contain_ok is not False and not fabricated

    return {
        "id": question.get("id"),
        "type": question.get("type"),
        "question": question.get("question"),
        "answer": answer,
        "tool_calls": [str(event.get("name") or "") for event in tool_calls],
        "retrieved": retrieved,
        "retrieved_sources": sorted(retrieved_sources),
        "cited_sources": cited,
        "resolved_cited_sources": sorted(item for item in resolved_cited if item),
        "fabricated_sources": fabricated,
        "expected_sources": sorted(expected_sources),
        "expected_source_cited": bool(expected_cited),
        "citation_present": bool(cited),
        "intent_source": intent_data.get("source"),
        "task_mode": bool(intent_data.get("is_task")) if intent_data else None,
        "structured_delivery": structured_delivery,
        "must_contain_ok": must_contain_ok,
        "retrieval_hit": retrieval_hit,
        "admits_no_answer": admits_no_answer(answer) if question.get("type") == "no_answer" else None,
        "passed": passed,
    }


# --------------------------------------------------------------------- 汇总


def summarize(results: list[dict[str, Any]]) -> dict[str, Any]:
    answerable = [item for item in results if item["type"] != "no_answer"]
    no_answer = [item for item in results if item["type"] == "no_answer"]

    def rate(items: list[dict[str, Any]], key: str) -> float:
        if not items:
            return 0.0
        return round(sum(1 for item in items if item.get(key)) / len(items), 4)

    total = len(results)
    return {
        "questions": total,
        "answerable": len(answerable),
        "no_answer": len(no_answer),
        "pass_rate": rate(results, "passed"),
        "citation_present_rate": rate(results, "citation_present"),
        "expected_source_cited_rate": rate(answerable, "expected_source_cited"),
        "must_contain_rate": rate(answerable, "must_contain_ok"),
        "retrieval_hit_rate": rate(answerable, "retrieval_hit"),
        "fabricated_source_questions": sum(1 for item in results if item["fabricated_sources"]),
        "no_answer_admit_rate": rate(no_answer, "admits_no_answer"),
        "task_mode_questions": sum(1 for item in results if item.get("task_mode")),
        "avg_tool_calls": round(
            sum(len(item["tool_calls"]) for item in results) / total, 2
        ) if total else 0.0,
    }


def aggregate_runs(runs: list[dict[str, Any]]) -> dict[str, Any]:
    """把同一题多次重复（--repeat）合并成题级结论：严格通过=每次都通过。"""
    base = dict(runs[-1])
    repeat = len(runs)
    passes = sum(1 for item in runs if item["passed"])
    base["passed"] = passes == repeat
    base["pass_count"] = passes
    base["repeat"] = repeat
    base["flaky"] = 0 < passes < repeat
    for key in ("citation_present", "expected_source_cited", "must_contain_ok", "retrieval_hit", "admits_no_answer"):
        values = [item.get(key) for item in runs if item.get(key) is not None]
        base[key] = all(values) if values else None
    base["fabricated_sources"] = sorted({source for item in runs for source in item["fabricated_sources"]})
    base["cited_sources"] = sorted({source for item in runs for source in item["cited_sources"]})
    base["task_mode"] = any(item.get("task_mode") for item in runs)
    base["structured_delivery"] = any(item.get("structured_delivery") for item in runs)
    base["answer"] = runs[-1]["answer"]
    return base


def render_console(summary: dict[str, Any]) -> str:
    lines = [
        "RAG 生成层评估",
        f"  题目数：{summary.get('unique_questions', summary['questions'])}"
        f" × 重复 {summary.get('repeat', 1)} 次 = {summary['questions']} 次运行"
        f"（可回答 {summary['answerable']} / 无答案 {summary['no_answer']} 次运行）",
        f"  通过率：{summary['pass_rate']:.1%}",
        f"  严格通过率：{summary.get('strict_pass_rate', summary['pass_rate']):.1%}"
        f"（每题重复 {summary.get('repeat', 1)} 次，全部通过才算过）",
        f"  波动题数：{summary.get('flaky_questions', 0)}"
        + (f"（{', '.join(summary.get('flaky_ids') or [])}）" if summary.get("flaky_ids") else ""),
        f"  引用出现率：{summary['citation_present_rate']:.1%}",
        f"  期望来源命中率：{summary['expected_source_cited_rate']:.1%}",
        f"  关键事实覆盖率：{summary['must_contain_rate']:.1%}",
        f"  检索命中率：{summary['retrieval_hit_rate']:.1%}",
        f"  编造引用题数：{summary['fabricated_source_questions']}",
        f"  无答案承认率：{summary['no_answer_admit_rate']:.1%}",
        f"  任务型结构化题数：{summary.get('task_mode_questions', 0)}",
        f"  平均工具调用：{summary['avg_tool_calls']}",
    ]
    return "\n".join(lines)


def render_markdown(
    results: list[dict[str, Any]],
    summary: dict[str, Any],
    *,
    generated_at: str | None = None,
    judge: bool = False,
) -> str:
    now = generated_at or datetime.now(timezone.utc).isoformat()
    lines = [
        "# RAG 生成层评估报告",
        "",
        f"- 生成时间：{now}",
        f"- 题目数：{summary.get('unique_questions', summary['questions'])}"
        f" × 重复 {summary.get('repeat', 1)} 次 = {summary['questions']} 次运行"
        f"（可回答 {summary['answerable']} / 无答案 {summary['no_answer']} 次运行）",
        f"- LLM-as-judge：{'开启' if judge else '关闭'}",
        "",
        "## 汇总",
        "",
        "| 指标 | 数值 |",
        "|------|------|",
        f"| 重复次数 | {summary.get('repeat', 1)} |",
        f"| 通过率 | {summary['pass_rate']:.1%} |",
        f"| 严格通过率 | {summary.get('strict_pass_rate', summary['pass_rate']):.1%} |",
        f"| 波动题数 | {summary.get('flaky_questions', 0)} |",
        f"| 引用出现率 | {summary['citation_present_rate']:.1%} |",
        f"| 期望来源命中率 | {summary['expected_source_cited_rate']:.1%} |",
        f"| 关键事实覆盖率 | {summary['must_contain_rate']:.1%} |",
        f"| 检索命中率 | {summary['retrieval_hit_rate']:.1%} |",
        f"| 编造引用题数 | {summary['fabricated_source_questions']} |",
        f"| 无答案承认率 | {summary['no_answer_admit_rate']:.1%} |",
        f"| 任务型结构化题数 | {summary.get('task_mode_questions', 0)} |",
        f"| 平均工具调用 | {summary['avg_tool_calls']} |",
        "",
        "## 逐题明细",
        "",
        "| 题号 | 类型 | 通过 | 通过次数 | 引用 | 期望来源 | 关键事实 | 检索命中 | 备注 |",
        "|------|------|------|----------|------|----------|----------|----------|------|",
    ]
    for item in results:
        notes: list[str] = []
        if item.get("flaky"):
            notes.append(f"波动（{item.get('pass_count', 0)}/{item.get('repeat', 1)}）")
        if item["fabricated_sources"]:
            notes.append("编造引用：" + ", ".join(item["fabricated_sources"]))
        if item["type"] == "no_answer" and not item["admits_no_answer"]:
            notes.append("未承认未覆盖")
        if item["type"] != "no_answer" and not item["expected_source_cited"]:
            notes.append("未引用期望来源")
        if item.get("structured_delivery") and not item.get("citation_present"):
            notes.append("结构化交付未带引用")
        if item.get("task_mode") is True:
            notes.append("任务型")
        lines.append(
            "| {id} | {type} | {passed} | {count}/{repeat} | {cited} | {expected} | {must} | {hit} | {notes} |".format(
                id=item["id"],
                type=item["type"],
                passed="✅" if item["passed"] else "❌",
                count=item.get("pass_count", 1),
                repeat=item.get("repeat", 1),
                cited="✅" if item["citation_present"] else "❌",
                expected="✅" if item["expected_source_cited"] else "❌",
                must=("—" if item["must_contain_ok"] is None else ("✅" if item["must_contain_ok"] else "❌")),
                hit=("—" if item["retrieval_hit"] is None else ("✅" if item["retrieval_hit"] else "❌")),
                notes="; ".join(notes) or "—",
            )
        )
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------- 跑 agent


def collect_events(agent: Any, question: str, thread_id: str) -> EventList:
    events: EventList = []
    for event in agent.stream_events(question, thread_id=thread_id):
        events.append(event)
        if event.get("type") in {"done", "error", "interrupt"}:
            break
    return events


def build_agent() -> Any:
    """构建待评估的 agent（HITL 关闭，避免评估在审批处中断）。"""
    os.environ["SKILLS_HITL"] = "false"
    from langchain_skills.agent import LangChainSkillsAgent

    return LangChainSkillsAgent()


def knowledge_sources(directory: Path = DEFAULT_KNOWLEDGE_DIR) -> set[str]:
    sources: set[str] = set()
    if not directory.exists():
        return sources
    for path in directory.rglob("*"):
        if path.suffix.lower() in {".md", ".markdown", ".txt", ".pdf"}:
            sources.add(path.relative_to(directory).as_posix())
    return sources


# --------------------------------------------------------------------- LLM-as-judge


class _JudgeResult:
    """结构化 judge 结果的轻量包装（避免 import pydantic 类型到模块级）。"""

    @staticmethod
    def schema() -> type:
        from pydantic import BaseModel, Field

        class JudgeResult(BaseModel):
            faithfulness: int = Field(ge=1, le=5, description="回答是否忠于检索结果，1-5 分")
            covers_expected: bool = Field(description="是否覆盖题目期望的关键事实")
            cites_sources: bool = Field(description="引用是否真实、不过期、不编造")
            reason: str = Field(description="一句话理由")

        return JudgeResult


def build_judge_model() -> Any:
    from langchain.chat_models import init_chat_model
    from langchain_skills.agent import resolve_model_config

    config = resolve_model_config(model=None, model_provider=None)
    kwargs: dict[str, Any] = {
        "temperature": 0.0,
        "max_tokens": 800,
        "model_provider": "openai" if config.provider == "deepseek" else config.provider,
    }
    if config.api_key:
        kwargs["api_key"] = config.api_key
    if config.base_url:
        kwargs["base_url"] = config.base_url
    if config.provider == "deepseek":
        kwargs["extra_body"] = {"thinking": {"type": "disabled"}}
    return init_chat_model(config.model, **kwargs)


def judge_question(model: Any, result: dict[str, Any]) -> dict[str, Any] | None:
    schema = _JudgeResult.schema()
    system = (
        "你是 RAG 评估裁判。给你题目、标准答案要素、检索到的资料和模型回答。"
        "只依据给的资料判断，不要用外部知识。返回结构化评分。"
    )
    user = json.dumps(
        {
            "question": result["question"],
            "type": result["type"],
            "expected_sources": result["expected_sources"],
            "retrieved": [
                {"source": item["source"], "section": item["section"], "content": item.get("content", "")}
                for item in result["retrieved"][:6]
            ],
            "answer": result["answer"],
        },
        ensure_ascii=False,
    )
    try:
        judged = model.with_structured_output(schema, method="function_calling").invoke(
            [{"role": "system", "content": system}, {"role": "user", "content": user}]
        )
    except Exception:  # noqa: BLE001 - judge 失败不影响主评估
        return None
    return judged.model_dump() if hasattr(judged, "model_dump") else dict(judged)


# --------------------------------------------------------------------- CLI


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="RAG 生成层（回答）评估")
    parser.add_argument("--questions", type=Path, default=DEFAULT_QUESTIONS, help="问题集 JSONL 路径")
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT, help="Markdown 报告输出路径")
    parser.add_argument("--limit", type=int, default=DEFAULT_LIMIT, help="只跑前 N 题（0=全部）")
    parser.add_argument("--repeat", type=int, default=1, help="每题重复跑几次（默认 1；>1 用于测非确定性）")
    parser.add_argument("--thread-prefix", default="eval-answer", help="每题独立 thread 的前缀")
    parser.add_argument("--judge", action="store_true", help="额外调用 LLM-as-judge（会产生费用）")
    parser.add_argument("--json", action="store_true", help="同时输出 JSON 到 stdout")
    return parser


def main(argv: list[str] | None = None, run_one: RunOne | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        questions = load_questions(args.questions)
    except RuntimeError as exc:
        print(f"[FAILED] {exc}")
        return 1
    if args.limit > 0:
        questions = questions[: args.limit]

    known_sources = knowledge_sources()

    if run_one is None:
        try:
            agent = build_agent()
        except Exception as exc:  # noqa: BLE001 - 启动失败给出可读提示
            print(f"[FAILED] 无法初始化 agent：{type(exc).__name__}: {exc}")
            return 1

        run_counter = {"n": 0}

        def run_one(question: dict[str, Any]) -> EventList:  # noqa: F811 - 覆盖占位
            run_counter["n"] += 1
            thread_id = f"{args.thread_prefix}-{question['id']}-{run_counter['n']}"
            return collect_events(agent, str(question["question"]), thread_id)

    judge_model = build_judge_model() if args.judge else None
    repeat = max(1, args.repeat)

    results: list[dict[str, Any]] = []
    aggregated: list[dict[str, Any]] = []
    for index, question in enumerate(questions, start=1):
        runs: list[dict[str, Any]] = []
        for run_index in range(1, repeat + 1):
            started = time.perf_counter()
            try:
                events = run_one(question)
            except Exception as exc:  # noqa: BLE001 - 单题失败继续跑完
                print(f"[WARN] {question['id']} 运行失败：{type(exc).__name__}: {exc}")
                events = [{"type": "error", "message": str(exc)}]
            elapsed = round(time.perf_counter() - started, 2)
            result = evaluate_question(question, events, known_sources=known_sources)
            result["elapsed_s"] = elapsed
            if judge_model is not None:
                result["judge"] = judge_question(judge_model, result)
            results.append(result)
            runs.append(result)
            mark = "PASS" if result["passed"] else "MISS"
            print(f"[{index}/{len(questions)}] {result['id']} r{run_index}/{repeat} {mark} ({elapsed}s)")
        aggregated.append(aggregate_runs(runs))

    summary = summarize(results)
    strict_passes = sum(1 for item in aggregated if item["passed"])
    summary.update(
        {
            "repeat": repeat,
            "unique_questions": len(aggregated),
            "strict_pass_rate": round(strict_passes / len(aggregated), 4) if aggregated else 0.0,
            "flaky_questions": sum(1 for item in aggregated if item.get("flaky")),
            "flaky_ids": [item["id"] for item in aggregated if item.get("flaky")],
        }
    )
    if args.judge:
        graded = [item["judge"] for item in results if item.get("judge")]
        if graded:
            summary["judge_faithfulness_avg"] = round(
                sum(item["faithfulness"] for item in graded) / len(graded), 2
            )
            summary["judge_covers_expected_rate"] = round(
                sum(1 for item in graded if item["covers_expected"]) / len(graded), 4
            )

    report = render_markdown(aggregated, summary, judge=args.judge)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(report, encoding="utf-8")

    print(render_console(summary))
    print(f"[OK] 报告：{args.report}")
    if args.json:
        print(json.dumps({"summary": summary, "results": results}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
