"""RAG 检索评估：用标注问题集给「检索层」打分。

只评检索，不评生成：给定一道题，看期望的 chunk 有没有出现在 top-k、排第几。
指标定义：

- **Hit@k**：期望 section 出现在前 k 条里（多跳题要求所有期望 section 都出现）。
- **MRR**：只统计单跳题，取第一个命中排名的倒数，再平均。
- **Precision@k**：单跳题 top-k 里相关 chunk 的比例。
- **无答案题**：知识库没有覆盖的问题，看 top-1 相似度是否足够低（默认 < 0.45）。

用法::

    # 前置：Chroma 已启动（docker compose up -d chroma）、索引已构建、.env 有 ZHIPU_API_KEY
    uv run python scripts/eval_rag.py
    uv run python scripts/eval_rag.py --top-k 6 --json
    uv run python scripts/eval_rag.py --questions evals/rag_questions.jsonl --report evals/report.md
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(PROJECT_ROOT / ".env")

DEFAULT_QUESTIONS = PROJECT_ROOT / "evals" / "rag_questions.jsonl"
DEFAULT_REPORT = PROJECT_ROOT / "evals" / "report.md"
DEFAULT_TOP_K = 6
HIT_AT = (1, 4)
NO_ANSWER_SCORE_THRESHOLD = 0.45
VALID_TYPES = {"single_hop", "multi_hop", "no_answer"}

Retrieve = Callable[[str], list[dict[str, Any]]]


# --------------------------------------------------------------------- 数据加载


def load_questions(path: Path = DEFAULT_QUESTIONS) -> list[dict[str, Any]]:
    """读取 JSONL 问题集，并做完整性校验（标签错了评估结果就没意义）。"""
    if not path.exists():
        raise RuntimeError(f"问题集不存在：{path}")

    questions: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for line_number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw.strip()
        if not line:
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"{path.name}:{line_number} 不是合法 JSON：{exc}") from exc

        qid = str(item.get("id") or "").strip()
        if not qid:
            raise RuntimeError(f"{path.name}:{line_number} 缺少 id")
        if qid in seen_ids:
            raise RuntimeError(f"{path.name}:{line_number} id 重复：{qid}")
        seen_ids.add(qid)

        if not str(item.get("question") or "").strip():
            raise RuntimeError(f"{path.name}:{line_number}（{qid}）question 为空")

        qtype = item.get("type")
        if qtype not in VALID_TYPES:
            raise RuntimeError(f"{path.name}:{line_number}（{qid}）type 非法：{qtype}")

        if qtype != "no_answer":
            if not item.get("expected_sources"):
                raise RuntimeError(f"{path.name}:{line_number}（{qid}）缺少 expected_sources")
            if not item.get("expected_sections"):
                raise RuntimeError(f"{path.name}:{line_number}（{qid}）缺少 expected_sections")

        item.setdefault("expected_sources", [])
        item.setdefault("expected_sections", [])
        item.setdefault("require_all_sections", qtype == "multi_hop")
        item.setdefault("must_contain", [])
        questions.append(item)

    if not questions:
        raise RuntimeError(f"问题集为空：{path}")
    return questions


# --------------------------------------------------------------------- 匹配逻辑


def section_matches(actual: str, expected: str) -> bool:
    """section 匹配：完全相等，或一方是另一方的前缀（章节被再细分时仍算命中）。"""
    actual_clean = (actual or "").strip()
    expected_clean = (expected or "").strip()
    if not actual_clean or not expected_clean:
        return False
    if actual_clean == expected_clean:
        return True
    return actual_clean.startswith(expected_clean) or expected_clean.startswith(actual_clean)


def is_relevant(item: dict[str, Any], question: dict[str, Any]) -> bool:
    """一条检索结果是否算「相关」：source 必须在期望列表里，且 section 命中任意期望节。"""
    if item.get("source") not in question.get("expected_sources", []):
        return False
    return any(
        section_matches(item.get("section", ""), expected)
        for expected in question.get("expected_sections", [])
    )


# --------------------------------------------------------------------- 评估主逻辑


def evaluate_questions(
    questions: list[dict[str, Any]],
    retrieve: Retrieve,
    *,
    top_k: int = DEFAULT_TOP_K,
) -> dict[str, Any]:
    """对每道题跑检索并汇总指标；retrieve 返回 [{source, section, score, ...}]。"""
    rows: list[dict[str, Any]] = []

    for question in questions:
        items = list(retrieve(question["question"]) or [])[:top_k]
        top1 = items[0] if items else None
        row: dict[str, Any] = {
            "id": question["id"],
            "question": question["question"],
            "type": question["type"],
            "top1": top1,
            "results": items,
        }

        if question["type"] == "no_answer":
            best_score = float(top1.get("score", 0.0)) if top1 else 0.0
            row["best_score"] = round(best_score, 4)
            row["passed"] = best_score < NO_ANSWER_SCORE_THRESHOLD
            rows.append(row)
            continue

        # 每节第一次出现的排名
        section_rank: dict[str, int] = {}
        relevant_ranks: list[int] = []
        for rank, item in enumerate(items, start=1):
            if not is_relevant(item, question):
                continue
            relevant_ranks.append(rank)
            for expected in question["expected_sections"]:
                if expected not in section_rank and section_matches(item.get("section", ""), expected):
                    section_rank[expected] = rank

        expected_sections = question["expected_sections"]
        require_all = bool(question.get("require_all_sections"))
        covered_ranks = [section_rank.get(expected) for expected in expected_sections]
        all_covered = all(rank is not None for rank in covered_ranks)

        if require_all:
            hit_rank = max(covered_ranks) if all_covered else None
        else:
            hit_rank = relevant_ranks[0] if relevant_ranks else None

        row.update(
            {
                "hit_rank": hit_rank,
                "relevant_ranks": relevant_ranks,
                "relevant_count": len(relevant_ranks),
                "covered_sections": [section for section in expected_sections if section in section_rank],
                "missing_sections": [section for section in expected_sections if section not in section_rank],
                "all_covered": all_covered if require_all else None,
            }
        )
        rows.append(row)

    answerable = [row for row in rows if row["type"] != "no_answer"]
    single_hop = [row for row in rows if row["type"] == "single_hop"]
    multi_hop = [row for row in rows if row["type"] == "multi_hop"]
    no_answer = [row for row in rows if row["type"] == "no_answer"]

    metrics: dict[str, Any] = {
        "question_count": len(rows),
        "answerable_count": len(answerable),
        "single_hop_count": len(single_hop),
        "multi_hop_count": len(multi_hop),
        "no_answer_count": len(no_answer),
        "top_k": top_k,
    }

    for k in HIT_AT:
        hits = sum(1 for row in answerable if row.get("hit_rank") is not None and row["hit_rank"] <= k)
        metrics[f"hit@{k}"] = round(hits / len(answerable), 4) if answerable else 0.0

    if single_hop:
        mrr = sum(1.0 / row["hit_rank"] for row in single_hop if row.get("hit_rank")) / len(single_hop)
        metrics["mrr"] = round(mrr, 4)
        precision = sum(row["relevant_count"] for row in single_hop) / len(single_hop) / top_k
        metrics["precision_at_k"] = round(precision, 4)
    else:
        metrics["mrr"] = 0.0
        metrics["precision_at_k"] = 0.0

    if multi_hop:
        metrics["multi_hop_full_coverage"] = round(
            sum(1 for row in multi_hop if row.get("all_covered")) / len(multi_hop), 4
        )
    else:
        metrics["multi_hop_full_coverage"] = 0.0

    if no_answer:
        metrics["no_answer_pass_rate"] = round(
            sum(1 for row in no_answer if row["passed"]) / len(no_answer), 4
        )
        metrics["no_answer_avg_best_score"] = round(
            sum(row["best_score"] for row in no_answer) / len(no_answer), 4
        )
        metrics["no_answer_threshold"] = NO_ANSWER_SCORE_THRESHOLD
    else:
        metrics["no_answer_pass_rate"] = 0.0
        metrics["no_answer_avg_best_score"] = 0.0
        metrics["no_answer_threshold"] = NO_ANSWER_SCORE_THRESHOLD

    return {"metrics": metrics, "rows": rows}


# --------------------------------------------------------------------- 报告输出


def _format_location(item: dict[str, Any] | None) -> str:
    if not item:
        return "-"
    location = str(item.get("source") or "-")
    if item.get("section"):
        location += f" > {item['section']}"
    return location


def render_console(result: dict[str, Any]) -> str:
    metrics = result["metrics"]
    lines = [
        f"[OK] {metrics['question_count']} 题（单跳 {metrics['single_hop_count']} / "
        f"多跳 {metrics['multi_hop_count']} / 无答案 {metrics['no_answer_count']}，top_k={metrics['top_k']}）",
        f"     Hit@1={metrics['hit@1']:.1%}  Hit@4={metrics['hit@4']:.1%}  "
        f"MRR={metrics['mrr']:.3f}  Precision@k={metrics['precision_at_k']:.3f}",
        f"     多跳全覆盖率={metrics['multi_hop_full_coverage']:.1%}  "
        f"无答案拒答率={metrics['no_answer_pass_rate']:.1%}"
        f"（平均 top-1 score={metrics['no_answer_avg_best_score']:.3f}，阈值 {metrics['no_answer_threshold']}）",
    ]

    failures = [row for row in result["rows"] if row["type"] != "no_answer" and row.get("hit_rank") is None]
    failures += [row for row in result["rows"] if row["type"] == "no_answer" and not row.get("passed")]
    if failures:
        lines.append("     失败题：")
        for row in failures:
            expected = " | ".join(row.get("covered_sections") or [])
            expected = expected or "（无答案题）"
            lines.append(
                f"       - {row['id']} [{row['type']}] top1={_format_location(row['top1'])}"
                f" score={float((row['top1'] or {}).get('score', 0.0)):.3f} 期望={expected}"
            )
    return "\n".join(lines)


def render_markdown(result: dict[str, Any], *, questions_path: Path, collection: str) -> str:
    metrics = result["metrics"]
    generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%SZ")
    lines = [
        "# RAG 检索评估报告",
        "",
        f"- 生成时间：{generated_at}",
        f"- 问题集：`{questions_path}`（{metrics['question_count']} 题）",
        f"- collection：`{collection}`，top_k={metrics['top_k']}",
        "- 指标口径：Hit@k 要求期望 section 出现在前 k 条（多跳题要求全部 section）；"
        "MRR/Precision@k 只统计单跳题；无答案题看 top-1 相似度是否低于阈值。",
        "",
        "## 汇总",
        "",
        "| 指标 | 值 |",
        "| --- | --- |",
        f"| Hit@1 | {metrics['hit@1']:.1%} |",
        f"| Hit@4 | {metrics['hit@4']:.1%} |",
        f"| MRR（单跳） | {metrics['mrr']:.3f} |",
        f"| Precision@k（单跳） | {metrics['precision_at_k']:.3f} |",
        f"| 多跳全覆盖率 | {metrics['multi_hop_full_coverage']:.1%} |",
        f"| 无答案拒答率 | {metrics['no_answer_pass_rate']:.1%} |",
        f"| 无答案平均 top-1 score | {metrics['no_answer_avg_best_score']:.3f}（阈值 {metrics['no_answer_threshold']}） |",
        "",
        "## 逐题明细",
        "",
        "| id | 类型 | 命中排名 | top-1 位置 | top-1 score | 期望 section |",
        "| --- | --- | --- | --- | --- | --- |",
    ]

    for row in result["rows"]:
        top1 = row["top1"] or {}
        score = f"{float(top1.get('score', 0.0)):.3f}" if top1 else "-"
        if row["type"] == "no_answer":
            rank = "拒答通过" if row.get("passed") else "拒答失败"
            expected = "（知识库未覆盖）"
        else:
            hit_rank = row.get("hit_rank")
            rank = f"{hit_rank}" if hit_rank else "未命中"
            expected = " | ".join(row.get("covered_sections") or []) or " | ".join(
                row.get("missing_sections") or []
            ) or "-"
        lines.append(
            f"| {row['id']} | {row['type']} | {rank} | {_format_location(top1) if top1 else '-'} "
            f"| {score} | {expected} |"
        )

    lines += ["", "## 失败分析", ""]
    failures = [row for row in result["rows"] if row["type"] != "no_answer" and row.get("hit_rank") is None]
    failures += [row for row in result["rows"] if row["type"] == "no_answer" and not row.get("passed")]
    if not failures:
        lines.append("无（全部通过当前指标）。")
    else:
        for row in failures:
            reason = (
                "无答案题但 top-1 相似度过高，模型可能误以为知识库有答案"
                if row["type"] == "no_answer"
                else "期望 section 未出现在 top-k"
            )
            lines.append(f"- **{row['id']}**（{row['question']}）：{reason}")
    lines.append("")
    return "\n".join(lines)


# --------------------------------------------------------------------- CLI


def build_retrieve(*, collection: str | None, host: str | None, port: int | None, top_k: int) -> tuple[Retrieve, Any]:
    """构建真实检索函数（Chroma + 智谱 embedding），返回 (retrieve, embeddings)。"""
    import search_docs as search_module
    from rag_embeddings import ZhipuEmbeddings

    env_host, env_port, use_ssl = search_module.chroma_settings()
    client = search_module.create_chroma_client(host or env_host, port or env_port, use_ssl)
    embeddings = ZhipuEmbeddings.from_env()
    store = search_module.open_vector_store(
        client,
        collection or search_module.DEFAULT_COLLECTION,
        embeddings,
    )

    def retrieve(query: str) -> list[dict[str, Any]]:
        pairs = store.similarity_search_with_score(query, k=top_k)
        results: list[dict[str, Any]] = []
        for document, distance in pairs:
            metadata = document.metadata or {}
            results.append(
                {
                    "source": str(metadata.get("source") or ""),
                    "section": str(metadata.get("section") or ""),
                    "score": round(1.0 - float(distance), 4),
                    "text": document.page_content,
                }
            )
        return results

    return retrieve, embeddings


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="RAG 检索层评估（Hit@k / MRR / Precision）")
    parser.add_argument("--questions", default=str(DEFAULT_QUESTIONS), help="问题集 JSONL 路径")
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K, help="检索条数，默认 6")
    parser.add_argument("--report", default=str(DEFAULT_REPORT), help="Markdown 报告输出路径")
    parser.add_argument("--collection", default=None, help="collection 名，默认 RAG_COLLECTION")
    parser.add_argument("--chroma-host", default=None, help="Chroma 主机，默认 RAG_CHROMA_HOST")
    parser.add_argument("--chroma-port", type=int, default=None, help="Chroma 端口，默认 RAG_CHROMA_PORT")
    parser.add_argument("--json", action="store_true", help="额外输出 JSON（不含 results 全文）")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    questions_path = Path(args.questions)
    if args.top_k <= 0:
        print("[FAILED] --top-k 必须大于 0。")
        return 1

    try:
        questions = load_questions(questions_path)
    except RuntimeError as exc:
        print(f"[FAILED] {exc}")
        return 1

    collection = args.collection or __import__("os").getenv("RAG_COLLECTION") or "knowledge"
    embeddings = None
    try:
        retrieve, embeddings = build_retrieve(
            collection=args.collection,
            host=args.chroma_host,
            port=args.chroma_port,
            top_k=args.top_k,
        )
        result = evaluate_questions(questions, retrieve, top_k=args.top_k)
    except (RuntimeError, ValueError) as exc:
        print(f"[FAILED] {exc}")
        return 1
    finally:
        if embeddings is not None:
            embeddings.close()

    print(render_console(result))

    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        render_markdown(result, questions_path=questions_path, collection=collection),
        encoding="utf-8",
    )
    print(f"     报告：{report_path}")

    if args.json:
        compact = {
            "metrics": result["metrics"],
            "questions": [
                {key: row.get(key) for key in ("id", "type", "hit_rank", "passed")}
                for row in result["rows"]
            ],
        }
        print(json.dumps(compact, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
