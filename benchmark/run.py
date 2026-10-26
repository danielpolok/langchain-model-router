"""The one command to reproduce the benchmark:

    uv run python -m benchmark.run

Needs no API key: every route and decision model is local (Ollama, plus any configured System One
endpoint), and answers are recorded ungraded. Grade them offline: `--export-ungraded RAW` writes
each answer with its grading prompt, a grader writes back `[{arm, item_id, score, rationale}]`, and
`--import-grades RAW GRADES` merges those into the results and renders the report.
`--split domain` routes the same items to one route per domain. Paid Gemini models run only on
the opt-ins described in `benchmark/README.md`.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from benchmark.arms import build_arms
from benchmark.costing import CostBreakdown
from benchmark.dataset import Split, load_dataset
from benchmark.judge import JudgeVerdict, grading_prompt, judge_model
from benchmark.report import render_markdown, summarize
from benchmark.runner import ItemResult, run_arm
from langchain_model_router import RoutingDecision

RESULTS_DIR = Path(__file__).parent / "results"


def _to_raw(result: ItemResult) -> dict[str, object]:
    """`ItemResult` -> plain JSON-able dict, for the versioned raw-results file."""
    return {
        "item_id": result.item_id,
        "arm": result.arm,
        "decision": result.decision.as_dict() if result.decision else None,
        "answer": result.answer,
        "tool_called": result.tool_called,
        "tool_correct": result.tool_correct,
        "usage_by_model": result.usage_by_model,
        "cost": asdict(result.cost),
        "judge": result.judge.model_dump() if result.judge else None,
        "error": result.error,
        "expected_route": result.expected_route,
        "decision_seconds": result.decision_seconds,
    }


def _from_raw(raw: dict[str, Any]) -> ItemResult:
    """The inverse of `_to_raw`, for `--report-only`'s re-render from a saved run."""
    return ItemResult(
        item_id=raw["item_id"],
        arm=raw["arm"],
        decision=RoutingDecision.from_dict(raw["decision"]) if raw["decision"] else None,
        answer=raw["answer"],
        tool_called=raw["tool_called"],
        tool_correct=raw["tool_correct"],
        usage_by_model=raw["usage_by_model"],
        cost=CostBreakdown(**raw["cost"]),
        judge=JudgeVerdict(**raw["judge"]) if raw["judge"] else None,
        error=raw["error"],
        expected_route=raw.get("expected_route"),
        decision_seconds=raw.get("decision_seconds"),
    )


def export_ungraded(raw_path: Path) -> Path:
    """Write every answered, ungraded result in `raw_path` to `<stem>-ungraded.json` beside it:
    arm, item id, prompt, rubric, answer, and the exact grading prompt an in-run judge would see.
    """
    items = {item.id: item for item in load_dataset()}
    raw = json.loads(raw_path.read_text())
    entries = []
    for arm, results in raw.items():
        for result in results:
            if result["error"] is not None or result["judge"] is not None:
                continue
            item = items[result["item_id"]]
            entries.append(
                {
                    "arm": arm,
                    "item_id": item.id,
                    "prompt": item.prompt,
                    "rubric": item.rubric,
                    "answer": result["answer"],
                    "grading_prompt": grading_prompt(item.prompt, item.rubric, result["answer"]),
                }
            )
    out = raw_path.with_name(f"{raw_path.stem}-ungraded.json")
    out.write_text(json.dumps(entries, indent=2))
    return out


def import_grades(raw_path: Path, grades_path: Path) -> int:
    """Merge `[{arm, item_id, score, rationale}]` from `grades_path` into `raw_path`'s results as
    their `judge` verdicts. Every grade is checked (a known result, a valid `JudgeVerdict`) before
    the file is rewritten. Returns how many answered results are still ungraded."""
    raw = json.loads(raw_path.read_text())
    index = {(arm, r["item_id"]): r for arm, results in raw.items() for r in results}
    for grade in json.loads(grades_path.read_text()):
        result = index.get((grade["arm"], grade["item_id"]))
        if result is None or result["error"] is not None:
            msg = f"no answered result for {grade['arm']} / {grade['item_id']} in {raw_path}"
            raise SystemExit(msg)
        verdict = JudgeVerdict(score=grade["score"], rationale=grade["rationale"])
        result["judge"] = verdict.model_dump()
    raw_path.write_text(json.dumps(raw, indent=2))
    return sum(1 for r in index.values() if r["error"] is None and r["judge"] is None)


def _run(
    arm_names: list[str] | None, limit: int | None, pause: float, split: Split
) -> tuple[dict[str, list[ItemResult]], Path]:
    items = list(load_dataset())
    if limit:
        items = items[:limit]
    arms = build_arms(split)
    if arm_names:
        arms = [arm for arm in arms if arm.name in arm_names]
        if not arms:
            msg = f"no arm matches {arm_names}"
            raise SystemExit(msg)

    judge = judge_model()
    by_arm: dict[str, list[ItemResult]] = {}
    for arm in arms:
        print(f"running {arm.name} ({len(items)} items)...", file=sys.stderr)
        results = run_arm(
            arm,
            items,
            judge,
            embed=(arm.name == "embedding"),
            pause_seconds=pause,
            split=split,
        )
        by_arm[arm.name] = results
        errors = sum(1 for r in results if r.error)
        print(f"  {len(results) - errors}/{len(results)} ok", file=sys.stderr)

    RESULTS_DIR.mkdir(exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    raw_path = RESULTS_DIR / f"{stamp}-{split}.json"
    raw = {arm: [_to_raw(r) for r in results] for arm, results in by_arm.items()}
    raw_path.write_text(json.dumps(raw, indent=2))
    print(f"raw results: {raw_path}", file=sys.stderr)
    return by_arm, raw_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arms", nargs="*", default=None, help="Arm names (default: all).")
    parser.add_argument(
        "--split",
        choices=["tier", "domain"],
        default="tier",
        help="Route by difficulty (small/frontier) or one route per domain.",
    )
    parser.add_argument("--limit", type=int, default=None, help="Cap the dataset to N items.")
    parser.add_argument(
        "--pause", type=float, default=0.0, help="Seconds between items (rate-limit courtesy)."
    )
    parser.add_argument(
        "--report-only",
        type=Path,
        default=None,
        help="Skip running: re-render the report beside a saved raw-results JSON.",
    )
    parser.add_argument(
        "--export-ungraded",
        type=Path,
        default=None,
        metavar="RAW",
        help="Skip running: write a saved run's ungraded answers for offline grading.",
    )
    parser.add_argument(
        "--import-grades",
        type=Path,
        nargs=2,
        default=None,
        metavar=("RAW", "GRADES"),
        help="Skip running: merge offline grades into a saved run, then re-render its report.",
    )
    args = parser.parse_args(argv)

    load_dotenv()

    if args.export_ungraded:
        print(f"ungraded answers: {export_ungraded(args.export_ungraded)}", file=sys.stderr)
        return 0
    if args.import_grades:
        remaining = import_grades(*args.import_grades)
        print(f"grades merged; {remaining} answers still ungraded", file=sys.stderr)

    if args.report_only or args.import_grades:
        raw_path = args.report_only or args.import_grades[0]
        raw = json.loads(raw_path.read_text())
        by_arm = {arm: [_from_raw(r) for r in results] for arm, results in raw.items()}
    else:
        by_arm, raw_path = _run(args.arms, args.limit, args.pause, args.split)

    summaries = summarize(by_arm)
    report = render_markdown(summaries)
    report_path = raw_path.with_suffix(".md")
    report_path.write_text(report)
    print(report)
    print(f"report written to {report_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
