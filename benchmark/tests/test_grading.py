"""The no-judge run path and offline grading: export ungraded answers, import grades, report.

Entirely offline: fake routes, no judge model, no network.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from benchmark import arms, judge
from benchmark.dataset import load_dataset
from benchmark.report import BASELINE_ARM, summarize
from benchmark.run import _from_raw, _to_raw, export_ungraded, import_grades, main
from benchmark.runner import run_arm
from langchain_model_router import ChatRouter
from langchain_model_router.strategies import HeuristicStrategy
from tests.fakes import NativeStructuredFakeChatModel

ITEMS = [item for item in load_dataset() if item.kind == "single_turn"][:2]


def _model(reply: str) -> NativeStructuredFakeChatModel:
    return NativeStructuredFakeChatModel(model_name="qwen3:8b", reply=reply)


def _saved_run(tmp_path: Path) -> Path:
    """A raw-results file from a no-judge run of two arms over two real dataset items."""
    baseline = arms.Arm(
        BASELINE_ARM, ChatRouter(routes={"frontier": _model("big")}, default_route="frontier")
    )
    heuristic = arms.Arm(
        "heuristic",
        ChatRouter(
            routes={"small": _model("small"), "frontier": _model("big")},
            default_route="small",
            strategy=HeuristicStrategy("small", "frontier"),
        ),
    )
    raw = {
        arm.name: [_to_raw(r) for r in run_arm(arm, ITEMS, None, embed=False)]
        for arm in [baseline, heuristic]
    }
    path = tmp_path / "run-tier.json"
    path.write_text(json.dumps(raw))
    return path


def test_with_no_judge_model_configured_there_is_no_judge(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(judge, "DEFAULT_JUDGE_MODEL", None)
    assert judge.judge_model() is None


def test_a_run_without_a_judge_records_every_answer_ungraded(tmp_path: Path) -> None:
    raw = json.loads(_saved_run(tmp_path).read_text())
    results = [r for results in raw.values() for r in results]
    assert len(results) == 4
    assert all(r["error"] is None and r["answer"] and r["judge"] is None for r in results)


def test_export_writes_each_ungraded_answer_with_its_grading_prompt(tmp_path: Path) -> None:
    exported = json.loads(export_ungraded(_saved_run(tmp_path)).read_text())

    assert len(exported) == 4
    entry = exported[0]
    item = ITEMS[0]
    assert (entry["arm"], entry["item_id"]) == (BASELINE_ARM, item.id)
    assert (entry["prompt"], entry["rubric"], entry["answer"]) == (item.prompt, item.rubric, "big")
    assert entry["grading_prompt"] == judge.grading_prompt(item.prompt, item.rubric, "big")


def test_imported_grades_give_the_report_its_mean_quality(tmp_path: Path) -> None:
    raw_path = _saved_run(tmp_path)
    grades = [
        {"arm": e["arm"], "item_id": e["item_id"], "score": 8, "rationale": "meets the rubric"}
        for e in json.loads(export_ungraded(raw_path).read_text())
    ]
    grades_path = tmp_path / "grades.json"
    grades_path.write_text(json.dumps(grades[:3]))

    assert import_grades(raw_path, grades_path) == 1  # one answer left ungraded
    assert export_ungraded(raw_path).read_text().count('"item_id"') == 1

    grades_path.write_text(json.dumps(grades[3:]))
    assert import_grades(raw_path, grades_path) == 0
    raw = json.loads(raw_path.read_text())
    summaries = summarize({arm: [_from_raw(r) for r in rs] for arm, rs in raw.items()})
    assert summaries[BASELINE_ARM].mean_quality == 8.0
    assert summaries["heuristic"].quality_pct_of_baseline == 100.0


def test_import_rejects_a_bad_grade_without_touching_the_results(tmp_path: Path) -> None:
    raw_path = _saved_run(tmp_path)
    before = raw_path.read_text()
    grades_path = tmp_path / "grades.json"

    grades_path.write_text(
        json.dumps([{"arm": "nope", "item_id": "x", "score": 5, "rationale": ""}])
    )
    with pytest.raises(SystemExit, match="no answered result"):
        import_grades(raw_path, grades_path)

    bad_score = {"arm": BASELINE_ARM, "item_id": ITEMS[0].id, "score": 11, "rationale": "too high"}
    grades_path.write_text(json.dumps([bad_score]))
    with pytest.raises(ValidationError):
        import_grades(raw_path, grades_path)

    assert raw_path.read_text() == before


def test_the_import_command_renders_the_report_beside_the_run(tmp_path: Path) -> None:
    raw_path = _saved_run(tmp_path)
    grades_path = tmp_path / "grades.json"
    grades_path.write_text(
        json.dumps(
            [
                {"arm": e["arm"], "item_id": e["item_id"], "score": 6, "rationale": "partly"}
                for e in json.loads(export_ungraded(raw_path).read_text())
            ]
        )
    )

    assert main(["--import-grades", str(raw_path), str(grades_path)]) == 0
    assert "| 6.00 |" in raw_path.with_suffix(".md").read_text()


def test_the_default_arms_are_all_local(monkeypatch: pytest.MonkeyPatch) -> None:
    """No paid model without the Gemini opt-in: every route is Ollama, and no Gemini-only arm."""
    from langchain_ollama import ChatOllama

    monkeypatch.delenv("MODEL_ROUTER_BENCHMARK_KEV_URL", raising=False)
    monkeypatch.setattr(arms, "GEMINI", False)

    for split in ("tier", "domain"):
        built = arms.build_arms(split)
        assert "embedding" not in {arm.name for arm in built}
        assert "classifier" not in {arm.name for arm in built}
        for arm in built:
            assert all(isinstance(route, ChatOllama) for route in arm.router.routes.values())
