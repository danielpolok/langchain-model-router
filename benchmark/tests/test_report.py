"""Report aggregation: cost-saved and quality-vs-baseline percentages, and the target verdict."""

from __future__ import annotations

import pytest

from benchmark.costing import CostBreakdown
from benchmark.judge import JudgeVerdict
from benchmark.report import BASELINE_ARM, render_markdown, summarize
from benchmark.runner import ItemResult
from langchain_model_router import RoutingDecision


def _result(
    item_id: str,
    arm: str,
    *,
    route: str,
    cost: float,
    score: int,
    error: str | None = None,
) -> ItemResult:
    return ItemResult(
        item_id=item_id,
        arm=arm,
        decision=RoutingDecision(route=route, reason="test") if error is None else None,
        answer="an answer",
        tool_called=None,
        tool_correct=None,
        usage_by_model={},
        cost=CostBreakdown(chat_usd=cost, embedding_usd=0.0),
        judge=JudgeVerdict(score=score, rationale="test") if error is None else None,
        error=error,
    )


def test_a_strategy_costing_half_as_much_at_full_quality_meets_the_target() -> None:
    by_arm = {
        BASELINE_ARM: [_result("1", BASELINE_ARM, route="frontier", cost=1.0, score=10)],
        "heuristic": [_result("1", "heuristic", route="small", cost=0.5, score=10)],
    }
    summaries = summarize(by_arm)
    heuristic = summaries["heuristic"]
    assert heuristic.cost_saved_pct == 50.0
    assert heuristic.quality_pct_of_baseline == 100.0
    assert heuristic.meets_target is True


def test_a_strategy_saving_little_cost_misses_the_target() -> None:
    by_arm = {
        BASELINE_ARM: [_result("1", BASELINE_ARM, route="frontier", cost=1.0, score=10)],
        "keyword": [_result("1", "keyword", route="frontier", cost=0.95, score=10)],
    }
    summaries = summarize(by_arm)
    assert summaries["keyword"].cost_saved_pct == pytest.approx(5.0)
    assert summaries["keyword"].meets_target is False


def test_a_strategy_that_tanks_quality_misses_the_target_even_if_cheap() -> None:
    by_arm = {
        BASELINE_ARM: [_result("1", BASELINE_ARM, route="frontier", cost=1.0, score=10)],
        "classifier": [_result("1", "classifier", route="small", cost=0.1, score=5)],
    }
    summaries = summarize(by_arm)
    classifier = summaries["classifier"]
    assert classifier.cost_saved_pct == 90.0
    assert classifier.quality_pct_of_baseline == 50.0
    assert classifier.meets_target is False


def test_the_baseline_itself_has_no_relative_figures() -> None:
    by_arm = {BASELINE_ARM: [_result("1", BASELINE_ARM, route="frontier", cost=1.0, score=10)]}
    summaries = summarize(by_arm)
    assert summaries[BASELINE_ARM].cost_saved_pct is None
    assert summaries[BASELINE_ARM].meets_target is None


def test_errors_are_excluded_from_quality_but_counted() -> None:
    by_arm = {
        "heuristic": [
            _result("1", "heuristic", route="small", cost=0.1, score=9),
            _result("2", "heuristic", route="small", cost=0.0, score=0, error="boom"),
        ]
    }
    summary = summarize(by_arm)["heuristic"]
    assert summary.n_errors == 1
    assert summary.mean_quality == 9.0  # the errored item is excluded, not scored as 0


def test_render_markdown_includes_every_arm_and_a_verdict() -> None:
    by_arm = {
        BASELINE_ARM: [_result("1", BASELINE_ARM, route="frontier", cost=1.0, score=10)],
        "heuristic": [_result("1", "heuristic", route="small", cost=0.5, score=10)],
    }
    report = render_markdown(summarize(by_arm))
    assert BASELINE_ARM in report
    assert "heuristic" in report
    assert "## Verdict" in report
    assert "Target met by: heuristic" in report


def test_routing_accuracy_and_decision_latency_are_computed_per_arm() -> None:
    results = [
        _result("1", "classifier", route="small", cost=0.1, score=8),
        _result("2", "classifier", route="small", cost=0.1, score=8),
        _result("3", "classifier", route="frontier", cost=0.1, score=8),
        _result("4", "classifier", route="small", cost=0.0, score=0, error="boom"),
    ]
    for result, expected, seconds in zip(
        results, ["small", "frontier", "frontier", "small"], [0.1, 0.3, 0.2, None], strict=True
    ):
        result.expected_route = expected
        result.decision_seconds = seconds
    results[1].decision = RoutingDecision(route="small", reason="test", fallback=True)

    summary = summarize({"classifier": results})["classifier"]

    assert (summary.routed_correctly, summary.routed) == (2, 3)  # the errored item isn't counted
    assert summary.fallbacks == 1
    assert summary.decision_p50_s == pytest.approx(0.2)
    assert summary.decision_p95_s == pytest.approx(0.3)


def test_the_report_shows_accuracy_latency_and_every_arm_without_a_baseline() -> None:
    result = _result("1", "classifier-qwen3", route="coder", cost=0.0, score=8)
    result.expected_route = "coder"
    result.decision_seconds = 0.5

    report = render_markdown(summarize({"classifier-qwen3": [result]}))

    assert "| classifier-qwen3 | 1/1 (100%) | 0 | 0.500 | 0.500 | — |" in report
    assert "priced at $0" in report
    assert "No always-frontier baseline" in report


def test_with_every_route_free_cost_saved_is_the_share_of_frontier_calls_avoided() -> None:
    by_arm = {
        BASELINE_ARM: [
            _result(i, BASELINE_ARM, route="frontier", cost=0.0, score=8) for i in "1234"
        ],
        "heuristic": [
            _result(i, "heuristic", route=route, cost=0.0, score=8)
            for i, route in zip("1234", ["small", "small", "small", "frontier"], strict=True)
        ],
    }
    summaries = summarize(by_arm)
    assert summaries["heuristic"].cost_saved_pct == 75.0
    assert summaries["heuristic"].meets_target is True

    report = render_markdown(summaries)
    assert "| Frontier calls avoided |" in report
    assert "| heuristic | 4 | 0 | 0.0000 | 75.0% |" in report
