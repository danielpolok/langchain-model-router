"""Aggregate `ItemResult`s into the report the task asks for: cost and quality per strategy
against the always-frontier baseline, and a verdict on the target (>=30% lower cost at
>=95% of baseline quality), plus each arm's routing accuracy and decision latency.

When the baseline cost $0 (every route a local model), there are no dollars to compare, so the
cost figure becomes the share of frontier calls avoided against the baseline, which sends every
item to the frontier route. That is the dollar saving only if a small call costs nothing next to
a frontier call; the report says so."""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass

from benchmark.runner import ItemResult

BASELINE_ARM = "baseline-always-frontier"
FRONTIER_ROUTE = "frontier"
COST_TARGET_PCT = 30.0
QUALITY_TARGET_PCT = 95.0


@dataclass(frozen=True)
class ArmSummary:
    """One arm's numbers, and its two derived figures against the baseline."""

    name: str
    n_items: int
    n_errors: int
    total_cost_usd: float
    mean_quality: float | None
    """Mean judge score out of 10, over items that graded successfully. `None` if none did."""
    routed: int
    """Items with both a decision and an expected route — what routing accuracy is out of."""
    routed_correctly: int
    """How many of those took the route the dataset expects (a fallback counts by its route)."""
    fallbacks: int
    """Decisions where the strategy abstained or failed and the default route ran."""
    decision_p50_s: float | None
    decision_p95_s: float | None
    """Median and 95th-percentile (nearest rank) strategy decision time, in seconds."""
    agent_items: int
    tool_correct: int
    """How many agent items called the tool `expected_tool` named."""
    frontier_calls: int = 0
    """Items answered on `FRONTIER_ROUTE`."""
    cost_saved_pct: float | None = None
    """vs `BASELINE_ARM`'s cost, or its frontier calls when the baseline cost $0. `None` for the
    baseline itself, or if the baseline is missing."""
    quality_pct_of_baseline: float | None = None
    """vs `BASELINE_ARM`'s mean quality. Same `None` cases."""

    @property
    def meets_target(self) -> bool | None:
        if self.cost_saved_pct is None or self.quality_pct_of_baseline is None:
            return None
        return self.cost_saved_pct >= COST_TARGET_PCT and self.quality_pct_of_baseline >= (
            QUALITY_TARGET_PCT
        )

    @property
    def routing_accuracy(self) -> float | None:
        return self.routed_correctly / self.routed if self.routed else None

    @property
    def tool_accuracy(self) -> float | None:
        return self.tool_correct / self.agent_items if self.agent_items else None


def _p95(values: list[float]) -> float:
    return sorted(values)[math.ceil(0.95 * len(values)) - 1]


def _summarize_one(name: str, results: list[ItemResult]) -> ArmSummary:
    scores = [r.judge.score for r in results if r.judge is not None]
    agent = [r for r in results if r.tool_correct is not None]
    routed = [r for r in results if r.decision and r.expected_route]
    times = [r.decision_seconds for r in results if r.decision_seconds is not None]
    return ArmSummary(
        name=name,
        n_items=len(results),
        n_errors=sum(1 for r in results if r.error is not None),
        total_cost_usd=sum(r.cost.total_usd for r in results),
        mean_quality=(sum(scores) / len(scores)) if scores else None,
        routed=len(routed),
        routed_correctly=sum(
            1 for r in routed if r.decision and r.decision.route == r.expected_route
        ),
        fallbacks=sum(1 for r in results if r.decision and r.decision.fallback),
        decision_p50_s=statistics.median(times) if times else None,
        decision_p95_s=_p95(times) if times else None,
        agent_items=len(agent),
        tool_correct=len([r for r in agent if r.tool_correct]),
        frontier_calls=sum(1 for r in results if r.decision and r.decision.route == FRONTIER_ROUTE),
    )


def _by_frontier_calls(summaries: dict[str, ArmSummary]) -> bool:
    """Whether cost is measured as frontier calls avoided: the baseline ran and cost $0."""
    baseline = summaries.get(BASELINE_ARM)
    return baseline is not None and baseline.total_cost_usd == 0


def summarize(by_arm: dict[str, list[ItemResult]]) -> dict[str, ArmSummary]:
    """One `ArmSummary` per arm, with `cost_saved_pct` / `quality_pct_of_baseline` filled in
    against `BASELINE_ARM` wherever it's present in `by_arm`."""
    summaries = {name: _summarize_one(name, results) for name, results in by_arm.items()}
    baseline = summaries.get(BASELINE_ARM)
    if baseline is None or baseline.mean_quality is None:
        return summaries
    by_calls = _by_frontier_calls(summaries)
    if by_calls and baseline.frontier_calls == 0:
        return summaries

    filled = {}
    for name, summary in summaries.items():
        if name == BASELINE_ARM or summary.mean_quality is None:
            filled[name] = summary
            continue
        if by_calls:
            cost_saved = (1 - summary.frontier_calls / baseline.frontier_calls) * 100
        else:
            cost_saved = (1 - summary.total_cost_usd / baseline.total_cost_usd) * 100
        quality_pct = (summary.mean_quality / baseline.mean_quality) * 100
        filled[name] = ArmSummary(
            **{
                **summary.__dict__,
                "cost_saved_pct": cost_saved,
                "quality_pct_of_baseline": quality_pct,
            }
        )
    filled[BASELINE_ARM] = baseline
    return filled


def _fmt_pct(value: float | None) -> str:
    return f"{value:.1f}%" if value is not None else "—"


def _fmt_target(summary: ArmSummary) -> str:
    if summary.name == BASELINE_ARM:
        return "—"
    met = summary.meets_target
    if met is None:
        return "n/a"
    return "**met**" if met else "not met"


def _fmt_seconds(value: float | None) -> str:
    return f"{value:.3f}" if value is not None else "—"


def render_markdown(summaries: dict[str, ArmSummary], *, title: str = "Benchmark results") -> str:
    """The markdown report: cost and quality, routing accuracy and decision latency, and a
    verdict."""
    rows = sorted(summaries.values(), key=lambda summary: summary.name != BASELINE_ARM)
    by_calls = _by_frontier_calls(summaries)
    saved = "Frontier calls avoided" if by_calls else "Cost saved"

    lines = [
        f"# {title}",
        "",
        f"Target: >= {COST_TARGET_PCT:.0f}% lower cost at "
        f">= {QUALITY_TARGET_PCT:.0f}% of always-frontier quality.",
        "",
        f"| Arm | Items | Errors | Cost ($) | {saved} | Mean quality (/10) "
        "| Quality vs baseline | Target |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for summary in rows:
        quality = f"{summary.mean_quality:.2f}" if summary.mean_quality is not None else "—"
        lines.append(
            f"| {summary.name} | {summary.n_items} | {summary.n_errors} | "
            f"{summary.total_cost_usd:.4f} | {_fmt_pct(summary.cost_saved_pct)} | {quality} | "
            f"{_fmt_pct(summary.quality_pct_of_baseline)} | {_fmt_target(summary)} |"
        )

    lines += [
        "",
        "Local Ollama models are priced at $0: their cost excludes the hardware and power that "
        "run them.",
        "",
    ]
    if by_calls:
        lines += [
            "Every route in this run is local, so cost is measured as the share of items kept "
            "off the frontier route, against the baseline's every item: the dollar saving only "
            "if a small call is free next to a frontier one. A decision model's own calls are "
            "not counted; see its decision latency.",
            "",
        ]
    lines += [
        "## Routing and tool-use detail",
        "",
        "| Arm | Routing accuracy | Fallbacks | Decision p50 (s) | Decision p95 (s) "
        "| Agent tool accuracy |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for summary in rows:
        accuracy = (
            f"{summary.routed_correctly}/{summary.routed} ({summary.routing_accuracy:.0%})"
            if summary.routing_accuracy is not None
            else "—"
        )
        tool_acc = f"{summary.tool_accuracy:.0%}" if summary.tool_accuracy is not None else "—"
        lines.append(
            f"| {summary.name} | {accuracy} | {summary.fallbacks} | "
            f"{_fmt_seconds(summary.decision_p50_s)} | {_fmt_seconds(summary.decision_p95_s)} | "
            f"{tool_acc} |"
        )

    met = [s.name for s in rows if s.meets_target]
    lines += ["", "## Verdict", ""]
    if BASELINE_ARM not in summaries:
        lines.append(
            "No always-frontier baseline in this run, so no cost/quality verdict: compare the "
            "arms on routing accuracy and decision latency."
        )
    elif any(s.name != BASELINE_ARM and s.meets_target is None for s in rows):
        lines.append(
            "Incomplete: at least one strategy has no baseline-relative figures to judge "
            "against (see the table above)."
        )
    elif met:
        lines.append(f"Target met by: {', '.join(met)}.")
    else:
        lines.append("Target not met by any strategy on this run.")

    return "\n".join(lines) + "\n"
