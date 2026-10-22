"""The decision-model strategy against a real System One server: Kev 0.8B behind `llama-server`.

What only a real model can show: the System One request body the strategy builds is one a real
server accepts, its answer has the `choice`, `confidence` and `probabilities` the strategy reads,
and a clear request is routed where it belongs. The routes are fakes, since only the decision
model's choice is under test. Skips when no server listens at `SYSTEM_ONE_URL`
(`requires_system_one`, root `conftest.py`).
"""

from __future__ import annotations

import json
import urllib.request
from typing import Any

import pytest
from langchain_core.runnables import RunnableLambda
from langchain_core.tracers.run_collector import RunCollectorCallbackHandler

from langchain_model_router import ChatRouter, DecisionModelStrategy, routing_decision
from tests.fakes import FakeChatModel
from tests.system_one import SYSTEM_ONE_URL

ROUTE_DESCRIPTIONS = {
    "general": "anything that isn't programming",
    "coder": "programming: code, bugs, errors, SQL, tooling",
}


def system_one(body: dict[str, Any]) -> dict[str, Any]:
    """POST the System One request body, return the response."""
    request = urllib.request.Request(
        SYSTEM_ONE_URL,
        data=json.dumps(body).encode(),
        headers={"content-type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        answer: dict[str, Any] = json.load(response)
        return answer


@pytest.mark.requires_system_one
@pytest.mark.parametrize(
    ("question", "route"),
    [
        ("My Python loop never terminates, what's wrong with it?", "coder"),
        ("Recommend a good book about the Roman Empire", "general"),
    ],
)
async def test_a_real_decision_model_routes_a_clear_request(question: str, route: str) -> None:
    strategy = DecisionModelStrategy(
        RunnableLambda(system_one, name="Kev"),
        ROUTE_DESCRIPTIONS,
        api="system_one",
        min_confidence=0.0,  # this test is about the choice; the threshold has its own tests
    )
    router = ChatRouter(
        routes={
            name: FakeChatModel(model_name=f"model-{name}", reply=f"{name} answer")
            for name in ROUTE_DESCRIPTIONS
        },
        default_route="general",
        strategy=strategy,
    )
    collector = RunCollectorCallbackHandler()

    message = await router.ainvoke(question, config={"callbacks": [collector]})

    decision = routing_decision(message)
    assert decision is not None
    assert (decision.route, decision.fallback) == (route, False)
    assert decision.reason.startswith(f"the decision model chose {route!r} (probability ")
    (router_run,) = collector.traced_runs
    strategy_run, _ = router_run.child_runs
    assert [run.name for run in strategy_run.child_runs] == ["Kev"]
