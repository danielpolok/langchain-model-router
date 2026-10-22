"""The opt-in decision-model strategy: the request it sends, the answers it reads, and fallback.

The decision model is a fake `Runnable` — a `RunnableLambda` that records the System One request
body and answers in one of the two shapes the strategy reads: the raw System One response dict,
or an object whose `choices` holds the answer, as `TypeSafeClassifier`'s result does.
"""

from __future__ import annotations

import re
import warnings
from types import SimpleNamespace
from typing import Any

import pytest
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig, RunnableLambda
from langchain_core.tracers.run_collector import RunCollectorCallbackHandler

from langchain_model_router import (
    ChatRouter,
    DecisionModelStrategy,
    FallbackWarning,
    RoutingChoice,
    RoutingDecision,
    RoutingError,
    RoutingRequest,
    RoutingWarning,
    routing_decision,
)
from langchain_model_router._extraction import build_request
from tests.conventions import CONVENTIONS, Convention, respond
from tests.fakes import FakeChatModel
from tests.tracing import model_name_of, model_runs

ROUTE_DESCRIPTIONS = {
    "support": "account issues: password resets, billing, cancellations",
    "coder": "programming help: code, debugging, stack traces, refactors",
}
ROUTES = ("coder", "support")


def make_request(text: str, *, routes: tuple[str, ...] = ROUTES) -> RoutingRequest:
    """The request the router would build from a one-turn conversation."""
    request = build_request(
        [HumanMessage(text)],
        routes=routes,
        tools_bound=False,
        wants_full_context=False,
        config=RunnableConfig(),
    )
    assert request is not None
    return request


def fake_routes() -> dict[str, BaseChatModel]:
    return {
        "coder": FakeChatModel(model_name="model-coder", reply="coder answer"),
        "support": FakeChatModel(model_name="model-support", reply="support answer"),
    }


def system_one_answer(choice: str, confidence: float = 0.9) -> dict[str, Any]:
    """The System One response to the strategy's `route` question."""
    probabilities = {"coder": 0.8, "support": 0.2}
    answer = {"type": "choice", "choice": choice, "confidence": confidence}
    return {"answers": {"route": {**answer, "probabilities": probabilities}}}


class FakeDecisionModel:
    """A System One `Runnable` stand-in that records each request body it was sent."""

    def __init__(self, answer: Any) -> None:
        self.bodies: list[dict[str, Any]] = []
        self.answer = answer
        self.runnable = RunnableLambda(self._call, name="FakeDecisionModel")

    def _call(self, body: dict[str, Any]) -> Any:
        self.bodies.append(body)
        if isinstance(self.answer, Exception):
            raise self.answer
        return self.answer


def strategy_answering(answer: Any, **kwargs: Any) -> tuple[DecisionModelStrategy, list[Any]]:
    model = FakeDecisionModel(answer)
    return DecisionModelStrategy(model.runnable, ROUTE_DESCRIPTIONS, **kwargs), model.bodies


def routing_warnings(caught: list[warnings.WarningMessage]) -> list[warnings.WarningMessage]:
    return [warning for warning in caught if issubclass(warning.category, RoutingWarning)]


# --- Construction ---


def test_construction_requires_the_model_and_the_route_descriptions() -> None:
    """No default for either -- this strategy can never be enabled by accident."""
    with pytest.raises(TypeError):
        DecisionModelStrategy()  # type: ignore[call-arg]
    with pytest.raises(TypeError):
        DecisionModelStrategy(RunnableLambda(lambda body: body))  # type: ignore[call-arg]


def test_route_descriptions_that_could_never_decide_fail_at_construction() -> None:
    with pytest.raises(
        RoutingError,
        match=r"^route_descriptions is empty: a DecisionModelStrategy needs at least one route$",
    ):
        DecisionModelStrategy(RunnableLambda(lambda body: body), {})


@pytest.mark.parametrize("min_confidence", ["0.5", None, True])
def test_min_confidence_must_be_a_number(min_confidence: Any) -> None:
    with pytest.raises(
        RoutingError,
        match=rf"^min_confidence must be a number, got {re.escape(repr(min_confidence))}$",
    ):
        DecisionModelStrategy(
            RunnableLambda(lambda body: body), ROUTE_DESCRIPTIONS, min_confidence=min_confidence
        )


# --- The request it sends ---


def test_it_asks_one_choice_question_over_the_router_s_routes_in_the_router_s_order() -> None:
    """The criteria are the routes the router and the descriptions both name, in the router's
    declaration order; a description for a route the router doesn't have is never sent."""
    model = FakeDecisionModel(system_one_answer("coder"))
    descriptions = {**ROUTE_DESCRIPTIONS, "billing": "billing questions"}
    strategy = DecisionModelStrategy(model.runnable, descriptions)

    strategy.decide(make_request("fix this stack trace"))

    assert model.bodies == [
        {
            "state": "fix this stack trace",
            "questions": {
                "route": {
                    "type": "choice",
                    "instructions": "Which route should answer this request?",
                    "criteria": {
                        "coder": ROUTE_DESCRIPTIONS["coder"],
                        "support": ROUTE_DESCRIPTIONS["support"],
                    },
                }
            },
        }
    ]
    assert list(model.bodies[0]["questions"]["route"]["criteria"]) == list(ROUTES)


def test_no_overlap_between_route_descriptions_and_the_router_s_routes_raises() -> None:
    strategy, bodies = strategy_answering(system_one_answer("coder"))

    with pytest.raises(RoutingError, match="no route in route_descriptions is one this router has"):
        strategy.decide(make_request("fix this stack trace", routes=("billing", "sales")))
    assert bodies == []


# --- The answers it reads ---

REASON = (
    "the decision model chose 'coder' (probability 0.80, confidence 0.90): programming help: "
    "code, debugging, stack traces, refactors"
)


def test_a_system_one_response_decides() -> None:
    strategy, _ = strategy_answering(system_one_answer("coder"))

    assert strategy.decide(make_request("fix this stack trace")) == RoutingChoice("coder", REASON)


def test_an_answer_object_with_choices_decides_too() -> None:
    """The shape `TypeSafeClassifier` returns: the answer under `choices`, read by attribute."""
    answer = SimpleNamespace(choice="coder", confidence=0.9, probabilities={"coder": 0.8})
    strategy, _ = strategy_answering(SimpleNamespace(choices={"route": answer}))

    assert strategy.decide(make_request("fix this stack trace")) == RoutingChoice("coder", REASON)


async def test_adecide_decides_natively() -> None:
    strategy, bodies = strategy_answering(system_one_answer("coder"))

    assert await strategy.adecide(make_request("fix this stack trace")) == RoutingChoice(
        "coder", REASON
    )
    assert len(bodies) == 1


@pytest.mark.parametrize("convention", ["decide", "adecide"])
async def test_an_answer_outside_the_options_abstains(convention: str) -> None:
    strategy, _ = strategy_answering(system_one_answer("billing"))
    request = make_request("fix this stack trace")

    choice = strategy.decide(request) if convention == "decide" else await strategy.adecide(request)

    assert choice is None


@pytest.mark.parametrize("convention", ["decide", "adecide"])
async def test_an_empty_request_abstains_without_a_call(convention: str) -> None:
    strategy, bodies = strategy_answering(system_one_answer("coder"))
    request = make_request("   ")

    choice = strategy.decide(request) if convention == "decide" else await strategy.adecide(request)

    assert (choice, bodies) == (None, [])


def test_confidence_at_min_confidence_decides() -> None:
    strategy, _ = strategy_answering(system_one_answer("coder", 0.5), min_confidence=0.5)

    choice = strategy.decide(make_request("fix this stack trace"))

    assert choice is not None
    assert choice.route == "coder"


# --- Through the router, every calling convention ---


@pytest.mark.parametrize("convention", CONVENTIONS)
async def test_through_the_router_the_decision_model_s_choice_answers(
    convention: Convention,
) -> None:
    strategy, _ = strategy_answering(system_one_answer("coder"))
    router = ChatRouter(routes=fake_routes(), default_route="support", strategy=strategy)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        message = await respond(router, convention, "fix this stack trace")

    assert message.content == "coder answer"
    assert routing_warnings(caught) == []
    assert routing_decision(message) == RoutingDecision(
        route="coder", reason=REASON, strategy="DecisionModelStrategy"
    )


@pytest.mark.parametrize("convention", CONVENTIONS)
async def test_below_min_confidence_the_router_takes_the_default_route(
    convention: Convention,
) -> None:
    strategy, _ = strategy_answering(system_one_answer("coder", 0.2), min_confidence=0.25)
    router = ChatRouter(routes=fake_routes(), default_route="support", strategy=strategy)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        message = await respond(router, convention, "fix this stack trace")

    assert message.content == "support answer"
    (warning,) = routing_warnings(caught)
    assert warning.category is FallbackWarning
    assert str(warning.message) == (
        "DecisionModelStrategy could not decide; falling back to the default route 'support'"
    )
    record = routing_decision(message)
    assert record is not None
    assert (record.route, record.fallback) == ("support", True)


@pytest.mark.parametrize("convention", CONVENTIONS)
async def test_a_failed_call_surfaces_unchanged_as_a_fallback(convention: Convention) -> None:
    strategy, _ = strategy_answering(ConnectionError("down"))
    router = ChatRouter(routes=fake_routes(), default_route="support", strategy=strategy)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        message = await respond(router, convention, "fix this stack trace")

    assert message.content == "support answer"
    (warning,) = routing_warnings(caught)
    assert str(warning.message) == (
        "DecisionModelStrategy raised ConnectionError: down; falling back to the default route "
        "'support'"
    )


def test_the_failed_call_itself_raises_unchanged() -> None:
    error = ConnectionError("down")
    strategy, _ = strategy_answering(error)

    with pytest.raises(ConnectionError) as raised:
        strategy.decide(make_request("fix this stack trace"))
    assert raised.value is error


# --- Tracing: the decision model's run nests under the strategy's, under the router's ---


@pytest.mark.parametrize("convention", CONVENTIONS)
async def test_the_decision_model_s_run_nests_under_the_router_s_run(
    convention: Convention,
) -> None:
    """Router run -> strategy run -> the decision model's own run; the route's call stays
    directly under the router's run and is the only model run, so nothing is billed twice."""
    strategy, _ = strategy_answering(system_one_answer("coder"))
    router = ChatRouter(routes=fake_routes(), default_route="support", strategy=strategy)
    collector = RunCollectorCallbackHandler()

    await respond(router, convention, "fix this stack trace", {"callbacks": [collector]})

    (router_run,) = collector.traced_runs
    strategy_run, route_run = router_run.child_runs
    assert strategy_run.name == "DecisionModelStrategy"
    (decision_run,) = strategy_run.child_runs
    assert (decision_run.name, decision_run.run_type) == ("FakeDecisionModel", "chain")
    assert decision_run.outputs == system_one_answer("coder")
    assert [model_name_of(run) for run in model_runs(collector.traced_runs)] == ["model-coder"]
    assert model_runs([route_run]) == [route_run]
