"""The opt-in decision-model strategy: the request it sends, the answers it reads, and fallback.

The decision model is a fake `Runnable` — a `RunnableLambda` that records the request body and
answers in one of the shapes the strategy reads: a System One response (the shape the Kev server
returns), an OpenAI Decisions response (the shape OpenAI's guide documents, probabilities as a
list), or the result object of the LangChain wrappers, whose `answers` maps names to answer
objects.
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


PROBABILITIES = {"coder": 0.8, "support": 0.2}


def system_one_answer(choice: str, confidence: float = 0.9) -> dict[str, Any]:
    """A System One response, the shape the Kev server returns, to the `route` question."""
    answer = {"type": "choice", "choice": choice, "confidence": confidence}
    return {"model": "kev", "answers": {"route": {**answer, "probabilities": PROBABILITIES}}}


def openai_decisions_answer(choice: str, confidence: float = 0.9) -> dict[str, Any]:
    """An OpenAI Decisions response, as OpenAI's guide documents it, to the `route` question."""
    probabilities = [{"value": value, "probability": p} for value, p in PROBABILITIES.items()]
    answer = {"type": "choice", "name": "route", "choice": choice, "confidence": confidence}
    return {
        "model": "gpt-6-luna",
        "answers": [{**answer, "probabilities": probabilities}],
        "usage": {"input_tokens": 40, "output_tokens": 0, "total_tokens": 40},
    }


def wrapper_answer(choice: str, confidence: float = 0.9) -> SimpleNamespace:
    """The result object of `TypeSafeClassifier` or `OpenAIDecisions`: answers by name, as
    attributes, with the Decisions API's probability list already turned into a mapping."""
    answer = SimpleNamespace(
        type="choice", choice=choice, confidence=confidence, probabilities=PROBABILITIES
    )
    return SimpleNamespace(model="m", answers={"route": answer})


ANSWERS = {
    "system_one": system_one_answer,
    "openai_decisions": openai_decisions_answer,
    "wrapper": wrapper_answer,
}
"""Every answer shape, keyed by a test id."""
API_OF = {
    "system_one": "system_one",
    "openai_decisions": "openai_decisions",
    "wrapper": "system_one",
}


class FakeDecisionModel:
    """A decision-model `Runnable` stand-in that records each request body it was sent."""

    def __init__(self, answer: Any) -> None:
        self.bodies: list[dict[str, Any]] = []
        self.answer = answer
        self.runnable = RunnableLambda(self._call, name="FakeDecisionModel")

    def _call(self, body: dict[str, Any]) -> Any:
        self.bodies.append(body)
        if isinstance(self.answer, Exception):
            raise self.answer
        return self.answer


def strategy_answering(
    answer: Any, *, api: Any = "system_one", min_confidence: float = 0.0
) -> tuple[DecisionModelStrategy, list[Any]]:
    model = FakeDecisionModel(answer)
    strategy = DecisionModelStrategy(
        model.runnable, ROUTE_DESCRIPTIONS, api=api, min_confidence=min_confidence
    )
    return strategy, model.bodies


def routing_warnings(caught: list[warnings.WarningMessage]) -> list[warnings.WarningMessage]:
    return [warning for warning in caught if issubclass(warning.category, RoutingWarning)]


# --- Construction ---


def test_construction_requires_the_model_the_descriptions_the_api_and_min_confidence() -> None:
    """No default for any of them: the strategy can't be enabled by accident, and the API and the
    confidence threshold are the application's own choices."""
    model = RunnableLambda(lambda body: body)
    with pytest.raises(TypeError):
        DecisionModelStrategy()  # type: ignore[call-arg]
    with pytest.raises(TypeError):
        DecisionModelStrategy(model)  # type: ignore[call-arg]
    with pytest.raises(TypeError):
        DecisionModelStrategy(model, ROUTE_DESCRIPTIONS, min_confidence=0.5)  # type: ignore[call-arg]
    with pytest.raises(TypeError):
        DecisionModelStrategy(model, ROUTE_DESCRIPTIONS, api="system_one")  # type: ignore[call-arg]


def test_route_descriptions_that_could_never_decide_fail_at_construction() -> None:
    with pytest.raises(
        RoutingError,
        match=r"^route_descriptions is empty: a DecisionModelStrategy needs at least one route$",
    ):
        DecisionModelStrategy(
            RunnableLambda(lambda body: body), {}, api="system_one", min_confidence=0.5
        )


def test_an_unknown_api_fails_at_construction() -> None:
    with pytest.raises(
        RoutingError,
        match=r"^api must be 'system_one' or 'openai_decisions', got 'openai'$",
    ):
        DecisionModelStrategy(
            RunnableLambda(lambda body: body),
            ROUTE_DESCRIPTIONS,
            api="openai",  # type: ignore[arg-type]
            min_confidence=0.5,
        )


@pytest.mark.parametrize("min_confidence", [-0.1, 1.5, "0.5", None, True])
def test_min_confidence_must_be_a_number_from_0_to_1(min_confidence: Any) -> None:
    with pytest.raises(
        RoutingError,
        match=(
            r"^min_confidence must be a number from 0 to 1, "
            rf"got {re.escape(repr(min_confidence))}$"
        ),
    ):
        DecisionModelStrategy(
            RunnableLambda(lambda body: body),
            ROUTE_DESCRIPTIONS,
            api="system_one",
            min_confidence=min_confidence,
        )


@pytest.mark.parametrize("min_confidence", [0, 1, 0.25])
def test_min_confidence_takes_any_number_from_0_to_1(min_confidence: float) -> None:
    strategy, _ = strategy_answering(system_one_answer("coder"), min_confidence=min_confidence)

    assert strategy.min_confidence == min_confidence


# --- The request it sends ---

INSTRUCTIONS = "Which route should answer this request?"


def test_the_system_one_request_is_one_choice_question_over_the_router_s_routes() -> None:
    """The criteria are the routes the router and the descriptions both name, in the router's
    declaration order; a description for a route the router doesn't have is never sent."""
    model = FakeDecisionModel(system_one_answer("coder"))
    descriptions = {**ROUTE_DESCRIPTIONS, "billing": "billing questions"}
    strategy = DecisionModelStrategy(
        model.runnable, descriptions, api="system_one", min_confidence=0.5
    )

    strategy.decide(make_request("fix this stack trace"))

    assert model.bodies == [
        {
            "state": "fix this stack trace",
            "questions": {
                "route": {
                    "type": "choice",
                    "instructions": INSTRUCTIONS,
                    "criteria": {
                        "coder": ROUTE_DESCRIPTIONS["coder"],
                        "support": ROUTE_DESCRIPTIONS["support"],
                    },
                }
            },
        }
    ]
    assert list(model.bodies[0]["questions"]["route"]["criteria"]) == list(ROUTES)


def test_the_openai_decisions_request_is_one_named_choice_question_in_a_list() -> None:
    model = FakeDecisionModel(openai_decisions_answer("coder"))
    descriptions = {**ROUTE_DESCRIPTIONS, "billing": "billing questions"}
    strategy = DecisionModelStrategy(
        model.runnable, descriptions, api="openai_decisions", min_confidence=0.5
    )

    strategy.decide(make_request("fix this stack trace"))

    assert model.bodies == [
        {
            "input": "fix this stack trace",
            "questions": [
                {
                    "type": "choice",
                    "name": "route",
                    "instructions": INSTRUCTIONS,
                    "choices": [
                        {"value": "coder", "description": ROUTE_DESCRIPTIONS["coder"]},
                        {"value": "support", "description": ROUTE_DESCRIPTIONS["support"]},
                    ],
                }
            ],
        }
    ]


def test_no_overlap_between_route_descriptions_and_the_router_s_routes_raises() -> None:
    strategy, bodies = strategy_answering(system_one_answer("coder"))

    with pytest.raises(RoutingError, match="no route in route_descriptions is one this router has"):
        strategy.decide(make_request("fix this stack trace", routes=("billing", "sales")))
    assert bodies == []


# --- The answers it reads, in every shape ---

REASON = (
    "the decision model chose 'coder' (probability 0.80, confidence 0.90): programming help: "
    "code, debugging, stack traces, refactors"
)
SHAPES = list(ANSWERS)


@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("convention", ["decide", "adecide"])
async def test_an_answer_naming_a_route_decides(shape: str, convention: str) -> None:
    strategy, bodies = strategy_answering(ANSWERS[shape]("coder"), api=API_OF[shape])
    request = make_request("fix this stack trace")

    choice = strategy.decide(request) if convention == "decide" else await strategy.adecide(request)

    assert choice == RoutingChoice("coder", REASON)
    assert len(bodies) == 1


@pytest.mark.parametrize("shape", SHAPES)
def test_an_answer_outside_the_routes_abstains(shape: str) -> None:
    strategy, _ = strategy_answering(ANSWERS[shape]("billing"), api=API_OF[shape])

    assert strategy.decide(make_request("fix this stack trace")) is None


def test_a_refusal_abstains() -> None:
    """The Decisions API answers a question it declines with a `refusal` instead of a choice."""
    refusal = {"model": "gpt-6-luna", "answers": [{"type": "refusal", "name": "route"}]}
    strategy, _ = strategy_answering(refusal, api="openai_decisions")

    assert strategy.decide(make_request("fix this stack trace")) is None


@pytest.mark.parametrize("shape", SHAPES)
def test_confidence_is_what_min_confidence_is_compared_with(shape: str) -> None:
    """Not the probability: 0.8 for 'coder' would clear 0.6, its confidence of 0.5 doesn't."""
    below, _ = strategy_answering(
        ANSWERS[shape]("coder", 0.5), api=API_OF[shape], min_confidence=0.6
    )
    at, _ = strategy_answering(ANSWERS[shape]("coder", 0.5), api=API_OF[shape], min_confidence=0.5)

    assert below.decide(make_request("fix this stack trace")) is None
    choice = at.decide(make_request("fix this stack trace"))
    assert choice is not None
    assert choice.route == "coder"


@pytest.mark.parametrize("convention", ["decide", "adecide"])
async def test_an_empty_request_abstains_without_a_call(convention: str) -> None:
    strategy, bodies = strategy_answering(system_one_answer("coder"))
    request = make_request("   ")

    choice = strategy.decide(request) if convention == "decide" else await strategy.adecide(request)

    assert (choice, bodies) == (None, [])


# --- Through the router, every calling convention ---


@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("convention", CONVENTIONS)
async def test_through_the_router_the_decision_model_s_choice_answers(
    convention: Convention, shape: str
) -> None:
    strategy, _ = strategy_answering(ANSWERS[shape]("coder"), api=API_OF[shape])
    router = ChatRouter(routes=fake_routes(), default_route="support", strategy=strategy)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        message = await respond(router, convention, "fix this stack trace")

    assert message.content == "coder answer"
    assert routing_warnings(caught) == []
    assert routing_decision(message) == RoutingDecision(
        route="coder", reason=REASON, strategy="DecisionModelStrategy"
    )


@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("convention", CONVENTIONS)
async def test_below_min_confidence_the_router_takes_the_default_route(
    convention: Convention, shape: str
) -> None:
    strategy, _ = strategy_answering(
        ANSWERS[shape]("coder", 0.2), api=API_OF[shape], min_confidence=0.25
    )
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
