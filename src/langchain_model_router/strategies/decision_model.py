r"""`DecisionModelStrategy`, opt-in because it makes a call: ask a decision model which route fits.

A decision model (TypeSafe's Jev, open System One models such as Kev, OpenAI's `gpt-6-luna`)
scores a fixed list of options and returns a calibrated probability for each, in one forward pass
and without generating text. That makes it a natural router: the routes are the options, each
described in plain words, and the model's confidence says when to trust its pick. Like
`ClassifierStrategy`, it makes a call every request, so construction takes the application's own
decision model, with no default.

Setting one up
--------------
```python
DecisionModelStrategy(
    my_decision_model,
    {
        "support": "account issues: password resets, billing, cancellations",
        "coder": "programming help: code, debugging, stack traces, refactors",
    },
    api="system_one",
    min_confidence=0.25,
)
```

The two APIs
------------
Decision models speak one of two APIs, with different request bodies, so `api` names the one the
`Runnable` expects. It is required rather than guessed from the `Runnable`, which is usually a
`RunnableLambda` and says nothing about the server behind it. The strategy asks one `choice`
question, named `route`, whose options are the routes `request.routes` and `route_descriptions`
both name, in the router's declaration order (the same rule, and the same `RoutingError` when
there is no overlap, as `ClassifierStrategy`):

- `"system_one"`, TypeSafe's System One API (`POST /v1/systemone`: TypeSafe, OpenRouter, the
  LangSmith gateway, a self-hosted Kev or `ollajev` server)::

      {"state": "<text>", "questions": {"route": {"type": "choice", "instructions": "...",
          "criteria": {"<route>": "<description>", ...}}}}

- `"openai_decisions"`, OpenAI's Decisions API (`POST /v1/decisions`: OpenAI, the LangSmith
  gateway, compatible servers)::

      {"input": "<text>", "questions": [{"type": "choice", "name": "route", "instructions": "...",
          "choices": [{"value": "<route>", "description": "<description>"}, ...]}]}

The body carries no `model`: the model's name, an API key and a timeout belong to the `Runnable`.
The strategy never makes an HTTP call itself.

The answer is read from whatever the `Runnable` returns, in any of these shapes: the System One
response (`answers` maps the question's name to the answer), the Decisions response (`answers` is
a list of answers, each with its `name`), or an object whose `answers` attribute maps the name to
the answer, as `TypeSafeClassifier`'s and `OpenAIDecisions`' results do. The answer itself may be a
dict or an object; it carries `choice`, `confidence` and `probabilities`, the latter a mapping of
option to probability or the Decisions API's list of `{"value", "probability"}`.

Deciding, and not deciding
---------------------------
- **The answer decides** when it names one of the routes and its `confidence` reaches
  `min_confidence`. Every decision-model API reports `confidence` for a `choice` answer, a number
  from 0 to 1 saying how concentrated the probabilities are; it is not the chosen option's
  probability (Kev, with two options, reports `2p - 1`). The reason carries both, so calibrating
  `min_confidence` is a matter of reading reasons.
- **Below `min_confidence`** it abstains, and the router takes the default route.
- **An empty request** — no text — abstains before any call is made.
- **An answer outside the routes, or a refusal**, abstains: it is no answer to the question asked.
- **A failed call** — a network error, a timeout, a response without the answer — is not caught
  here. It propagates out of `decide`/`adecide`, and the router turns it into the default route
  with a `FallbackWarning` naming the cause, as it does for `ClassifierStrategy`.

**Its own call is traced.** The `Runnable` is invoked with `request.config`, so its run nests
under the strategy's run. `adecide` is a native async implementation for the same reason
`ClassifierStrategy`'s is: an async call needs its `config` passed explicitly.

No dependency beyond `langchain-core` is imported.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, Literal

from langchain_model_router.errors import RoutingError
from langchain_model_router.strategies.classifier import _checked_route_descriptions, _offered
from langchain_model_router.strategy import RoutingChoice, RoutingRequest, RoutingStrategy

if TYPE_CHECKING:
    from langchain_core.runnables import Runnable

__all__ = ["DecisionModelStrategy"]

_QUESTION = "route"
_INSTRUCTIONS = "Which route should answer this request?"
_APIS = ("system_one", "openai_decisions")


class DecisionModelStrategy(RoutingStrategy):
    """Routes by asking a decision model which route fits.

    ```python
    ChatRouter(
        routes={"support": support_model, "coder": coder_model},
        default_route="support",
        strategy=DecisionModelStrategy(
            my_decision_model,
            {
                "support": "account issues: password resets, billing, cancellations",
                "coder": "programming help: code, debugging, stack traces, refactors",
            },
            api="system_one",
            min_confidence=0.25,
        ),
    )
    ```

    The module docstring describes the request each API gets, the answer shapes it reads, and
    when it abstains.

    Args:
        model: A `Runnable` that takes the request body of `api` and returns its response, such
            as a `RunnableLambda` around an HTTP call. There is no default, so this strategy can
            never be enabled by accident.
        route_descriptions: Each route's description, at least one: the question's options.
        api: The API the `Runnable` speaks: `"system_one"` or `"openai_decisions"`.
        min_confidence: The answer's `confidence`, from 0 to 1, must reach this to decide;
            below it, the strategy abstains.

    Raises:
        RoutingError: for configuration that could never route — no routes, a blank route or
            description — an unknown `api`, or a `min_confidence` that isn't a number from 0 to 1.
    """

    def __init__(
        self,
        model: Runnable[dict[str, Any], Any],
        route_descriptions: Mapping[str, str],
        *,
        api: Literal["system_one", "openai_decisions"],
        min_confidence: float,
    ) -> None:
        """Check the configuration and keep the decision `model`.

        Args:
            model: The `Runnable` that answers requests in `api`'s format.
            route_descriptions: Each route's description.
            api: The API the `Runnable` speaks.
            min_confidence: The confidence the answer must reach to decide.

        Raises:
            RoutingError: for configuration that could never route, an unknown `api`, or a
                `min_confidence` outside 0 to 1.
        """
        if api not in _APIS:
            msg = f"api must be 'system_one' or 'openai_decisions', got {api!r}"
            raise RoutingError(msg)
        if (
            isinstance(min_confidence, bool)
            or not isinstance(min_confidence, (int, float))
            or not 0 <= min_confidence <= 1
        ):
            msg = f"min_confidence must be a number from 0 to 1, got {min_confidence!r}"
            raise RoutingError(msg)
        self.model = model
        self.route_descriptions = _checked_route_descriptions(
            route_descriptions, "DecisionModelStrategy"
        )
        self.api = api
        self.min_confidence = float(min_confidence)

    def decide(self, request: RoutingRequest) -> RoutingChoice | None:
        """Ask the decision model, or return `None` if there's nothing to ask about.

        Args:
            request: The current request; only its text is sent.

        Returns:
            The route the model chose and why, or `None` to abstain.

        Raises:
            RoutingError: if no route in `route_descriptions` is one the router has.
        """
        choices = _offered(request.routes, self.route_descriptions)
        if not request.text.strip():
            return None
        result = self.model.invoke(self._body(request.text, choices), config=request.config)
        return self._finish(result, choices)

    async def adecide(self, request: RoutingRequest) -> RoutingChoice | None:
        """Async `decide`: a native implementation that passes `request.config` on.

        Args:
            request: The current request; only its text is sent.

        Returns:
            The route the model chose and why, or `None` to abstain.
        """
        choices = _offered(request.routes, self.route_descriptions)
        if not request.text.strip():
            return None
        body = self._body(request.text, choices)
        result = await self.model.ainvoke(body, config=request.config)
        return self._finish(result, choices)

    def _body(self, text: str, choices: tuple[str, ...]) -> dict[str, Any]:
        descriptions = {route: self.route_descriptions[route] for route in choices}
        if self.api == "system_one":
            question = {"type": "choice", "instructions": _INSTRUCTIONS, "criteria": descriptions}
            return {"state": text, "questions": {_QUESTION: question}}
        options = [{"value": route, "description": about} for route, about in descriptions.items()]
        question = {"type": "choice", "name": _QUESTION, "instructions": _INSTRUCTIONS}
        return {"input": text, "questions": [{**question, "choices": options}]}

    def _finish(self, result: Any, choices: tuple[str, ...]) -> RoutingChoice | None:
        """Turn the answer, in any of the shapes the module docstring lists, into a choice."""
        answers = _get(result, "answers")
        if not isinstance(answers, Mapping):  # the Decisions API's list of named answers
            answers = {_get(answer, "name"): answer for answer in answers}
        answer = answers[_QUESTION]
        if _get(answer, "type") == "refusal":
            return None
        route = _get(answer, "choice")
        if route not in choices:
            return None  # an answer outside the options is no answer
        confidence = float(_get(answer, "confidence"))
        if confidence < self.min_confidence:
            return None
        probabilities = _get(answer, "probabilities")
        if not isinstance(probabilities, Mapping):  # the Decisions API's list
            probabilities = {_get(p, "value"): _get(p, "probability") for p in probabilities}
        reason = (
            f"the decision model chose {route!r} (probability {probabilities[route]:.2f}, "
            f"confidence {confidence:.2f}): {self.route_descriptions[route]}"
        )
        return RoutingChoice(route=route, reason=reason)


def _get(value: Any, name: str) -> Any:
    """`value[name]` for a dict, `value.name` for an object; `None` when it has neither."""
    return value.get(name) if isinstance(value, Mapping) else getattr(value, name, None)
