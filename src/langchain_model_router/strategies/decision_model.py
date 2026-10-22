r"""`DecisionModelStrategy`, opt-in because it makes a call: ask a decision model which route fits.

A decision model (TypeSafe's Jev, or an open System One model such as Kev) scores a fixed list of
options and returns a calibrated probability for each, in one forward pass and without generating
text. That makes it a natural router: the routes are the options, each described in plain words,
and the model's confidence says when to trust its pick. Like `ClassifierStrategy`, it makes a call
every request, so construction takes the application's own decision model, with no default.

Setting one up
--------------
```python
DecisionModelStrategy(
    my_decision_model,
    {
        "support": "account issues: password resets, billing, cancellations",
        "coder": "programming help: code, debugging, stack traces, refactors",
    },
    min_confidence=0.25,
)
```

The model it takes
------------------
Any `Runnable` that speaks the System One API: it takes the request body as a dict and returns
the answer. The strategy never makes an HTTP call itself, so the application chooses the client —
`TypeSafeClassifier` from `langchain-typesafe`, or a `RunnableLambda` around its own call to a
local server. It sends one `choice` question, named `route`:

    {"state": "<the request's text>", "questions": {"route": {"type": "choice",
        "instructions": "Which route should answer this request?",
        "criteria": {"<route>": "<description>", ...}}}}

and reads the answer to it from either of the two shapes such a `Runnable` returns: the raw
System One response, `{"answers": {"route": {...}}}`, or an object whose `choices["route"]` is the
answer. The answer carries `choice`, `confidence` and `probabilities`, as keys or as attributes.
The model's name, a timeout or an API key belong to the `Runnable`, not to this strategy.

The criteria are the routes that `request.routes` and `route_descriptions` both name, in the
router's own declaration order — the same rule, and the same `RoutingError` when there is no
overlap, as `ClassifierStrategy`.

Deciding, and not deciding
---------------------------
- **The answer decides** when it names one of the options and its `confidence` reaches
  `min_confidence`. The reason carries the chosen option's probability and the confidence, so a
  trace shows how sure the model was, and calibrating `min_confidence` is a matter of reading
  reasons.
- **Below `min_confidence`** it abstains, and the router takes the default route.
- **An empty request** — no text — abstains before any call is made.
- **An answer outside the options** abstains: it is no answer to the question asked.
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
from typing import TYPE_CHECKING, Any

from langchain_model_router.errors import RoutingError
from langchain_model_router.strategies.classifier import _checked_route_descriptions, _offered
from langchain_model_router.strategy import RoutingChoice, RoutingRequest, RoutingStrategy

if TYPE_CHECKING:
    from langchain_core.runnables import Runnable

__all__ = ["DecisionModelStrategy"]

_QUESTION = "route"
_INSTRUCTIONS = "Which route should answer this request?"


class DecisionModelStrategy(RoutingStrategy):
    """Routes by asking a decision model, through the System One API, which route fits.

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
            min_confidence=0.25,
        ),
    )
    ```

    The module docstring describes the request it sends, the two answer shapes it reads, and
    when it abstains.

    Args:
        model: A `Runnable` that takes a System One request body and returns its answer, such as
            `TypeSafeClassifier` or a `RunnableLambda` around an HTTP call. There is no default,
            so this strategy can never be enabled by accident.
        route_descriptions: Each route's description, at least one: the question's criteria.
        min_confidence: The confidence the answer must reach to decide; below it, the strategy
            abstains. Defaults to `0`, so every answer that names an option decides.

    Raises:
        RoutingError: for configuration that could never route — no routes, a blank route or
            description — or a `min_confidence` that isn't a number.
    """

    def __init__(
        self,
        model: Runnable[dict[str, Any], Any],
        route_descriptions: Mapping[str, str],
        *,
        min_confidence: float = 0.0,
    ) -> None:
        """Check the configuration and keep the decision `model`.

        Args:
            model: The `Runnable` that answers System One requests.
            route_descriptions: Each route's description.
            min_confidence: The confidence the answer must reach to decide.

        Raises:
            RoutingError: for configuration that could never route, or a non-numeric
                `min_confidence`.
        """
        if isinstance(min_confidence, bool) or not isinstance(min_confidence, (int, float)):
            msg = f"min_confidence must be a number, got {min_confidence!r}"
            raise RoutingError(msg)
        self.model = model
        self.route_descriptions = _checked_route_descriptions(
            route_descriptions, "DecisionModelStrategy"
        )
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
        criteria = {route: self.route_descriptions[route] for route in choices}
        question = {"type": "choice", "instructions": _INSTRUCTIONS, "criteria": criteria}
        return {"state": text, "questions": {_QUESTION: question}}

    def _finish(self, result: Any, choices: tuple[str, ...]) -> RoutingChoice | None:
        """Turn the answer, in either shape, into a choice, or `None` to abstain."""
        answers = result["answers"] if isinstance(result, Mapping) else result.choices
        answer = answers[_QUESTION]

        def field(name: str) -> Any:
            return answer[name] if isinstance(answer, Mapping) else getattr(answer, name)

        route = field("choice")
        if route not in choices:
            return None  # an answer outside the options is no answer
        confidence = float(field("confidence"))
        if confidence < self.min_confidence:
            return None
        probability = float(field("probabilities")[route])
        reason = (
            f"the decision model chose {route!r} (probability {probability:.2f}, confidence "
            f"{confidence:.2f}): {self.route_descriptions[route]}"
        )
        return RoutingChoice(route=route, reason=reason)
