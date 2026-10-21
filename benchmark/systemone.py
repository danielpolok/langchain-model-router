"""A benchmark-only strategy for "System One" decision models (Jev, JevK5, Kev).

These models score a fixed list of options and return a probability for each, in one forward
pass and without generating text, behind a `POST /v1/systemone` endpoint. Two local servers expose
it: `ollajev serve` (default `127.0.0.1:8000`) and the Kev-enabled `llama-server` (default
`127.0.0.1:8080`). The request is one `choice` question whose `criteria` map each route to its
description, and the answer carries the chosen label and every label's probability:

    {"state": "...", "questions": {"route": {"type": "choice", "instructions": "...",
        "criteria": {"support": "...", "coder": "..."}}}}
    -> {"answers": {"route": {"type": "choice", "choice": "coder", "confidence": 0.9,
        "probabilities": {"support": 0.1, "coder": 0.9}}}}

It lives here, not in the package: it needs no dependency (standard-library `urllib`) but it is a
strategy for one benchmark question, not something the package promises.
"""

from __future__ import annotations

import json
import urllib.request
from collections.abc import Mapping

from langchain_model_router import RoutingChoice, RoutingRequest, RoutingStrategy

_QUESTION = "Which route should answer this request?"


class SystemOneStrategy(RoutingStrategy):
    """Routes to the option a System One model gives the highest probability."""

    def __init__(
        self,
        url: str,
        route_descriptions: Mapping[str, str],
        *,
        model: str | None = None,
        timeout: float = 120.0,
    ) -> None:
        self.url = url
        self.route_descriptions = dict(route_descriptions)
        self.model = model
        self.timeout = timeout

    def decide(self, request: RoutingRequest) -> RoutingChoice | None:
        if not request.text.strip():
            return None
        criteria = {
            route: self.route_descriptions[route]
            for route in request.routes
            if route in self.route_descriptions
        }
        body: dict[str, object] = {
            "state": request.text,
            "questions": {
                "route": {"type": "choice", "instructions": _QUESTION, "criteria": criteria}
            },
        }
        if self.model:
            body["model"] = self.model
        http_request = urllib.request.Request(
            self.url,
            data=json.dumps(body).encode(),
            headers={"content-type": "application/json"},
        )
        with urllib.request.urlopen(http_request, timeout=self.timeout) as response:
            answer = json.loads(response.read())["answers"]["route"]
        route = answer["choice"]
        if route not in criteria:
            return None  # an answer outside the options is no answer: abstain
        probability = answer["probabilities"].get(route, answer.get("confidence", 0.0))
        return RoutingChoice(
            route=route, reason=f"the decision model chose {route!r} ({probability:.0%})"
        )
