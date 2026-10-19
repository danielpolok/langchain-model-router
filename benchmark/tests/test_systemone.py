"""`SystemOneStrategy` against a stub `/v1/systemone` server, over real HTTP and offline."""

from __future__ import annotations

import json
import threading
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any

import pytest

from benchmark.systemone import SystemOneStrategy
from langchain_model_router import ChatRouter
from tests.fakes import NativeStructuredFakeChatModel

DESCRIPTIONS = {"support": "orders and refunds", "coder": "code and stack traces"}
seen: list[dict[str, Any]] = []
answer: dict[str, Any] = {}


class _Handler(BaseHTTPRequestHandler):
    def do_POST(self) -> None:
        seen.append(json.loads(self.rfile.read(int(self.headers["content-length"]))))
        body = json.dumps({"answers": {"route": answer}}).encode()
        self.send_response(200)
        self.send_header("content-length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args: object) -> None:
        pass


@pytest.fixture
def url() -> Iterator[str]:
    seen.clear()
    server = HTTPServer(("127.0.0.1", 0), _Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{server.server_port}/v1/systemone"
    server.shutdown()
    server.server_close()


def _router(strategy: SystemOneStrategy) -> ChatRouter:
    def model() -> NativeStructuredFakeChatModel:
        return NativeStructuredFakeChatModel(model_name="fake", reply="ok")

    return ChatRouter(
        routes={"support": model(), "coder": model()}, default_route="support", strategy=strategy
    )


def test_the_highest_probability_route_runs_and_the_request_carries_the_routes(url: str) -> None:
    answer.clear()
    answer.update(
        type="choice", choice="coder", confidence=0.9, probabilities={"support": 0.1, "coder": 0.9}
    )
    reply = _router(SystemOneStrategy(url, DESCRIPTIONS)).invoke("fix this KeyError")

    assert reply.response_metadata["routing"]["route"] == "coder"
    assert "90%" in reply.response_metadata["routing"]["reason"]
    question = seen[0]["questions"]["route"]
    assert seen[0]["state"] == "fix this KeyError"
    assert question["type"] == "choice"
    assert question["criteria"] == DESCRIPTIONS


def test_an_answer_outside_the_options_abstains_to_the_default_route(url: str) -> None:
    answer.clear()
    answer.update(type="choice", choice="other", confidence=0.5, probabilities={})
    with pytest.warns(UserWarning, match="could not decide"):
        reply = _router(SystemOneStrategy(url, DESCRIPTIONS)).invoke("hello")
    assert reply.response_metadata["routing"]["route"] == "support"
    assert reply.response_metadata["routing"]["fallback"] is True


def test_an_unreachable_server_falls_back_with_a_warning() -> None:
    strategy = SystemOneStrategy("http://127.0.0.1:1/v1/systemone", DESCRIPTIONS, timeout=1)
    with pytest.warns(UserWarning, match="falling back"):
        reply = _router(strategy).invoke("hello")
    assert reply.response_metadata["routing"]["fallback"] is True
