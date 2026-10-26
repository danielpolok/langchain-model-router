"""The baseline and the strategies the benchmark compares.

Two candidate routes, both local Ollama models by default so the benchmark runs with no API key
and no paid call: `"small"` (`ollama:qwen3:8b`, thinking off) and `"frontier"`
(`ollama:gpt-oss:20b`, which always reasons). The baseline always answers on `"frontier"`; each
strategy decides per request whether `"small"` is safe.

Every strategy that needs no paid model runs by default: `keyword`, `heuristic`,
`ClassifierStrategy` on two local decision models (`classifier-qwen3`, and
`classifier-arch-router` on Katanemo's Arch-Router-1.5B), and one `SystemOneStrategy` arm per
configured System One endpoint. Gemini is an explicit opt-in: `MODEL_ROUTER_BENCHMARK_GEMINI=1`
switches the routes back to the Gemini pair and adds the `embedding` arm (Gemini embeddings) and
the `classifier` arm (the small route's own model).

With `split="domain"` the same items are routed to one route per domain instead, by the
classifier arms only; every domain route is the small model, so the run measures the decision,
not the answers.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from dataclasses import dataclass

from langchain.chat_models import init_chat_model
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel

from benchmark.dataset import Split
from benchmark.systemone import SystemOneStrategy
from langchain_model_router import ChatRouter
from langchain_model_router.strategies import (
    ClassifierStrategy,
    EmbeddingStrategy,
    HeuristicStrategy,
    KeywordStrategy,
)

GEMINI = os.environ.get("MODEL_ROUTER_BENCHMARK_GEMINI") == "1"
"""The opt-in for paid Gemini calls: the Gemini route pair, and the `embedding` and `classifier`
arms. Off by default."""
FRONTIER_MODEL = os.environ.get(
    "MODEL_ROUTER_BENCHMARK_FRONTIER_MODEL",
    "google_genai:gemini-3.8-flash" if GEMINI else "ollama:gpt-oss:20b",
)
SMALL_MODEL = os.environ.get(
    "MODEL_ROUTER_BENCHMARK_SMALL_MODEL",
    "google_genai:gemini-3.5-flash-lite" if GEMINI else "ollama:qwen3:8b",
)
EMBED_MODEL = os.environ.get("MODEL_ROUTER_GEMINI_EMBED_MODEL", "gemini-embedding-2-preview")
OLLAMA_SMOKE_MODEL = os.environ.get("MODEL_ROUTER_OLLAMA_MODEL", "ollama:qwen3:8b")
"""The `classifier-qwen3` arm's decision model, and `test_live_ollama_smoke.py`'s free, local,
budget-independent sanity check."""
ARCH_ROUTER_MODEL = os.environ.get(
    "MODEL_ROUTER_BENCHMARK_ARCH_ROUTER_MODEL", "ollama:hf.co/katanemo/Arch-Router-1.5B.gguf:Q4_K_M"
)
"""The `classifier-arch-router` arm's decision model: the Q4_K_M GGUF build (about 1 GB), pulled
with `ollama pull hf.co/katanemo/Arch-Router-1.5B.gguf:Q4_K_M`."""

# Uncalibrated pending a real run (embedding.py: EmbeddingStrategy's threshold has no shipped
# default by design). Override with
# MODEL_ROUTER_BENCHMARK_EMBEDDING_THRESHOLD once a live run's similarity scores say what a good
# bar actually is; see benchmark/README.md.
EMBEDDING_THRESHOLD = float(os.environ.get("MODEL_ROUTER_BENCHMARK_EMBEDDING_THRESHOLD", "0.5"))

# Calibration examples for EmbeddingStrategy and KeywordStrategy — deliberately *not* drawn from
# benchmark/data/workload.json: reusing graded items as the strategy's own few-shot examples
# would let a strategy "recognise" the exact things it is tested on, inflating its apparent
# accuracy over what it would do on traffic it hasn't seen.
_EASY_EXAMPLES = (
    "What is the capital of Japan?",
    "Convert 10 kilometers to miles.",
    "What does 'HTTP' stand for?",
    "What's 12 times 8?",
    "Is a tomato a fruit or a vegetable?",
)
_HARD_EXAMPLES = (
    "Compare two sorting algorithms' time complexity and explain when you'd choose one over "
    "the other.",
    "Here's a stack trace — explain the root cause and how to fix it.",
    "Derive the formula for compound interest and justify each step.",
    "Critique this argument for a logical flaw and suggest a better way to reason about it.",
    "Refactor this function and explain the trade-offs of your approach.",
)

_HARD_KEYWORDS = (
    "compare",
    "contrast",
    "prove",
    "derive",
    "justify",
    "critique",
    "refactor",
    "traceback",
    "root cause",
    "trade-off",
    "trade-offs",
)

# Opt-in arms: each runs only when its endpoint variable is set, because each needs its own local
# server (see benchmark/README.md). Value: the server's `/v1/systemone` URL.
_SYSTEMONE_ARMS = {
    "systemone-jevk5-4b": "MODEL_ROUTER_BENCHMARK_JEVK5_4B_URL",
    "systemone-jevk5-2b": "MODEL_ROUTER_BENCHMARK_JEVK5_2B_URL",
    "systemone-kev-0.8b": "MODEL_ROUTER_BENCHMARK_KEV_URL",
}

_TIER_DESCRIPTIONS = {
    "small": "simple, short requests: single facts, basic arithmetic, "
    "one-step lookups, short definitions",
    "frontier": "requests that need real reasoning: multi-step analysis, "
    "comparisons, proofs or derivations, debugging, refactors with "
    "trade-offs, or several distinct sub-questions at once",
}
_DOMAIN_DESCRIPTIONS = {
    "support": "customer support: orders, refunds, billing, accounts and passwords, "
    "and explaining a company's policies",
    "coder": "programming: writing, reviewing, debugging or refactoring code, "
    "stack traces, APIs and software design",
    "research": "research and general knowledge: explaining concepts, summarising, "
    "weighing evidence and arguments, weather and facts about the world",
    "math": "mathematics: arithmetic, algebra, probability, proofs and derivations, "
    "and word problems",
}


@dataclass(frozen=True)
class Arm:
    """One thing under test: a name for the report, and the router that implements it."""

    name: str
    router: ChatRouter


ArmFactory = Callable[[], Arm]


def frontier_model() -> BaseChatModel:
    return init_chat_model(FRONTIER_MODEL)


def small_model() -> BaseChatModel:
    # Thinking off on a local small model: it stays the quick tier and the frontier route is the
    # one that reasons, as in the Gemini pair, where only the frontier model reasoned.
    if SMALL_MODEL.startswith("ollama:"):
        return init_chat_model(SMALL_MODEL, reasoning=False)
    return init_chat_model(SMALL_MODEL)


def ollama_smoke_model() -> BaseChatModel:
    """The one place `OLLAMA_SMOKE_MODEL` is used — a free, local, real-provider sanity check
    that doesn't touch a budget at all. Not part of any arm."""
    return init_chat_model(OLLAMA_SMOKE_MODEL)


def qwen3_decider() -> BaseChatModel:
    # Thinking off: with it on, one decision took 14-26 s instead of about 0.6 s on a local run.
    return init_chat_model(OLLAMA_SMOKE_MODEL, reasoning=False)


def arch_router_decider() -> BaseChatModel:
    return init_chat_model(ARCH_ROUTER_MODEL)


def embeddings() -> Embeddings:
    from langchain_google_genai import GoogleGenerativeAIEmbeddings

    return GoogleGenerativeAIEmbeddings(model=EMBED_MODEL)


def _classifier_arms(
    routes: Callable[[], dict[str, BaseChatModel]], default: str, descriptions: dict[str, str]
) -> list[Arm]:
    """`ClassifierStrategy` on qwen3 and on Arch-Router (and on the small Gemini model, with the
    Gemini opt-in), then one `SystemOneStrategy` arm per decision-model endpoint that is
    configured."""
    deciders = {
        "classifier-qwen3": qwen3_decider,
        "classifier-arch-router": arch_router_decider,
    }
    if GEMINI:
        # "ask a small model" (classifier.py): the small route's own. Locally that is qwen3, the
        # same as `classifier-qwen3`, so the arm only exists with the Gemini pair.
        deciders = {"classifier": small_model, **deciders}
    return [
        Arm(
            name,
            ChatRouter(
                routes=routes(),
                default_route=default,
                strategy=ClassifierStrategy(decider(), descriptions),
            ),
        )
        for name, decider in deciders.items()
    ] + [
        Arm(
            name,
            ChatRouter(
                routes=routes(),
                default_route=default,
                strategy=SystemOneStrategy(os.environ[variable], descriptions),
            ),
        )
        for name, variable in _SYSTEMONE_ARMS.items()
        if os.environ.get(variable)
    ]


def build_arms(split: Split = "tier") -> list[Arm]:
    """The tier split's arms: the baseline plus keyword, heuristic and the classifiers (and
    embedding, with the Gemini opt-in) — or, for the domain split, the classifiers alone. A fresh
    instance of each model per arm, since a route must not be shared between two `ChatRouter`s
    that might run concurrently (nothing here forbids it, but each arm owning its own keeps the
    arms independent for a parallel run)."""
    if split == "domain":
        return _classifier_arms(
            lambda: {domain: small_model() for domain in _DOMAIN_DESCRIPTIONS},
            "support",
            _DOMAIN_DESCRIPTIONS,
        )
    return [
        Arm(
            "baseline-always-frontier",
            ChatRouter(routes={"frontier": frontier_model()}, default_route="frontier"),
        ),
        Arm(
            "keyword",
            ChatRouter(
                routes={"small": small_model(), "frontier": frontier_model()},
                default_route="small",
                strategy=KeywordStrategy({"frontier": list(_HARD_KEYWORDS)}),
            ),
        ),
        Arm(
            "heuristic",
            ChatRouter(
                routes={"small": small_model(), "frontier": frontier_model()},
                default_route="small",
                strategy=HeuristicStrategy("small", "frontier"),
            ),
        ),
        *(
            [
                Arm(
                    "embedding",
                    ChatRouter(
                        routes={"small": small_model(), "frontier": frontier_model()},
                        default_route="small",
                        strategy=EmbeddingStrategy(
                            embeddings(),
                            {"small": list(_EASY_EXAMPLES), "frontier": list(_HARD_EXAMPLES)},
                            threshold=EMBEDDING_THRESHOLD,
                        ),
                    ),
                )
            ]
            if GEMINI
            else []
        ),
        *_classifier_arms(
            lambda: {"small": small_model(), "frontier": frontier_model()},
            "small",
            _TIER_DESCRIPTIONS,
        ),
    ]
