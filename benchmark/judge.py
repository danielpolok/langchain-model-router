"""Quality scoring against each item's rubric: a judge evaluates; it never routes.

By default there is no judge model: `run.py` records answers ungraded, exports them with the
grading prompt below (`--export-ungraded`), a grader outside the run (Claude, in the recorded
results) writes back one `JudgeVerdict` per answer, and `--import-grades` merges them in. Set
`MODEL_ROUTER_BENCHMARK_JUDGE_MODEL` (for example `google_genai:gemini-3.1-pro-preview`, a paid
model) to grade inside the run instead. Either way the judge is distinct from both candidate
routes, so grading isn't a route grading its own answer, and it is never added to a router's
`routes=`; `runner.py` only ever calls it through `grade`/`agrade`, never through `ChatRouter`.
"""

from __future__ import annotations

import os

from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel
from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel, Field

DEFAULT_JUDGE_MODEL = os.environ.get("MODEL_ROUTER_BENCHMARK_JUDGE_MODEL")
"""An in-run judge model, or `None` (the default) to grade offline."""


class JudgeVerdict(BaseModel):
    """A quality score against the item's rubric, out of 10, with a one-line rationale."""

    score: int = Field(
        ge=0, le=10, description="0 = fails the rubric entirely, 10 = fully meets it"
    )
    rationale: str = Field(description="one sentence: why this score, referencing the rubric")


_GRADING_PROMPT = """You are grading one AI assistant's answer against a rubric. Score strictly \
against the rubric only — do not reward style, length or confidence beyond what the rubric asks \
for, and do not penalise the answer for anything the rubric doesn't mention.

Request the assistant was given:
---
{prompt}
---

What a good answer must do (the rubric):
---
{rubric}
---

The assistant's answer:
---
{answer}
---

Score 0-10: 0 means the answer fails the rubric entirely (wrong, missing, or refuses); 10 means \
it fully satisfies every part of the rubric. Give one sentence of rationale that names which \
part of the rubric the score turns on."""


def judge_model(model: str | None = None) -> BaseChatModel | None:
    """The judge model, `DEFAULT_JUDGE_MODEL` unless overridden; `None` when neither names one."""
    model = model or DEFAULT_JUDGE_MODEL
    return init_chat_model(model, temperature=0) if model else None


def grading_prompt(prompt: str, rubric: str, answer: str) -> str:
    """The exact text a judge grades from, in-run or exported for offline grading."""
    return _GRADING_PROMPT.format(prompt=prompt, rubric=rubric, answer=answer or "(no answer)")


def grade(
    judge: BaseChatModel,
    *,
    prompt: str,
    rubric: str,
    answer: str,
    config: RunnableConfig | None = None,
) -> JudgeVerdict:
    """Score `answer` against `rubric` for the request `prompt`. Sync."""
    structured = judge.with_structured_output(JudgeVerdict)
    result = structured.invoke(grading_prompt(prompt, rubric, answer), config=config)
    assert isinstance(result, JudgeVerdict)
    return result


async def agrade(
    judge: BaseChatModel,
    *,
    prompt: str,
    rubric: str,
    answer: str,
    config: RunnableConfig | None = None,
) -> JudgeVerdict:
    """Score `answer` against `rubric` for the request `prompt`. Async."""
    structured = judge.with_structured_output(JudgeVerdict)
    result = await structured.ainvoke(grading_prompt(prompt, rubric, answer), config=config)
    assert isinstance(result, JudgeVerdict)
    return result
