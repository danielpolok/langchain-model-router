# Cost/quality benchmark

Does routing actually save money without costing quality — and are the strategies that make their
own model or embedding calls worth their overhead, or do the free ones already get there?

The target: **at least 30% lower inference cost at at least 95% of always-frontier quality** on a
mixed workload. This harness runs each arm over one dataset against real models, records every
answer and routing decision, has every answer graded against its rubric, and reports each arm
against an always-frontier baseline.

It runs with **no API key and no paid call**: the routes and decision models are local Ollama
models, and grading is a separate, offline step. Rerun it any time with:

```bash
ollama pull qwen3:8b gpt-oss:20b hf.co/katanemo/Arch-Router-1.5B.gguf:Q4_K_M  # once
uv run python -m benchmark.run                       # writes results/<stamp>-tier.json, ungraded
uv run python -m benchmark.run --export-ungraded results/<stamp>-tier.json
# grade results/<stamp>-tier-ungraded.json, writing results/<stamp>-tier-grades.json
uv run python -m benchmark.run --import-grades results/<stamp>-tier.json results/<stamp>-tier-grades.json
```

The last command merges the grades into the results file and writes the report beside it. See
[Grading](#grading) and [Cost without dollars](#cost-without-dollars).

## Results: local models, graded by Claude

One run of each split, 2026-10-22, with no paid model anywhere:

- **Routes:** `ollama:qwen3:8b`, thinking off (`small`), and `ollama:gpt-oss:20b` (`frontier`).
- **Decision models:** `qwen3:8b` (thinking off) and Arch-Router-1.5B through Ollama, and Kev 0.8B
  on a local System One server (`kev-0.8b-q8_0.gguf`).
- **Grader:** Claude (Claude Opus 5.5), offline, blind to the arm, every answer graded; see
  [Grading](#grading).
- **Machine:** an Apple M4 with 24 GB, which cannot hold `gpt-oss:20b` and `qwen3:8b` in memory
  together, so Ollama reloads models between calls.

Raw data, grades and reports:
[`results/20261022T191458Z-tier.json`](results/20261022T191458Z-tier.json) (report
[`.md`](results/20261022T191458Z-tier.md), grades
[`-grades.json`](results/20261022T191458Z-tier-grades.json)) and
[`results/20261022T201918Z-domain.json`](results/20261022T201918Z-domain.json) (report
[`.md`](results/20261022T201918Z-domain.md), grades
[`-grades.json`](results/20261022T201918Z-domain-grades.json)).

**Tier split** (route `small` / `frontier` by the dataset's easy/hard label): 6 arms × 32 items,
**0 errors**, 192 answers graded.

| Arm | Routing accuracy | Frontier calls avoided | Mean quality (/10) | Quality vs baseline | Decision p50 / p95 | Tool accuracy |
| --- | --- | --- | --- | --- | --- | --- |
| `baseline-always-frontier` | — | — | 8.97 | — | — | 69% |
| `keyword` | 22/32 (69%) | 81.2% | 8.00 | 89.2% | 0.000 s / 0.002 s | 88% |
| `heuristic` | 20/32 (62%) | 87.5% | 8.31 | 92.7% | 0.000 s / 0.001 s | 94% |
| `classifier-qwen3` | 29/32 (91%) | 40.6% | 7.94 | 88.5% | 5.01 s / 5.67 s | 81% |
| `classifier-arch-router` | 21/32 (66%) | 84.4% | 7.81 | 87.1% | 0.37 s / 1.70 s | 88% |
| `systemone-kev-0.8b` | 19/32 (59%) | 59.4% | 7.72 | 86.1% | 0.27 s / 0.38 s | 94% |

**No arm meets the target on this pair** (at least 95% of baseline quality with at least 30% of
frontier calls avoided). Every strategy avoids enough frontier calls; all of them lose 7-14% of
the baseline's quality. How to read it:

- **The gap between the routes is real here, unlike the Gemini pair.** Across all arms, `qwen3:8b`
  scored 8.85 on easy items and 6.61 on hard ones; `gpt-oss:20b` scored 8.58 and 8.11. The small
  model is as good on easy items and clearly worse on hard ones, which is the trade-off routing is
  meant to manage, so every hard item sent to `small` costs quality.
- **Accuracy alone doesn't buy quality back.** `classifier-qwen3` routes most accurately (29/32)
  and still scores lowest-but-two, because `gpt-oss:20b` sometimes returns an empty final answer
  after a tool call: 8 of its 79 frontier answers across the arms were empty, 4 of them in this
  arm, and an empty answer scores 0. The baseline had one. That is a model failure, not a routing
  one, and with one run per arm it is enough to reorder arms this close.
- **The cheap strategies hold up.** `heuristic` keeps the most quality (92.7%) while avoiding the
  most frontier calls (87.5%), with no decision model at all, as in the Gemini run.
- **Decision latency includes model reloads.** `classifier-qwen3`'s 5 s is mostly Ollama swapping
  `qwen3:8b` back in after `gpt-oss:20b`; the warm figure from the earlier run was 0.75 s.
  Arch-Router's p95 has the same cause. Kev runs on its own server, so it never waits for Ollama.
- **Tool accuracy is measured from the trace, not graded.** The exported answer carries no record
  of the tool call, so the grades judge the reported result, and the tool column says whether the
  expected tool was called.
- **Why the always-frontier baseline has the lowest tool accuracy (69%).** All five of its misses
  are arithmetic items where `calculator` was expected (`coder-hard-agent-1/2`,
  `math-easy-agent-2`, `math-hard-agent-1/2`), and in all five `gpt-oss:20b` called no tool at
  all: it reasons and does the arithmetic itself. It called `lookup_order_status`, `search_docs`
  and `lint_code` every time. So it is not weak at calling tools, it skips the calculator when it
  can work the sum out, and the cheap arms score higher because they send most of those items to
  `qwen3:8b`, which called the calculator on 21 of the 25 arithmetic items it answered. The
  baseline scored 10/10 on all five of those answers; its real weakness is the empty final answer
  described above.
- **Why `keyword` has 26 fallbacks.** `KeywordStrategy` here only lists words that send a request
  to `frontier` ("compare", "derive", "refactor"...), and abstains when none appears, so the
  default route `small` answers. 26 of 32 items had no such word: all 16 easy items (correctly
  `small`) and 10 hard ones (missed). The 6 it decided were all hard and all right. A fallback is
  this strategy's normal "small" answer, not a failure.

**Domain split** (one route per domain: `support`, `coder`, `research`, `math`; every route is
`qwen3:8b`, so only the decision differs): 3 arms × 32 items, **0 errors**, 96 answers graded.

| Arm | Routing accuracy | Mean quality (/10) | Decision p50 / p95 | Tool accuracy |
| --- | --- | --- | --- | --- |
| `classifier-qwen3` | 27/32 (84%) | 8.31 | 0.99 s / 1.65 s | 94% |
| `classifier-arch-router` | 19/32 (59%) | 7.97 | 0.27 s / 0.91 s | 88% |
| `systemone-kev-0.8b` | 23/32 (72%) | 8.09 | 0.25 s / 0.43 s | 88% |

- Routing accuracy matches the earlier runs for `classifier-qwen3` (27/32) and Kev (23/32);
  Arch-Router is 19/32 here against 22/32 before, on the same generic prompt.
- The quality column is a sanity check, not a comparison: the same model answers whichever
  domain route is picked, so the 0.3-point spread is sampling noise between runs of one model.
- Latency is lower than in the tier run because only one answering model is loaded, so
  `qwen3:8b`'s decisions stay warm (about 1 s).

**Verdict for both splits, local and graded by Claude.** The free `heuristic` strategy is still
the best trade-off: on the tier split it avoids the most frontier calls and keeps the most
quality, and no decision model beats it. `qwen3:8b` remains the most accurate local decision
model on both splits, but on the tier split that accuracy did not turn into quality, and it is
the slowest when the routes compete for memory. Kev and Arch-Router are fast but no more accurate
than the heuristic on tiers. None of the arms met the target against `gpt-oss:20b`, which says
the local small model gives up more on hard items than the old Gemini small model did.

## Earlier results: Gemini routes and judge (2026-09-23)

Every number in this section came from paid Gemini models: the routes, the classifier's decision
model, the embeddings and the judge. One real run, 2026-09-23. Raw per-item data is in
[`results/20260923T085017Z.json`](results/20260923T085017Z.json) and the rendered report in
[`results/20260923T085017Z.md`](results/20260923T085017Z.md).

- **Models:** `gemini-3.8-flash` ("frontier"), `gemini-3.5-flash-lite` ("small"), and
  `gemini-3.1-pro-preview` as the judge — never a route.
- **Cost of the run:** $0.3568 for 5 arms × 32 items × (answer + judge). Zero errors.

| Arm | Cost saved vs baseline | Quality vs baseline | Meets target |
| --- | --- | --- | --- |
| `keyword` | 41.8% | 106.6% | **yes** |
| `heuristic` | 50.1% | 110.4% | **yes** |
| `embedding` | 14.7% | 97.2% | no (cost) |
| `classifier` | 19.0% | 108.5% | no (cost) |

**The free strategies clear the target; the opt-in ones don't, on this run.** `keyword` and
`heuristic` make no extra call at all and beat both opt-in strategies on net cost — even though
`classifier` routes far more *accurately*:

| Arm | Routing accuracy vs the dataset's own easy/hard label | Cost saved |
| --- | --- | --- |
| `classifier` | 31/32 correct | 19.0% |
| `heuristic` | 20/32 correct (all 16 easy right; 12 of 16 hard under-escalated) | 50.1% |

The classifier almost perfectly recovers the intended difficulty split and still saves *less*,
because its own call is a real, billed request (here the same model as the "small" route) and that
overhead eats into the saving faster than the extra accuracy earns it back. The embedding strategy
shows the same shape for a different reason: its threshold is an uncalibrated placeholder (see
below), and it sends two whole domains (`support`, `research`) to the frontier route because their
calibration examples read closer to the "hard" set than the "easy" one. Its 14.7% is mostly a
calibration artefact, not evidence it can't do better.

**Conclusion:** the embedding and classifier strategies are useful but **optional** — reach for
them once the free strategies measurably fall short on *your* traffic, not by default.

**Not an argument for retuning the heuristic's defaults.** It sent 12 of 16 "hard" items to the
small model and still scored *higher* mean quality than the always-frontier baseline (7.31 vs
6.62), so tightening its thresholds would only add cost on this workload. The shipped defaults are
unchanged.

### Surprises

1. **The frontier model reasons even on trivial prompts, and isn't obviously better for it.** A
   direct call with "What is 17 + 26?" cost 122 output tokens on `gemini-3.8-flash` (111 of them
   reasoning) against 10 on `gemini-3.5-flash-lite`, and the heuristic arm's mean quality beat the
   baseline's while routing most traffic to the cheaper model. Either the rubric doesn't reward
   the extra reasoning, or the small model is simply strong for a workload this size.
2. **Routing accuracy and net cost saving are different things.** The classifier is the most
   accurate router and second-worst on cost, purely from paying for its own decision.
3. **One global embedding threshold doesn't travel across domains.** The same calibration
   examples separated `math`/`coder` requests cleanly but not `support`/`research` ones.

### Limitations

- **One run, no repeats.** Routes weren't called at `temperature=0`, so quality figures this close
  (97–110% of baseline) could shift on a rerun. The judge was pinned at `temperature=0`.
- **32 items show a shape, not a tight bound.** Treat the exact percentages as directional.
- **`gemini-3.5-flash-lite` may be an unusually strong "small" model** for this difficulty ceiling,
  which would flatter every strategy at once. A harder dataset, or a bigger capability gap between
  the routes, is what would stress-test this conclusion rather than confirm it.
- **The embedding threshold and example set are placeholders**, so this run measures *that
  configuration*, not what the strategy could do calibrated.
- **Provider prompt-caching loss from switching routes is not measured here.**

## Earlier results: local decision models on Gemini routes (2026-10-19 to 2026-10-21)

Only the decision models in this section were local. In the tier table, the answering routes and
the judge were the Gemini models above, so its cost and quality columns are Gemini numbers; the
Kev rows and the local domain rerun used `gemma4` routes and an in-run `gemma4` judge whose scores
are not used.

Can a small local model make the classifier's routing decision for free? Same 32 items, run on
2026-10-19 with `ollama:qwen3:8b` (thinking off; with it on one decision took 14-26 s instead of
about 0.6 s) and Arch-Router-1.5B (`hf.co/katanemo/Arch-Router-1.5B.gguf:Q4_K_M`, 986 MB), both
through the unchanged `ClassifierStrategy` and its generic prompt. Raw data:
[`results/20261019T162230Z-tier.json`](results/20261019T162230Z-tier.json),
[`results/20261019T163717Z-tier.json`](results/20261019T163717Z-tier.json),
[`results/20261019T183746Z-domain.json`](results/20261019T183746Z-domain.json). Local models are
priced at $0, which leaves out the hardware and power that run them.

**Tier split** (route `small` / `frontier` by the dataset's easy/hard label). The baseline is the
always-frontier arm of the same run; the two tier runs have different baselines, so quality is
relative to each.

| Arm | Routing accuracy | Cost saved | Quality vs baseline | Decision p50 / p95 |
| --- | --- | --- | --- | --- |
| `heuristic` | 20/32 (62%) | 51.9% | 98.6% | 0.000 s / 0.001 s |
| `classifier` (Gemini flash-lite) | 31/32 (97%) | 18.9% | 100.0% | 0.92 s / 1.68 s |
| `classifier-qwen3` | 28/32 (88%) | 14.4% | 96.4% | 0.75 s / 0.92 s |
| `classifier-arch-router` | 20/32 (62%) | 53.1% | 105.8% | 0.23 s / 0.30 s |
| `systemone-kev-0.8b` | 19/32 (59%) | not measured | not measured | 0.21 s / 0.31 s |

**Domain split** (one route per domain: `support`, `coder`, `research`, `math`; the routes are all
the same small model, so only the decision is compared).

| Arm | Routing accuracy | Decision p50 / p95 |
| --- | --- | --- |
| `classifier` (Gemini flash-lite) | 28/32 (88%) | 0.89 s / 1.41 s |
| `classifier-qwen3` | 27/32 (84%) | 5.50 s / 5.97 s |
| `classifier-arch-router` | 22/32 (69%) | 0.35 s / 0.48 s |
| `systemone-kev-0.8b` | 23/32 (72%) | 0.32 s / 0.42 s |

How to read the numbers:

- The Gemini classifier's domain row is from a run that was cut short when the Gemini credits ran
  out; its 32 decisions finished, but its raw file was not kept, so this row is from that run's
  printed report. The two local arms were rerun with the answering routes and the judge on a local
  model (`gemma4`) and the answers' quality scores are not used here.
- That rerun makes the local domain latencies unreliable: Ollama reloads each model between the
  decision and the answer, which is most of qwen3's 5.5 s. The warm latencies in the tier table are
  the fair ones.
- Quality differences between the tier runs (for example 105.8% for Arch-Router) are within the
  noise this benchmark already warns about; the baseline's own mean moved from 6.91 to 6.47
  between the two runs.
- Arch-Router sends exactly what the heuristic sends on the tier split (all 16 easy right, 12 of
  16 hard to `small`), which is expected: it was trained on domain-style routes, not difficulty
  tiers. On the domain split it routes 22/32 correctly, mostly `research` to `support` (4) and
  `coder` to `research` or `support` (4).

**Verdict.**

- *Does a local decision model keep the classifier's accuracy?* Not on this data. `qwen3:8b` is
  the closest (88% vs 97% on tiers, 84% vs 88% on domains); Arch-Router on the generic classifier
  prompt is well behind (62% and 69%).
- *Does it recover the saving the paid call cost?* No, on tiers. qwen3's decision is free, but its
  four easy items sent to the frontier model cost more than the Gemini call it replaced: 14.4%
  saved against the Gemini classifier's 18.9%, and both far below the heuristic's 51.9%. Arch-Router
  saves 53.1% only because it behaves like the heuristic on tiers.
- *Latency:* Arch-Router decides in about 0.23 s on this machine, well ahead of any other model
  arm; `qwen3:8b` is no faster than the Gemini call (0.75 s vs 0.92 s).
- *Follow-up:* worth filing one, narrow: the prompt and parse hooks on `ClassifierStrategy`, to see
  whether Arch-Router's 69% is its own prompt format rather than the model (its published
  accuracy uses that format, and its `"other"` answer should abstain). Not worth a new strategy or
  anything that needs `transformers`, until that test says so.

### System One decision models

Models in the Jev family score a fixed list of options and return a probability for each in one
forward pass, behind a `POST /v1/systemone` endpoint. `systemone.py` is the benchmark-only
adapter (`SystemOneStrategy`, standard library only) and `arms.py` adds one opt-in arm per endpoint
whose variable is set. Only Kev has been run; JevK5 has not.

| Arm | Set to the server's `/v1/systemone` URL | Model |
| --- | --- | --- |
| `systemone-jevk5-4b` | `MODEL_ROUTER_BENCHMARK_JEVK5_4B_URL` | JevK5 4B (Apache-2.0), not run |
| `systemone-jevk5-2b` | `MODEL_ROUTER_BENCHMARK_JEVK5_2B_URL` | JevK5 2B (Apache-2.0), not run |
| `systemone-kev-0.8b` | `MODEL_ROUTER_BENCHMARK_KEV_URL` | Kev 0.8B (Apache-2.0 on its model card) |

**Kev 0.8B, run 2026-10-21** on the Kev-enabled `llama-server` (`kev-0.8b-q8_0.gguf`, 127.0.0.1,
one slot) with the unchanged adapter. Raw data:
[`results/20261021T053940Z-tier.json`](results/20261021T053940Z-tier.json) and
[`results/20261021T062655Z-domain.json`](results/20261021T062655Z-domain.json). Both runs: 32 items,
0 errors, 0 fallbacks. The answering routes and the judge were `gemma4` through Ollama, so there is
no always-frontier baseline in these runs and no cost or quality figure; the rows above compare
routing accuracy and decision latency only. The latencies are wall-clock time for the HTTP call
to the local server on a busy laptop, so read them as an order of magnitude.

- *Tier split:* 19/32 (59%). It sends 8 of 16 hard items to `small` and 5 of 16 easy items to
  `frontier`, so it is neither the heuristic's "every easy right, most hard wrong" nor a better
  version of it: its errors are on both sides.
- *Domain split:* 23/32 (72%). `math` is 8/8 and `coder` 7/8, but `research` is 2/8 (4 to `coder`,
  1 to `math`, 1 to `support`) and 2 of 8 `support` items go to `coder`.

**Verdict.** Kev is fast (about 0.2-0.3 s, in the Arch-Router range and ahead of every classifier
that generates text) and never errored or abstained, but it is not accurate enough here: 59% on
tiers is no better than the free heuristic (62%) and 72% on domains is slightly ahead of
Arch-Router (69%) and behind `qwen3:8b` (84%) and the Gemini classifier (88%). One run, 32 items,
a 0.8B model on route descriptions that were written for the text classifiers and not tuned for it,
so this is a first reading, not a bound. It does not change the recommendation above: no new
strategy is justified by this result.

**Earlier failed Kev runs (not in git).** Two first attempts on 2026-10-19 (`20261019T193327Z-tier`
and `20261019T193329Z-domain`) recorded 32 errors out of 32 and no routing accuracy. The Kev
endpoint and the adapter were fine; the errors were in the local answering route (`No data
received from Ollama stream`), and a rerun without a price entry for `gemma4` fails every item
with `KeyError: no price for model 'gemma4'`. The working runs used
`MODEL_ROUTER_BENCHMARK_PRICE_OVERRIDES` with a $0 price for `gemma4` and `gemma4:latest`. The two
failed files were left untracked and are not part of the results. The 2026-10-21 runs took hours of
wall-clock time because the Mac slept mid-run.

## What it does

Every arm runs over the same 32-item dataset (`data/workload.json`); `--split domain` routes the
same items to one route per domain instead, with the classifier arms only.

| Arm | What decides | Runs |
| --- | --- | --- |
| `baseline-always-frontier` | nothing — every request goes to the frontier route | always |
| `keyword` | `KeywordStrategy`: a short list of "this needs the frontier" keywords | always |
| `heuristic` | `HeuristicStrategy` on its shipped defaults (`DEFAULT_WEIGHTS`/`DEFAULT_THRESHOLD`) | always |
| `classifier-qwen3` | `ClassifierStrategy` on `ollama:qwen3:8b`, thinking off | always |
| `classifier-arch-router` | `ClassifierStrategy` on Arch-Router-1.5B through Ollama | always |
| `systemone-kev-0.8b` (and the JevK5 arms) | `SystemOneStrategy` on a local System One server | when its URL is set |
| `embedding` | `EmbeddingStrategy`, Gemini embeddings, an **uncalibrated** threshold | Gemini opt-in |
| `classifier` | `ClassifierStrategy` on the small route's own model | Gemini opt-in |

**The routes are local by default:** `"small"` is `ollama:qwen3:8b` with thinking off and
`"frontier"` is `ollama:gpt-oss:20b`. Why this pair, of the models installed here (`qwen3:8b`,
`gemma4`, `gpt-oss:20b`, Arch-Router):

- `gpt-oss:20b` is the largest (21B parameters, a mixture of experts with about 3.6B active per
  token, so it still answers at a usable speed on a laptop) and always reasons, like the Gemini
  frontier model did. `gemma4` is an 8B model, the same size class as `qwen3:8b`, so it would not
  give a real capability gap.
- `qwen3:8b` with thinking off is the quick, cheap tier, and it is already the `classifier-qwen3`
  decision model, matching the original design of a classifier on the small route's own model.

Override with `MODEL_ROUTER_BENCHMARK_SMALL_MODEL` / `MODEL_ROUTER_BENCHMARK_FRONTIER_MODEL`.

**Gemini is an explicit opt-in:** `MODEL_ROUTER_BENCHMARK_GEMINI=1` switches the defaults back to
the paid pair (`gemini-3.5-flash-lite` small, `gemini-3.8-flash` frontier) and adds the
`embedding` and `classifier` arms. Without it, nothing in a run calls Gemini. `arms.py` builds
every arm.

For each (arm, item) pair, `runner.run_item`:

1. Invokes the router — a two-step tool round trip for `kind == "agent"` items (bind the full
   toolkit in `tools.py`, execute at most the first tool call, invoke again with its result).
2. Captures usage with `langchain_core.callbacks.get_usage_metadata_callback` — which the router's
   own tests prove is exactly the routes' own usage, including a classifier strategy's nested call
   (it inherits the router's config, so the same callback sees it without special-casing).
3. Reads the routing decision straight off the response (`routing_decision`) — which route ran,
   why, and whether the strategy decided or the default route fell back.
4. Prices it (`costing.py`, `pricing.py`) — real usage × pinned $/token, plus an *estimate* for
   `EmbeddingStrategy`'s per-request query: `Embeddings` reports no usage at all, so there is
   nothing to measure, and the strategy's own chars-per-token method stands in.
5. Leaves the answer ungraded for the offline step (see [Grading](#grading)), or grades it with an
   in-run judge model if one is configured.

`report.py` aggregates every arm's `ItemResult`s into cost-saved-vs-baseline and
quality-vs-baseline percentages, routing accuracy against the dataset's expected route, the median
and p95 time the strategy took to decide, and a verdict against the target.

## The dataset

`data/workload.json`, generated by `_build_dataset.py` (edit the `ITEMS` list there, then rerun it
— the JSON is the versioned artefact, not something to hand-edit). 32 items: 4 domains (`support`,
`coder`, `research`, `math`) × 2 difficulties (`easy`, `hard`) × 2 kinds (`single_turn`, `agent`),
2 items per cell. Every item carries `provenance`: all of it is author-written for this benchmark
(no verbatim reuse of any copyrighted problem set), patterned after common question styles where
noted. `dataset.validate()` checks the shape (enough domains, both difficulties, both kinds, every
agent item names a real tool) every time it loads.

## Grading

By default a run has no judge model: every result's `judge` is empty. Grading is a separate step,
so it needs no paid model:

1. `--export-ungraded RAW` writes `<RAW stem>-ungraded.json` beside the results: one entry per
   answered, ungraded result with its arm, item id, prompt, rubric, answer, and `grading_prompt`,
   the exact text an in-run judge would see (`judge.grading_prompt`).
2. A grader reads each `grading_prompt` and writes a grades file: a JSON list of
   `{"arm", "item_id", "score", "rationale"}`, where `score` (0-10) and `rationale` follow the
   `JudgeVerdict` schema.
3. `--import-grades RAW GRADES` checks every grade (a known answered result, a valid
   `JudgeVerdict`) before changing anything, writes them into the results file, says how many
   answers are still ungraded, and renders the report.

**The recorded grades come from Claude** (Claude Opus 5.5, `claude-opus-5-5`), working in the
Claude Code session that ran the benchmark: it read every exported answer and scored it strictly
against its rubric with the grading prompt above. That is not a separate paid API call from the
benchmark; nothing in the harness calls Claude. The answers were shown blind (shuffled within each
item and labelled by export index, not by arm), and the grades files are kept beside the results
(`results/<stamp>-<split>-grades.json`).

An in-run judge is still available: set `MODEL_ROUTER_BENCHMARK_JUDGE_MODEL` (for example
`google_genai:gemini-3.1-pro-preview`, the paid judge of the earlier Gemini runs). It is never
added to any arm's `routes=`; nothing in `runner.py` ever routes to it.

## Cost without dollars

Local models cost $0 per token (`pricing.py` lists them; the hardware and power that run them are
not priced), so a run on local routes has no dollar cost to save. When the baseline cost $0,
`report.py` reports **frontier calls avoided** instead of cost saved: the share of items the arm
answered on the `small` route, against a baseline that answers every item on `frontier`, and the
target check uses that figure. It equals the dollar saving only if a small call costs nothing next
to a frontier one, and it does not count a decision model's own calls; the decision latency
column shows what those cost in time. With the Gemini opt-in, cost is real dollars as before.

## What's uncalibrated, on purpose

- **`EmbeddingStrategy`'s threshold** (`EMBEDDING_THRESHOLD` in `arms.py`, default `0.5`) —
  `embedding.py` ships no default at all, because a number that's silently wrong is worse than one
  the caller has to supply, and a real threshold needs real similarity scores from a live run to
  calibrate against. Override with `MODEL_ROUTER_BENCHMARK_EMBEDDING_THRESHOLD`.
- **`HeuristicStrategy`'s defaults** (`DEFAULT_WEIGHTS`, `DEFAULT_THRESHOLD`,
  `DEFAULT_LENGTH_RANGE` in `src/langchain_model_router/strategies/heuristic.py`) are used as
  shipped (all weights equal, threshold `1.0`). If a future run shows a clear miss — say the
  heuristic arm's routing accuracy is poor on one signal — edit those constants and rerun to check
  the fix, following that module's own notes on what a benchmark result should retune.

## Running it for real

```bash
uv run python -m benchmark.run                  # tier split
uv run python -m benchmark.run --split domain   # domain split
```

Needs no API key. It needs a local Ollama server with `qwen3:8b`, `gpt-oss:20b` and
`hf.co/katanemo/Arch-Router-1.5B.gguf:Q4_K_M` pulled, and, for the Kev arm,
`MODEL_ROUTER_BENCHMARK_KEV_URL` set to a running System One server. A full local run is slow:
the 2026-10-22 runs took about 2 hours (tier) and 1 hour (domain) on an Apple M4 with 24 GB, mostly answer
generation and Ollama reloading models that don't fit in memory together. On a Mac, keep it awake
for the whole run (`caffeinate -dimsu uv run python -m benchmark.run`): an earlier run lost hours
when the machine slept.

With `MODEL_ROUTER_BENCHMARK_GEMINI=1` every Gemini call is real and billed (`GEMINI_API_KEY`);
`pricing.py`'s comments cite the source and date. A live call on the earlier Gemini pair measured
$0.0005–0.010/call for `gemini-3.8-flash` and $0.00003–0.0025/call for `gemini-3.5-flash-lite`,
and about $0.0024/call for the `gemini-3.1-pro-preview` judge: roughly $1.60–1.70 for a full run in
the worst case.

Useful flags:

- `--split domain` — route the same items to one route per domain instead of by difficulty.
- `--limit N` — cap the dataset to the first N items, for a quick smoke test before the full run.
- `--arms baseline-always-frontier heuristic` — run a subset of arms.
- `--pause SECONDS` (default `0`) — courtesy delay between items, for a rate-limited key.
- `--export-ungraded RAW` / `--import-grades RAW GRADES` — the offline grading step above.
- `--report-only path/to/raw.json` — re-render the report beside a saved run's raw results without
  calling anything again.

Each run writes its raw, per-item results to `results/<timestamp>-<split>.json` and the rendered
report to `results/<timestamp>-<split>.md` beside it — so a run's output is versioned and
reproducible, not just its input.

## Testing this harness itself

`benchmark/tests/` is entirely offline (fake routes, a fake judge, `DeterministicFakeEmbedding`)
except `test_live_ollama_smoke.py`, which makes real calls to a local Ollama server via
`arms.ollama_smoke_model()` (`@pytest.mark.requires_ollama`, skips without one) and never touches
Gemini. `test_grading.py` covers the no-judge run, the export and the import. Run them on their
own:

```bash
uv run pytest benchmark/tests
```
