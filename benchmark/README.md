# Cost/quality benchmark

Does routing actually save money without costing quality — and are the strategies that make their
own model or embedding calls worth their overhead, or do the free ones already get there?

The target: **at least 30% lower inference cost at at least 95% of always-frontier quality** on a
mixed workload. This harness runs five arms over one dataset against real models, prices every call
from real usage, has an LLM judge grade every answer, and reports each arm against an
always-frontier baseline. Rerun it any time — after changing the dataset, the models, or the
embedding threshold — with:

```bash
uv run python -m benchmark.run
```

## Results

One real run, 2026-09-23. Raw per-item data is in
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

## Local decision models

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

Five arms, over the same 32-item dataset (`data/workload.json`), plus the local-decision-model
arms above (`--split domain` routes the same items to one route per domain instead):

| Arm | What decides |
| --- | --- |
| `baseline-always-frontier` | nothing — every request goes to `gemini-3.8-flash` |
| `keyword` | `KeywordStrategy`: a short list of "this needs the frontier" keywords |
| `heuristic` | `HeuristicStrategy` on its shipped defaults (`DEFAULT_WEIGHTS`/`DEFAULT_THRESHOLD`) |
| `embedding` | `EmbeddingStrategy`, Gemini embeddings, an **uncalibrated** threshold (see below) |
| `classifier` | `ClassifierStrategy`, classifying on the small route's own model |

Two candidate routes throughout, both Gemini so the whole benchmark runs from one API key with no
local-server dependency: `"small"` (`gemini-3.5-flash-lite`) and `"frontier"` (`gemini-3.8-flash`)
— override with `MODEL_ROUTER_BENCHMARK_SMALL_MODEL` / `MODEL_ROUTER_BENCHMARK_FRONTIER_MODEL`. The
repository's other usual small/frontier pair (`ollama:qwen3:8b` / `gemini-3-flash-preview`) is what
`test_live_ollama_smoke.py` uses for its free, budget-independent sanity check — see "Testing this
harness itself". `arms.py` builds all five arms.

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
5. Grades the final answer against the item's `rubric` with an LLM judge (`judge.py`) — a model
   distinct from both routes, so grading isn't a route grading its own answer.

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

## The judge

`DEFAULT_JUDGE_MODEL` (`judge.py`) is `gemini-3.1-pro-preview` — a separate, stronger Gemini model
than either candidate route, so it never grades its own answer under the "frontier" name. It is
never added to any arm's `routes=`; nothing in `runner.py` ever routes to it. Override with
`MODEL_ROUTER_BENCHMARK_JUDGE_MODEL` if you'd rather point it at a different provider (e.g. an
Anthropic model, for a judge with no shared vendor with either route at all — this needs
`langchain-anthropic` and its own API key, neither of which this repo installs by default).

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
uv run python -m benchmark.run
```

Needs only `GEMINI_API_KEY` — every route, the classifier's own call, embeddings and the judge are
all Gemini. Every call is real and billed — `pricing.py`'s comments cite the source and date;
recheck it if it's old.

**Measured cost, not guessed:** a live call to each candidate model on one easy and one hard
dataset-style prompt gave $0.0005–0.010/call for `gemini-3.8-flash` and $0.00003–0.0025/call for
`gemini-3.5-flash-lite` (both models reason by default — `gemini-3.8-flash` even on a trivial
prompt — which is *why* the per-call range is so wide); the judge (`gemini-3.1-pro-preview`)
measured ~$0.0024/call. Even the worst case — every arm routing every item to the frontier model —
puts a full run (32 items × 5 arms, answers + judging) at roughly **$1.60–1.70**.

The `classifier-qwen3` and `classifier-arch-router` arms also need a local Ollama server with
`ollama pull qwen3:8b` and `ollama pull hf.co/katanemo/Arch-Router-1.5B.gguf:Q4_K_M` done first.

Useful flags:

- `--split domain` — route the same items to one route per domain instead of by difficulty.

- `--limit N` — cap the dataset to the first N items, for a cheap smoke test before the full run.
- `--arms baseline-always-frontier heuristic` — run a subset of arms.
- `--pause SECONDS` (default `1.0`) — courtesy delay between items, gentler on a rate-limited key.
- `--report-only path/to/raw.json` — re-render the report beside a saved run's raw results without
  calling anything again.

Each run writes its raw, per-item results to `results/<timestamp>.json` and the rendered report to
`results/<timestamp>.md` beside it — so a run's output is versioned and reproducible, not just its
input.

## Testing this harness itself

`benchmark/tests/` is entirely offline (fake routes, a fake judge, `DeterministicFakeEmbedding`)
except `test_live_ollama_smoke.py`, which makes real calls to a local Ollama server via
`arms.ollama_smoke_model()` — not the benchmark's own `small_model()`/`frontier_model()`
(`@pytest.mark.requires_ollama`, skips without one) — and deliberately never touches Gemini. Run
with the rest of the suite (`uv run pytest`) or on their own:

```bash
uv run pytest benchmark/tests
```
