# CLAUDE.md: Nexora Forecaster

Claude Code loads this file automatically at the start of every session. It is a handoff document from the Claude session that built the project on 2026-09-26.

This repo is meant to be **public**. Never put secrets or personal information in any file. Keys live only in `.env` (gitignored) and in GitHub Actions secrets.

## What this project is

An automated forecasting bot for the **Metaculus FutureEval AI bot tournaments**: the Fall 2026 seasonal tournament plus MiniBench.

- **What it does:** GitHub Actions runs it every 20 minutes. It forecasts each new question exactly once and posts a private comment plus the forecast.
- **How it earns:** prize money if it scores well. Nothing is traded and no capital is at risk. The only cost is LLM usage through OpenRouter, which Metaculus partly sponsors.
- **Owner's goal:** win prize money this season while keeping LLM spend low relative to expected winnings. Longer term, the forecasting engine may be reused in the owner's separate trading project, which is not part of this repo.

Flow per question:

```
research (parallel, all optional and time-boxed):
  statistical model for MiniBench data questions (FRED value, stock direction, community-prediction move)
  + AskNews + Perplexity web search + read-only Polymarket/Manifold prices + similar RESOLVED Metaculus questions
-> ensemble of 3-5 LLM forecasters across model families (Claude-led)
-> aggregation (binary: trimmed log-odds mean, quant blend, recalibration, 2-98% cap;
               MC: mean + 1% floor; numeric/date: mean of CDFs + quant blend)
-> post PRIVATE comment first, then the forecast
```

## Status at handoff (2026-09-26)

- Code is complete. **73 offline tests pass** (`uv run pytest -q`), and a fresh install from `uv.lock` was verified.
- **2026-10-08 update (owner's Mac):**
  - Verified live: all model ids resolve; FRED CSV works; Yahoo answers 429, so stock prices fall back to Nasdaq; the Metaculus token works and tournament ids 33121 / `minibench` are right; `search=` is honoured; the real discrete MiniBench questions run through the pipeline.
  - Found and handled: Fall 2026 MiniBench is news-driven (see Tournament facts), so `lean` now includes web search.
  - Sandbox verified from the bot account: every model in `lean` and `standard` answered (medium and high effort, Gemini included), and research used Perplexity, AskNews and markets. Discrete, binary and MC forecasts were recorded on Metaculus, and the comments are `is_private: true`.
  - The first sandbox run used a personal-account token, and Metaculus refused every forecast (403 `api_forecasting_not_enabled`). `nexora/account.py` now stops `main.py` before anything is posted if the token is not a bot account, and `_publish_report` stops publishing for the rest of a run on that 403.
  - Published to GitHub (`haotian21/nexora-forecaster`) with the schedule paused (`NEXORA_PAUSED=true`).
- **Nothing has run against live services yet.** The build sandbox could not reach Metaculus, OpenRouter, AskNews, FRED, Yahoo or the market APIs, so the first job on a real machine is live verification (next steps 3-4).
- There is **no git history**: the project arrived as a zip. Start with `git init -b main`.
- **Timing:**
  - Fall 2026 seasonal questions start opening **Mon 2026-09-28**, slowly for the first 1-2 weeks. The season runs until 2027-01-06.
  - MiniBench rounds start every other Monday.
  - Being live early matters, but a few missed days early on are cheap.

## Next steps (in order)

1. **Environment (macOS):**
   ```bash
   brew install uv gh    # or: curl -LsSf https://astral.sh/uv/install.sh | sh
   uv sync               # fetches Python 3.11 if needed
   uv run pytest -q      # expect 73 passed
   ```
2. **Accounts.** The owner creates these himself. **Never ask him to paste secrets into the chat**; he writes them into `.env` directly.
   - Metaculus bot account and token: https://www.metaculus.com/futureeval/participate/ (one bot per person).
   - Participant form, which also requests free LLM credits (they arrive as an OpenRouter key): https://forms.gle/aQdYMq9Pisrf1v7d8. Until then, use his own key from https://openrouter.ai/keys. **Set a credit limit on the key**; it is the hard budget cap.
   - AskNews (free for bot makers, renew every season), process per the resources page as of 2026-10-08:
     1. Create an account at https://my.asknews.app with the **same email as the Metaculus bot account**.
     2. Message @freqai on the AskNews Discord, or email contact@asknews.app, with: bot name, AskNews email, first and last name, LinkedIn, and affiliation.
     3. Once activated, generate `ASKNEWS_API_KEY` at https://my.asknews.app/en/settings/api-credentials.
3. **Live checks.** Run `cp .env.template .env`; the owner fills it in. Then:
   ```bash
   uv run python scripts/check_setup.py     # token, key, resolved models per profile, 1 tiny LLM call, FRED/Yahoo/markets
   uv run python scripts/smoke_quant.py     # live FRED / Yahoo / community-prediction models, no LLM cost
   uv run python main.py --mode test --dry-run --limit 3 --profile lean   # sandbox only; read the outputs
   uv run python main.py --mode test --limit 3 --profile lean             # posts to the sandbox tournament
   ```
   Then confirm on the bot's Metaculus profile that the forecasts **and private comments** appeared.
4. **Fix what the live checks reveal.** The most likely problems are model ids and endpoint details; see "Unverified assumptions".
5. **Publish to GitHub.** Use a public repo, because Actions minutes are then free; a private free plan has only 2,000 min/month. **Set the secrets before pushing**, because the 20-minute schedule starts as soon as the workflow is on `main`.
   ```bash
   git init -b main && git add -A && git commit -m "Nexora forecaster"
   gh auth login                       # owner runs this; also let it set up git credentials
   gh repo create nexora-forecaster --public
   gh secret set -f .env -R <owner>/nexora-forecaster   # every uncommented KEY=value line in .env becomes a secret
   git remote add origin https://github.com/<owner>/nexora-forecaster.git && git push -u origin main
   gh workflow run test_bot.yaml -f profile=lean -f limit=3
   ```
   `gh secret set -f` keeps key values out of the chat. Don't run `gh secret set NAME` without `--body` or `-f` from a non-interactive shell: it waits for input. Re-run `gh secret set -f .env ...` whenever `.env` gains keys, e.g. AskNews later.
   To hold the schedule: `gh variable set NEXORA_PAUSED --body true -R <owner>/nexora-forecaster`.
6. **From 2026-09-28, watch the first scheduled runs.** Use `gh run list` and `gh run view <id> --log`. Each run prints forecasts, failures, deferred questions and estimated cost. Also check the OpenRouter usage page.
7. **Later:**
   - Once at least 60 binary questions have resolved, run `uv run python scripts/fit_calibration.py` and set `NEXORA_CALIB_A` / `NEXORA_CALIB_B` only if it reports a clear gain.
   - For post-mortems, use https://github.com/LouisP96/metaculus-bot-review.

## Commands

| Command | Purpose |
|---|---|
| `uv run pytest -q` | offline tests (no keys, no network) |
| `uv run python main.py --mode tournament` | seasonal + MiniBench, what the scheduled workflow runs |
| `--mode seasonal` / `--mode minibench` | one target only |
| `--mode test [--dry-run] [--limit N]` | bot-testing-area sandbox; `--dry-run` is refused on live tournaments by design |
| `--profile lean\|standard\|max\|claude-only`, `--minibench-profile ...` | choose the ensemble |
| `scripts/check_setup.py`, `scripts/smoke_quant.py`, `scripts/fit_calibration.py` | setup check, live quant check, recalibration from resolved questions |

**Environment variables** (see `.env.template`; in GitHub they are repo variables, except keys, which are secrets):

- **Required:** `METACULUS_TOKEN`, `OPENROUTER_API_KEY`.
- **Recommended:** `ASKNEWS_CLIENT_ID` / `ASKNEWS_SECRET`, or `ASKNEWS_API_KEY`.
- **Optional:** `FRED_API_KEY`.
- **Tuning:**
  - `NEXORA_PROFILE` (default `standard`) and `NEXORA_MINIBENCH_PROFILE` (the workflow defaults it to `lean`)
  - `NEXORA_FORECASTERS`, e.g. `anthropic/claude-opus-5.5:high,openai/gpt-6-sol:high`, replaces the ensemble
  - `NEXORA_BINARY_CAP` (0.02), `NEXORA_CALIB_A` / `_B`, `NEXORA_MC_FLOOR`
  - `NEXORA_QUANT_WEIGHT_NUMERIC` (0.6), `NEXORA_QUANT_WEIGHT_BINARY` (0.7)
  - `NEXORA_MAX_CONCURRENT_QUESTIONS` (4), `NEXORA_MAX_RUN_MINUTES` (38; job timeout is 50)
  - `NEXORA_USE_ASKNEWS` / `_WEB` / `_MARKETS` / `_QUANT` / `_SIMILAR`
  - `NEXORA_SEASONAL_TOURNAMENT` / `NEXORA_MINIBENCH_TOURNAMENT` (id or slug; for the next season)
  - `NEXORA_PAUSED=true` (GitHub variable) skips scheduled runs.

## Code map and invariants

| File | Role |
|---|---|
| `main.py` | CLI, targets, soft run deadline, run summary. Exit 1 only if every question failed; 2 on setup problems. |
| `nexora/bot.py` | `NexoraBot(ForecastBot)`, built on `forecasting-tools` 0.3.1 (pinned in `uv.lock`) |
| `nexora/config.py` | profiles (`ModelSlot` candidate lists = fallback order), post-processing params, env parsing |
| `nexora/models.py` | `ModelRegistry` (checks OpenRouter's public model list), `call_slot` (tries candidates in order), `make_llm` |
| `nexora/research/` | `__init__.py` orchestrates; `news.py` AskNews; `markets.py`; `similar.py` |
| `nexora/quant/` | `fred.py` (filtered historical simulation), `stocks.py` (lognormal direction probability incl. dividend drop), `community.py` (CP as log-odds random walk), `stats.py` |
| `nexora/prompts.py` | prompts + "playbooks" per MiniBench template (Google Trends base rates, stock direction, FRED, CP) |
| `nexora/parsing.py` | regex parsers; the LLM parser `structure_output` is only a fallback |
| `nexora/distributions.py` | fits percentiles to question bounds; rejects likely unit mistakes (median > 1 range-width outside) |
| `nexora/aggregation.py` | pooling, recalibration, caps, floors |

Invariants that tests rely on. Keep them:

- **Ensemble slots:** each forecaster call picks its slot with `_next_plan()` *synchronously before any await*, so the N concurrent predictions per question use N different slots. `predictions_per_research_report == len(plans)`, and at least 50% must succeed.
- **Research bundles:** stored per question id in `_bundles`. Prompts get the research text and the quant summary separately. A numeric quant model is validated against the question bounds in `run_research` and dropped if implausible.
- **Publishing order:** the framework's own publishing is disabled. `_publish_report` posts the **comment first, then the forecast**. If the forecast post fails, the question stays unforecasted and is retried next run. A forecast without a comment would break prize eligibility.
- **Multiple choice:** forecasting-tools clamps options to at least 1% and rejects renormalisations above 5 points. Always floor with `apply_floor(..., 0.0101)` before building a `PredictedOptionList`.
- **Numeric:** build distributions only via `_make_distribution` (i.e. `fit_points_to_range`). The pooled 201-point CDF is rebuilt with `NumericDistribution.from_question`. `tests/test_numeric_stress.py` covers closed, open and log-scaled bounds.
- **Soft deadline:** questions not started before the deadline raise `DeferredQuestion`. They are counted as deferred, not failed, and the next run picks them up.
- **Network:** every network source is optional and guarded. One failing source must never lose a question.

## Tournament facts (verified 2026-09-26)

- **Fall 2026 seasonal:** 2026-09-28 to 2027-01-06, $50,000 pool, roughly 300-500 questions.
  - forecasting-tools 0.3.1: `CURRENT_AI_COMPETITION_ID = 33121` (`fall-futureeval-2026`), `CURRENT_MINIBENCH_ID = "minibench"`.
  - Questions are released in batches of up to 5 at random times, and each is open for only **about 1.5 hours** (once temporarily 3 h). Hence the 20-minute cron. *Observed 2026-10-08: windows of 3 h (e.g. 12:00-15:00 UTC); 36 questions by then. Post 45615 in the tournament is a notebook, not a question.*
  - **Spot scoring:** only the last forecast before close counts.
- **MiniBench, Fall 2026 (observed 2026-10-08, round 2026-10-05 to 10-23, id 33129, slug `minibench`):** about 60 questions per round, $1,000 each. They are **news-driven short-horizon questions**: politics, courts, sports, crypto and commodity prices, outbreaks. They are released **one at a time at random hours**, each open about **3 hours**, resolving about 10 days later. Of 59 closed questions, 35 were binary, 10 **discrete**, 11 numeric and 3 multiple choice, and **none** matched the FRED / stock / CP / Google Trends templates below. Discrete questions post a CDF of `question.cdf_size` points (4-202 seen), which `tests/test_numeric_stress.py::test_discrete_distributions` covers. `lean` therefore includes Perplexity web search.
- **MiniBench, earlier seasons:** back-to-back two-week rounds of about 60 questions, $1,000 each. Its questions were auto-generated from FRED, stocks, Metaculus CPs and Google Trends. Real title examples:
  - "What will the value of FRED series DGS5 be?" (question text: "What will be the value of '<series title>' on 2026-04-08?")
  - "Will MCO's market close price on 2026-03-13 be higher than its market close price on 2026-03-04?"
  - "Will the community prediction be higher than 31.00% on 2026-01-30 for the Metaculus question '...'?"
  - Google Trends questions are multiple choice (Increases / Doesn't change / Decreases). One public review of Fall 2025 found them resolving 46% decrease / 37% increase / 17% no change, with a ±3-point threshold that creates a floor effect.
- **Rules:**
  - No human in the loop.
  - One prize bot per person; extra bots must be labelled, e.g. with v2.
  - A comment is required on every forecast. Keep comments private; Metaculus publishes them after close.
  - Prize winners must share code or a description and verify their identity.
- **Prizes:** the share is proportional to the **square** of the summed peer score, and nothing is paid if the sum is negative. Unforecast questions score 0.
- **Resources:** AskNews gives about 1,000 calls/month and 4,000 per tournament. A latest-news search (48 h) costs 1 call and an archive search costs 5. So `lean` (MiniBench) searches latest news only (1 call per question), the other profiles use both (6 calls), and calls are spaced 12 s apart for the free-tier rate limit. Metaculus' donated OpenRouter credits cover only OpenAI, Anthropic and Google models, so Perplexity web search may not work on that key; AskNews is then the only news source. Metaculus sponsors LLM costs through the credits form (amount not published).
- **Benchmarks:**
  - Spring 2026 (Jan 7 – Apr 15): 173 bots, 111 external. Bots came close to the pros (gap not statistically significant). High-reasoning variants won 8/8 paired comparisons.
  - Summer 2026: "nostreambot" placed 15th of 277 with about $2.60 per question (3 models, median, AskNews plus several other sources).
- **GitHub Actions:** free for public repos on standard runners; private repos get 2,000 min/month on Free and 3,000 on Pro.

## Design decisions (evidence-based; don't undo without new evidence)

These come from published FutureEval analyses (sources below):

- **Frontier reasoning models at high effort** matter most. The standard profile uses Opus 5.5 ×2 + GPT-6 Sol ×2 + Gemini 3.1 Pro.
- **Ensemble across families:** 86% of Fall 2025 winners ensembled, and team aggregation improved up to about 10 forecasters.
- **Several research sources:** the number of sources correlated with score (r = 0.42).
- **Explicit base rates** (r = +0.38) and **similar resolved questions**: 34% of winners did the latter versus 0% of non-winners.
- **Capping extreme probabilities:** 38% of winners capped; it correlated with rank.
- **Recalibration** (Platt scaling) helped, but only fit it on resolved questions.
- **No "Bayesian" wording** in prompts; it measurably hurt.
- **Numeric handling** was the weak spot in the template: use 11 elicited percentiles, a CDF mixture (linear pool) and the quant blend.
- **Cost:** top-15 bots spent about $1.40 per question versus $0.50 in the bottom half. Defaults are seasonal = `standard` (about $0.6-1.2 per question) and MiniBench = `lean` (about $0.2-0.4, including Perplexity web search), roughly $250-750 per season.
- **FRED model:** tuned on synthetic backtests (80% EWMA / 20% long-run vol, 5% widening). 90% intervals cover about 88-93%.
- **Stock direction** is essentially a coin flip. The model only moves off 50% if the price has already moved since the reference close.

## Hard rules

- **Competition:**
  - Never run the bot on open or upcoming tournament questions, look at the output, and then change the bot. `--dry-run` is locked to the sandbox for this reason.
  - Iterate only on the bot-testing-area or on **resolved** questions.
  - Big changes during the season should be justified by resolved-question evidence.
- **Repo hygiene:**
  - Public repo: no secrets, no personal data.
  - Keep `uv.lock` committed; the CI workflow runs `uv sync --frozen` and the tests on every push.
  - Add or adjust tests with every change and run `uv run pytest -q` before pushing.
- **Upgrading forecasting-tools** changes tournament ids and behaviour. Do it at season boundaries (`uv lock --upgrade-package forecasting-tools`) and re-run the tests.

## Unverified assumptions: check first on a machine with internet

1. **Model ids in `nexora/config.py`.** They were taken from OpenRouter's pages and LiteLLM's model map on 2026-09-26: `anthropic/claude-opus-5.5`, `claude-fable-5.1`, `claude-sonnet-5`, `claude-haiku-4.5`; `openai/gpt-6-sol`, `gpt-6-astra`, `gpt-5.5`; `google/gemini-3.1-pro-preview`, `gemini-3.8-flash`; `perplexity/sonar-pro`. They are resolved at runtime against https://openrouter.ai/api/v1/models, and `check_setup.py` prints the result. Replace any that don't resolve. *2026-10-08: all resolve; the dead Gemini fallback `google/gemini-3-pro-preview` was replaced by `~google/gemini-pro-latest`.*
2. **Reasoning parameter.** Reasoning effort is sent as `extra_body={"reasoning": {"effort": ...}}` with `max_tokens`. The request body was verified against a local capture server, but not the provider behaviour (e.g. Anthropic thinking budgets, or truncation when reasoning eats `max_tokens`). Empty answers raise and fall back to the next candidate.
3. **Similar-questions search.** `research/similar.py` passes `search=<keywords>` to Metaculus `/api/posts/`. If Metaculus ignores the parameter, the keyword-overlap filter drops unrelated results; failures are guarded. *2026-10-08: honoured (search is fuzzy; nonsense terms still return loose matches, which the overlap filter drops).*
4. **CP history.** `quant/community.py` assumes the CP history lives at `question.aggregations.recency_weighted.history` in `/api/posts/{id}/`, and it needs the linked question's URL in the question text. If MiniBench CP questions don't include that URL, the model never runs and only the playbook applies. *Unverifiable for now: the Fall 2026 MiniBench round had no CP questions.*
5. **Title fields.** Which field forecasting-tools maps to `question_text` versus the post title is assumed. Detection checks both.
6. **FRED CSV.** The header may be `observation_date,<ID>` or `DATE,<ID>`, and missing values may be `.` or empty; both are handled. `FRED_API_KEY` switches to the official API.
7. **Yahoo.** The chart endpoint answered 429 to every request from a home IP (2026-09-26 and 2026-10-08), with or without cookie/crumb. `stocks.fetch_history` therefore falls back to Nasdaq's public API (`api.nasdaq.com`; asset class `stocks`, then `etf`; share-class dots kept, e.g. `BRK.B`), verified live. `check_setup.py` prints which source answered. If both fail, the question goes LLM-only.
8. **Publishing.** *Verified on the sandbox 2026-10-08: forecasts recorded, comments private.* Private comments only appear in `/api/comments/` with `author=<bot user id>&is_private=true`; the bot account is required, since personal accounts get 403 on forecasts. Comments are sent with `is_private=True, included_forecast=False`, because our forecast doesn't exist yet when the comment goes first.

## Backlog (only after live verification; validate on sandbox or resolved questions)

- **Market Pulse Q4 2026** (`market-pulse-26q4`; bots are prize-eligible). It needs periodic re-forecasting of numeric group questions, not the current forecast-once logic.
- **Google Trends data** for the MC trend questions; pytrends is unreliable.
- **Stock-price *numeric* questions:** ticker resolution plus unit handling (e.g. "millions of KRW").
- **Agentic gap-filling research** (list missing facts, then search again) for the `max` profile.
- **Backtest harness** on resolved Summer 2026 questions with time-restricted AskNews search to avoid look-ahead leakage.
- **Weight ensemble members** by their resolved-question track record.

## Troubleshooting

- **"Setup problem: ... missing"** means a secret or `.env` entry is missing.
- **A model 404s or keeps falling back:** update the candidates in `config.py` or set `NEXORA_FORECASTERS`.
- **Many "deferred" questions** (e.g. after an outage, when many questions are open at once): raise `NEXORA_MAX_CONCURRENT_QUESTIONS`. Keep `NEXORA_MAX_RUN_MINUTES` below 45.
- **Runs start late:** GitHub cron often starts 5-30 minutes late. That is fine given 1.5-hour windows.
- **Schedule disabled:** GitHub disables scheduled workflows in public repos after 60 days without commits. `forecast.yaml` re-enables itself through the API every Monday around 00h UTC. If it was disabled anyway, re-enable it in the Actions tab (or `gh workflow enable forecast.yaml`) or push any commit.
- **AskNews errors:** check whether access needs renewing this season. Calls are serialised by a lock because of the free-tier rate limit.

## Sources

- **Tournament:** https://www.metaculus.com/tournament/fall-futureeval-2026/ · rules https://www.metaculus.com/aib/contest-rules/ · scoring and prizes https://www.metaculus.com/help/scores-faq/
- **Official template:** https://github.com/Metaculus/metac-bot-template (this bot follows its forecasting-tools structure)
- **Spring 2026 results:** https://www.lesswrong.com/posts/wZBbDqzfBjYG58CxK/futureeval-spring-results-pros-beat-bots-but-the-gap-is
- **What winners did:** https://forum.effectivealtruism.org/posts/Spyz3wESZu2eeqhDj/ai-forecasting-in-2026-what-11-analyses-say
- **Summer 2026 announcement:** https://forum.effectivealtruism.org/posts/ZfLAN557rGWACKtmc/announcing-metaculus-summer-2026-futureeval-bot-tournament
- **Question windows and spot scoring notes:** https://github.com/Daatan/retro/issues/616
- **MiniBench retrospective:** https://predictably.substack.com/p/fall-ai-forecasting-retrospective
- **Comparable open-source bot:** https://github.com/No-Stream/nostreambot-metaculus-bot
