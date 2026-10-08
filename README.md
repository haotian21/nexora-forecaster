# Nexora Forecaster

A Claude-led forecasting bot for the **Metaculus FutureEval bot tournaments**:

- **Fall 2026 seasonal tournament**: Sep 28, 2026 – Jan 6, 2027, $50,000 prize pool.
- **MiniBench**: $1,000 per two-week round of about 60 questions.

It forecasts every new question automatically from GitHub Actions and posts the required private comment with its reasoning. Nothing is traded and no money is at risk; the only cost is LLM usage, which Metaculus partly sponsors.

> **What to expect:** Prize share is proportional to the *square* of your total peer score, so most of the money goes to the top bots. In Spring 2026, 111 external bots competed. A well-built newcomer can realistically earn anywhere from €0 to a few thousand euros per season, and the first season is partly an investment in a public track record. A strong result is also worth something on a quant CV.

## How it works

```
new question ──► research (parallel, each source optional and time-boxed)
                 ├─ statistical model for data-series questions (FRED values, stock direction, CP moves)
                 ├─ AskNews articles (free for tournament bots)
                 ├─ Perplexity web research (via OpenRouter)
                 ├─ read-only prediction-market prices (Polymarket, Manifold)
                 └─ similar RESOLVED Metaculus questions (base rates)
             ──► ensemble: 3–5 forecasters across model families
                 (Claude Opus 5.5 ×2, GPT-6 Sol ×2, Gemini in the standard profile)
             ──► aggregation
                 binary: trimmed log-odds mean → blend with the statistical model → recalibration → cap at 2–98%
                 multiple choice: averaged, every option ≥ 1%
                 numeric/date: averaged CDFs (mixture) blended with the statistical model
             ──► post private comment ──► post forecast   (in that order, so a forecast never lacks its comment)
```

The design follows published FutureEval analyses of what separated winners from the rest:

- Use frontier reasoning models at high effort.
- Ensemble across model families.
- Research from several sources, not one.
- Compute base rates explicitly and look up similar resolved questions.
- Cap extreme probabilities.
- Use proper numeric distributions.
- Avoid the classic bugs: unit mistakes and misreading open questions as already resolved.

The MiniBench question templates of earlier seasons (FRED values, "will X's close be higher", community-prediction moves, Google Trends) each get a dedicated statistical model or playbook. In Fall 2026 MiniBench switched to news-driven questions; the models stay idle unless a question matches a template.

## Setup (about 20 minutes, one time)

1. **Metaculus bot account.** Log in at <https://www.metaculus.com/futureeval/participate/>, create your bot (one bot per person) and copy its **token**.
2. **Register for Fall 2026 and request free LLM credits** with the participant form: <https://forms.gle/aQdYMq9Pisrf1v7d8>. The credits arrive as an **OpenRouter key**. Until then you can create your own key at <https://openrouter.ai/keys>. Either way, **set a credit limit on the key** in the OpenRouter dashboard; that is your hard budget cap.
3. **AskNews (recommended, free):** follow "Getting AskNews Setup" on the [resources page](https://www.metaculus.com/notebooks/38928/ai-benchmark-resources/). Make an AskNews account with your bot account's email, ask AskNews to activate it, then create an `ASKNEWS_API_KEY`. You get about 1,000 calls/month and 4,000 per tournament; the bot spends 1 call per MiniBench question and 6 per seasonal question.
4. **GitHub repository.** Push this folder to a new repository.
   - **Public** is simplest: Actions minutes are free.
   - **Private** works too, but the free plan includes only 2,000 minutes/month and a 20-minute schedule can exceed that. GitHub Pro gives 3,000 and is free with the GitHub Student Developer Pack.
5. **Secrets.** Add these under *Settings → Secrets and variables → Actions → Secrets*:
   - `METACULUS_TOKEN` (required)
   - `OPENROUTER_API_KEY` (required)
   - `ASKNEWS_CLIENT_ID` + `ASKNEWS_SECRET`, or `ASKNEWS_API_KEY` (recommended)
   - `FRED_API_KEY` (optional; the public CSV endpoint normally works without it)
6. **Enable Actions** in the *Actions* tab. Run **"Test bot (bot-testing-area)"** with the defaults (lean profile, 3 questions). The run checks your setup, then forecasts on Metaculus' sandbox. Afterwards, check that the forecasts appear on your bot's profile.
7. **Done.** The **"Forecast on new tournament questions"** workflow runs every 20 minutes. It forecasts each new Fall 2026 and MiniBench question once and skips the ones it has already forecast.

To run locally instead: `uv sync`, copy `.env.template` to `.env`, then run `uv run python scripts/check_setup.py` and `uv run python main.py --mode test --limit 2`.

## Profiles and cost

| Profile | Forecasters | Research | Est. cost / question |
|---|---|---|---|
| `lean` | Opus 5.5, GPT-6 Sol, Sonnet 5 (medium reasoning) | Perplexity, AskNews, markets, similar Qs, stats | ~$0.2–0.4 |
| `standard` (default, seasonal) | Opus 5.5 ×2, GPT-6 Sol ×2, Gemini (high) | + Perplexity | ~$0.6–1.2 |
| `max` | Fable 5.1 ×2, GPT-6 Astra ×2, Opus 5.5 (high) | + Perplexity | ~$2–4 |
| `claude-only` | Fable 5.1, Opus 5.5 ×2, Sonnet 5 (high) | + Perplexity | ~$0.8–2 |

MiniBench uses `lean` by default. In Fall 2026 its questions are short-horizon news questions (released one at a time, each open about 3 hours), so `lean` keeps web search but uses three medium-effort forecasters; at $1k per round the five-model profiles are unlikely to pay back.

A season is roughly 300–500 seasonal questions plus about 480 MiniBench questions. With the defaults, that comes to about **$250–750 per season**. The OpenRouter credit limit caps the spend in any case.

Model ids are checked against OpenRouter's live model list at every run, and each slot falls back to its next candidate if a model is retired or fails. Override the ensemble with the repository variable `NEXORA_FORECASTERS`, for example `anthropic/claude-opus-5.5:high,openai/gpt-6-sol:high`.

## Operating it

- **Pause:** set the repository *variable* `NEXORA_PAUSED=true`, or disable the workflow in the Actions tab.
- **Switch profile:** set the variable `NEXORA_PROFILE` (seasonal) or `NEXORA_MINIBENCH_PROFILE`.
- **Watch it:** each Actions run prints what it forecast, what failed, and the estimated LLM cost. A run only turns red if *every* question failed; the next run retries anything that failed.
- **New season:** tournament ids rotate. Run `uv lock --upgrade-package forecasting-tools`, commit the new lock file, or set `NEXORA_SEASONAL_TOURNAMENT` (id or slug, e.g. `fall-futureeval-2026`). Also renew AskNews access and the credits form each season.
- **Improve it on resolved questions only.** Once at least 60 binary questions have resolved, run `uv run python scripts/fit_calibration.py`. If it reports a clear improvement, set `NEXORA_CALIB_A` and `NEXORA_CALIB_B`. For qualitative post-mortems, the community [metaculus-bot-review](https://github.com/LouisP96/metaculus-bot-review) tool works with any bot.

## Rules you must follow (prize eligibility)

- **No human in the loop.** Never run the bot on open or upcoming tournament questions, read the output, and then change the bot to improve those forecasts. `--dry-run` is therefore limited to the bot-testing-area. Tuning on resolved questions is fine.
- **One bot per person.** A second, experimental bot must be labelled as such, for example with "v2" in the username.
- Every forecast needs a comment; the bot posts a private one first.
- Prize winners must share the code or a description of how the bot works, and verify their identity. Taxes on prizes are the winner's responsibility; in Germany, keep records and check with a Steuerberater.

## Layout

```
main.py                   entry point (modes: tournament | seasonal | minibench | test)
nexora/config.py          profiles, caps, env overrides
nexora/models.py          OpenRouter model resolution + fallback calls
nexora/research/          AskNews, markets, similar questions, orchestration
nexora/quant/             FRED simulation, stock direction, community-prediction models
nexora/prompts.py         prompts + playbooks for MiniBench templates
nexora/parsing.py         strict answer parsers (LLM parser only as fallback)
nexora/distributions.py   fitting percentiles to question bounds, units guard
nexora/aggregation.py     pooling, recalibration, caps
nexora/bot.py             ForecastBot subclass tying it together
scripts/                  check_setup.py, fit_calibration.py
tests/                    offline test suite (no keys needed): uv run pytest
```
