"""Numeric distributions must survive every bound configuration Metaculus uses."""

import itertools
import logging

import pytest
from forecasting_tools import DiscreteQuestion, NumericQuestion
from forecasting_tools.data_models.numeric_report import NumericDefaults

from nexora.bot import NexoraBot
from nexora.config import Settings
from nexora.models import ModelRegistry
from nexora.parsing import parse_numeric_percentiles
from nexora.prompts import NUMERIC_PERCENTILES

logging.getLogger("forecasting_tools").setLevel(logging.ERROR)

CONFIGS = [
    (0, 100, False, False, None),
    (0, 100, True, True, None),
    (3, 5, True, True, None),
    (0, 1000, False, True, None),
    (1, 1e6, False, True, 0),  # log-scaled
    (-50, 50, True, False, None),
    (0.001, 0.05, True, True, None),
]


def question(lo, hi, olo, ohi, zp):
    return NumericQuestion(
        question_text="x", id_of_post=1, id_of_question=2, upper_bound=hi, lower_bound=lo,
        open_upper_bound=ohi, open_lower_bound=olo, zero_point=zp, unit_of_measure="u", already_forecasted=False,
    )


@pytest.fixture(scope="module")
def bot():
    return NexoraBot(Settings.from_env(publish=False, profile_name="lean"), ModelRegistry(None), metaculus_client=object())


@pytest.mark.parametrize("config", CONFIGS)
async def test_make_and_pool_distributions(bot, config):
    q = question(*config)
    lo, hi = config[0], config[1]
    width = hi - lo
    made = []
    for center_frac, spread_frac in itertools.product([0.01, 0.3, 0.5, 0.99, 1.3], [0.0005, 0.01, 0.2, 0.6]):
        center = lo + center_frac * width
        points = [(p / 100, center + (p - 50) / 50 * spread_frac * width) for p in [5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 95]]
        dist = bot._make_distribution(points, q)
        cdf = [pt.percentile for pt in dist.get_cdf()]
        assert len(cdf) == 201 and all(a <= b for a, b in zip(cdf, cdf[1:]))
        if not config[2]:
            assert cdf[0] == pytest.approx(0.0, abs=1e-9)
        if not config[3]:
            assert cdf[-1] == pytest.approx(1.0, abs=1e-9)
        made.append(dist)
    pooled = await bot._aggregate_predictions(made[:5], q)
    final_cdf = [pt.percentile for pt in pooled.get_cdf()]
    assert all(a <= b for a, b in zip(final_cdf, final_cdf[1:]))


# Discrete questions (about 1 in 6 MiniBench questions in Fall 2026) post a CDF with
# question.cdf_size points, not 201. Bounds and sizes mirror real questions; small
# integer ranges get the tied answers models actually give ("Percentile 5: 0", "Percentile 10: 0", ...).
DISCRETE = [
    ((-0.5, 3.5, False, False, 5), [0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3]),
    ((-0.5, 2.5, False, False, 4), [0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2]),
    ((19.5, 66.5, False, True, 48), [30, 33, 36, 39, 41, 43, 45, 47, 50, 54, 58]),
    ((2.55, 3.65, True, True, 12), [2.9, 3.0, 3.0, 3.1, 3.1, 3.1, 3.1, 3.2, 3.2, 3.3, 3.3]),
    ((79.495, 81.505, True, True, 202), [80.1, 80.2, 80.3, 80.35, 80.4, 80.45, 80.5, 80.55, 80.6, 80.7, 80.8]),
]


@pytest.mark.parametrize("config,answer", DISCRETE)
async def test_discrete_distributions(bot, config, answer):
    lo, hi, olo, ohi, size = config
    q = DiscreteQuestion(
        question_text="x", id_of_post=1, id_of_question=2, upper_bound=hi, lower_bound=lo, open_upper_bound=ohi,
        open_lower_bound=olo, cdf_size=size, unit_of_measure="u", already_forecasted=False,
    )
    text = "\n".join(f"Percentile {p}: {v}" for p, v in zip(NUMERIC_PERCENTILES, answer))
    points = parse_numeric_percentiles(text, NUMERIC_PERCENTILES)
    shifted = [(p, v + 0.4 * (hi - lo) / size) for p, v in points]  # a second, slightly different forecaster
    made = [bot._make_distribution(points, q), bot._make_distribution(shifted, q)]
    pooled = await bot._aggregate_predictions(made, q)
    max_step = NumericDefaults.get_max_pmf_value(size, include_wiggle_room=False)
    for dist in [*made, pooled]:
        cdf = [pt.percentile for pt in dist.get_cdf()]
        steps = [b - a for a, b in zip(cdf, cdf[1:])]
        assert len(cdf) == size
        assert all(0 <= s <= max_step for s in steps)
        if not olo:
            assert cdf[0] == pytest.approx(0.0, abs=1e-9)
        if not ohi:
            assert cdf[-1] == pytest.approx(1.0, abs=1e-9)


async def test_units_guard(bot):
    q = question(3, 5, True, True, None)
    with pytest.raises(ValueError, match="units"):
        bot._make_distribution([(p / 100, 400 + p) for p in [5, 50, 95]], q)
