import math

import pytest

from nexora.aggregation import (
    aggregate_binary,
    aggregate_multiple_choice,
    apply_floor,
    blend_weights,
    logit,
    pool_cdfs,
    postprocess_binary,
    sigmoid,
)


def test_logit_sigmoid_roundtrip():
    for p in [0.01, 0.2, 0.5, 0.77, 0.99]:
        assert sigmoid(logit(p)) == pytest.approx(p, abs=1e-9)


def test_aggregate_binary_symmetric_and_trimmed():
    assert aggregate_binary([0.5, 0.5, 0.5]) == pytest.approx(0.5)
    # geometric mean of odds: 0.2 and 0.8 cancel
    assert aggregate_binary([0.2, 0.8]) == pytest.approx(0.5)
    # trimming drops the single wild outlier when n >= 5
    trimmed = aggregate_binary([0.30, 0.32, 0.35, 0.33, 0.99])
    untrimmed = aggregate_binary([0.30, 0.32, 0.35, 0.33, 0.99], trim=False)
    assert trimmed < 0.36 < untrimmed


def test_aggregate_binary_rejects_bad_input():
    with pytest.raises(ValueError):
        aggregate_binary([])
    with pytest.raises(ValueError):
        aggregate_binary([1.2])
    with pytest.raises(ValueError):
        aggregate_binary([float("nan")])


def test_postprocess_caps_and_quant_blend():
    assert postprocess_binary(0.999) == pytest.approx(0.98)
    assert postprocess_binary(0.0001) == pytest.approx(0.02)
    # quant blend pulls towards the statistical model in log-odds space
    blended = postprocess_binary(0.8, quant_probability=0.5, quant_weight=0.7)
    assert 0.5 < blended < 0.65
    # calibration slope > 1 extremizes, < 1 shrinks towards 50%
    assert postprocess_binary(0.7, calib_a=1.5) > 0.7
    assert postprocess_binary(0.7, calib_a=0.7) < 0.7


def test_apply_floor_properties():
    out = apply_floor({"a": 0.0, "b": 1.0, "c": 0.0}, 0.01)
    assert sum(out.values()) == pytest.approx(1.0)
    assert min(out.values()) >= 0.01 - 1e-12
    # many options: floor is reduced so it cannot eat the whole distribution
    many = apply_floor({str(i): (1.0 if i == 0 else 0.0) for i in range(80)}, 0.02)
    assert sum(many.values()) == pytest.approx(1.0)
    assert many["0"] > 0.4


def test_aggregate_multiple_choice():
    options = ["Up", "Same", "Down"]
    pooled = aggregate_multiple_choice(
        [{"Up": 0.6, "Same": 0.1, "Down": 0.3}, {"Up": 0.2, "Same": 0.5, "Down": 0.3}], options, floor=0.01
    )
    assert sum(pooled.values()) == pytest.approx(1.0)
    assert pooled["Up"] == pytest.approx(0.01 + 0.97 * 0.4)
    with pytest.raises(ValueError):
        aggregate_multiple_choice([{"Up": 1.0}], options)


def test_pool_cdfs_monotone_and_weighted():
    a = [0.0, 0.2, 0.9, 1.0]
    b = [0.0, 0.6, 0.7, 1.0]
    pooled = pool_cdfs([a, b])
    assert pooled == pytest.approx([0.0, 0.4, 0.8, 1.0])
    weighted = pool_cdfs([a, b], [3, 1])
    assert weighted[1] == pytest.approx(0.3)
    assert all(x <= y for x, y in zip(weighted, weighted[1:]))


def test_blend_weights():
    assert blend_weights(4, None) == [0.25] * 4
    w = blend_weights(4, 0.6)
    assert w[-1] == pytest.approx(0.6) and math.isclose(sum(w), 1.0)
