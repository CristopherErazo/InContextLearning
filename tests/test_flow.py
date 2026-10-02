"""The gradient-flow integrator: the closed-form solution of a quadratic loss,
freezing by omission, and descent of the effective loss."""
from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from icl import ORDER_PARAMS, EffectiveLoss, integrate


def quadratic_loss(curvature, minimum):
    def loss(order_params):
        return sum(curvature[name] * (torch.as_tensor(order_params.get(name, 0.0), dtype=torch.float64)
                                      - minimum[name]) ** 2 for name in curvature)
    return loss


def test_quadratic_flow_matches_the_closed_form():
    curvature = {"M_on": 1.0, "Q_on": 0.5, "G_on": 2.0}
    minimum = {"M_on": 3.0, "Q_on": -1.0, "G_on": 0.5}
    rates = {"M_on": 0.01, "Q_on": 0.05}                          # G_on is frozen
    start = {"M_on": 0.0, "Q_on": 2.0, "G_on": 1.0}
    flow = integrate(quadratic_loss(curvature, minimum), start, rates, steps=500,
                     record_steps=[0, 100, 250, 500], rtol=1e-10, atol=1e-12)
    assert list(flow.index) == [0, 100, 250, 500]
    assert list(flow.columns) == [*ORDER_PARAMS, "loss"]
    for name, rate in rates.items():
        decay = np.exp(-2 * curvature[name] * rate * flow.index.to_numpy())
        expected = minimum[name] + (start[name] - minimum[name]) * decay
        assert flow[name].to_numpy() == pytest.approx(expected, rel=1e-7, abs=1e-9)
    assert (flow["G_on"] == 1.0).all()                             # not in rates: stays put
    assert (flow["M_off"] == 0.0).all()                            # not given: 0, and frozen
    assert flow["loss"].is_monotonic_decreasing


def test_effective_loss_decreases_along_the_flow():
    loss = EffectiveLoss(48, 96, 9, 0.25, mus=[24, 48, 72, 96])
    start = {"M_on": 2.0, "M_off": 0.01, "Q_on": 3.0, "Q_T": 0.1, "G_on": 2.0, "G_T": 0.1}
    rates = {name: 50.0 for name in ("M_on", "Q_on", "G_on")}     # the signal-only flow
    flow = integrate(loss, start, rates, steps=200, record_steps=np.linspace(0, 200, 11))
    assert flow["loss"].is_monotonic_decreasing
    assert flow["loss"].iloc[-1] < flow["loss"].iloc[0]
    for frozen in ("M_off", "Q_T", "G_T"):
        assert (flow[frozen] == start[frozen]).all()


def test_zero_is_a_fixed_point_and_options_are_checked():
    loss = EffectiveLoss(48, 96, 9, 0.25, mus=[96])
    flow = integrate(loss, {}, {"M_on": 1.0, "Q_on": 1.0, "G_on": 1.0}, steps=10, record_steps=[0, 10])
    assert (flow[["M_on", "Q_on", "G_on"]] == 0.0).all().all()
    assert flow["loss"].iloc[-1] == pytest.approx(math.log(48))
    with pytest.raises(ValueError):
        integrate(loss, {}, {"M_typo": 1.0}, steps=1)
    with pytest.raises(ValueError):
        integrate(loss, {}, {}, steps=1)
