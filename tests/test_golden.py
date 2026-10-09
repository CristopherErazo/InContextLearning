"""Golden values of the public `icl.theory` usages (tests/golden_theory.py): every value
recomputed on the fixture's inputs must equal the stored one, except the documented
changes of EXPECTED_CHANGES. On failure the message lists every value that moved, with
its largest relative change, so a refactor shows exactly which numbers it changed."""
from __future__ import annotations

import fnmatch
from pathlib import Path

import pytest
import torch

from golden_theory import CHANGES, FIXTURE, K, compute
from icl import measure_order_params, measure_variances

# pattern of golden names -> why it moved and by how much. Phase 3 of
# archive/scratch/2026-10-08-1103_plan-pruned-effective-model.md (2026-10-08): noise.py and the closure on the term
# layer, audit items a1, b1, b2, c1, c2 of archive/scratch/2026-10-08-1119_pruned-effective-model.tex. Sizes are the
# largest change relative to the largest stored value of the array. Their new values are in CHANGES.
EXPECTED_CHANGES: dict[str, str] = {
    "*/noise_variances*": "a1 + b1/b2: given S from the term layer (corrected token-coherent rule, no same-source "
                          "pairs). Up to 4% (run001) and 16% (mid, target_trigger) per row; the means move by <= 0.4% "
                          "(Cov(h^on, h_tau) 1.1%), and per row the new values are closer to the exact covariance "
                          "(rms error of target_trigger 11% -> 5%, mid, mu = 512).",
    "*/sample_logits": "follows noise_variances: <= 0.4%.",
    "*/non_trigger_loss": "c1: second order of eq. pairs_nontrigger_mean with Lambda = p_T (mu - 2): <= 1e-6 of L_N "
                          "(the excess over log V moves by <= 2%).",
    "*/EffectiveLoss/mean/full/*": "c2: the closure with the covariance given ell of the term layer, mean noise + spread "
                                   "of the means: loss +1.6e-4, loss_trigg +3.4e-3 (run001); gradients up to 3x "
                                   "(the profile), the spread being 49-85% of Var h^on at ell >= 1.",
    "*/EffectiveLoss/mean/scalar/*": "as mean/full: loss +4e-5, loss_trigg +8.5e-4 (run001).",
    "*/EffectiveLoss/mean/S3/*": "spread=\"always\" default (2026-10-08): the signal-only loss with the closure and the "
                                 "spread of the means: loss +4.6e-5, loss_trigg +9.8e-4 (run001); gradients up to 51%.",
    "integrate/*": "the flow of the signal-only \"mean\" loss, with the spread=\"always\" default: Q_T 13%, others <= 0.3%.",
    "*/EffectiveLoss/mc/full/*": "a1, c1 through sample_logits and non_trigger_loss: losses <= 1.1e-4, gradients <= 7%.",
    "*/EffectiveLoss/mc/scalar/*": "as mc/full: losses <= 1.2e-4, gradients <= 5%.",
}

RTOL = 1e-9


def _relative_change(new: torch.Tensor, old: torch.Tensor) -> float:
    if new.shape != old.shape:
        return float("inf")
    if not old.is_floating_point():
        return float(not torch.equal(new, old))
    new, old = new.double(), old.double()
    scale = old.abs().max().clamp(min=1e-300)
    return ((new - old).abs().max() / scale).item()


@pytest.fixture(scope="module")
def golden():
    if not FIXTURE.exists():
        pytest.skip(f"no fixture at {FIXTURE}: run tests/golden_theory.py")
    return torch.load(FIXTURE)


def test_public_usages_keep_their_values(golden):
    values = compute(golden["inputs"])
    # new values (e.g. a new column of measure_variables) are allowed; every stored one must still exist
    assert set(golden["values"]) <= set(values), sorted(set(golden["values"]) - set(values))
    moved = {name: _relative_change(values[name], old) for name, old in golden["values"].items()}
    moved = {name: change for name, change in moved.items() if change > RTOL}
    unexpected = {name: change for name, change in moved.items()
                  if not any(fnmatch.fnmatch(name, pattern) for pattern in EXPECTED_CHANGES)}
    report = "\n".join(f"  {name}: {change:.3e}" for name, change in sorted(unexpected.items()))
    assert not unexpected, f"{len(unexpected)} golden values moved (max relative change):\n{report}"
    # the expected changes are pinned to their new values
    changes = torch.load(CHANGES) if CHANGES.exists() else {}
    assert set(moved) <= set(changes), sorted(set(moved) - set(changes))[:5]
    drift = {name: _relative_change(values[name], new) for name, new in changes.items()}
    drift = {name: change for name, change in drift.items() if change > RTOL}
    assert not drift, f"{len(drift)} changed golden values moved again: {sorted(drift.items())[:5]}"


def test_run_order_parameters_are_the_measured_ones(golden):
    """The fixture's run_001 inputs are what RunData measures (only where data/ is present)."""
    if not Path("data/full_ansatz/run_001").exists():
        pytest.skip("data/full_ansatz/run_001 not present")
    from icl import RunData
    matrices = RunData("full_ansatz", "run_001", base_dir="data").matrices("last")
    measured = {**measure_order_params(matrices, K), **measure_variances(matrices, K)}
    for name, value in golden["inputs"]["run001"].items():
        if name == "M_profile":
            assert torch.equal(value, matrices["M"].double().diagonal(-1))
        else:
            assert value.item() == measured[name], name
