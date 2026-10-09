"""Phase 5 of archive/scratch/2026-10-08-1103_plan-pruned-effective-model.md: the power counting
of the terms at a trigger query given ell.

1. Orders. Every term of `trigger_terms_ell` is written (icl.theory.terms.symbolic) in units
   of the signal per earlier a, h0 = (beta / L) G_on Q_on Phi1 (h0^2 for a covariance), as a
   sum of monomials in mu and the ratios of `RATIOS` (r_M = M_off mu / Phi1, s_Moff =
   var_M_off mu / Phi1^2, s_Mon = var_M_on / Phi1^2, r_Q = Q_T / Q_on, s_Q* = var_Q_* / Q_on^2,
   r_G = G_T / G_on, s_G* = var_G_* / G_on^2, d2 = Phi2 / Phi1^2 - 1, dPsi = Psi / Phi1 - 1/2),
   with coefficients in ell, Lambda = p_T mu and rho = K / V (p_T = Lambda / mu).
2. Run_001. At the Phase 4 snapshots and cells: the ratios; per term its power-counting
   estimate (the largest of its monomials with unit coefficient), the dominant monomial, the
   coefficient |value| / estimate and the cancellation |value| / sum |monomials|, next to the
   term's value and share from Phase 4.
3. Scaling. The ratios at mu = L at matched stages of the runs of experiment `power_counting`
   and of run_001: t - t_mid as at the Phase 4 snapshots after the transition (t_mid the first
   evaluation with accuracy >= 1/2; after it, runs of different seeds coincide in t - t_mid,
   not in t / t_mid), and as a check the first step with h0 >= 7 (equal signal); local exponents in L (at V = 128) and in V (at L = 512) by least squares,
   and along fixed lambda = L p_T (L and V together) their sum.
4. The asymptotic order of each term along fixed lambda, ell, mu / L: the largest over its
   monomials of (power of mu) + sum_i n_i a_i, relative to the largest term of its pair at
   run_001; per matched stage, and "robust": the "last" exponents of the ratios whose sign
   is stable across the stages and estimates, 0 for the others. And the local exponent
   d log|t| / d log L at run_001 (monomial exponents weighted by their size there).
5. Against Phase 4: each term's estimated share (estimate with counts over the sum of its
   moment's estimates, largest over the valid cells) vs its measured share and the pruned sets
   (per snapshot; "all" = the three snapshots after the transition against Phase 4's joint set
   over all five).

Writes data/hierarchy/phase5_{monomials,orders,ratios,spearman,estimated_shares,stage_ratios,
scaling,terms}.csv and phase5_summary.json; reads the Phase 4 tables (phase4_terms.csv,
phase4_moments.csv, phase4_summary.json).

    python -m scripts.power_counting             # everything (~5 min, CPU)
    python -m scripts.power_counting --no-runs   # without the scaling (run_001 only)
"""
from __future__ import annotations

import argparse
import json
import math

import numpy as np
import pandas as pd
import sympy as sp
import torch
from tracklab import ExperimentReader

from icl import RunData
from icl.theory.terms import window_moments
from icl.theory.terms.symbolic import COUNT_SYMBOLS, ORDER_SYMBOLS, RATIOS, SYMBOLS, monomials, normalized_terms_ell, \
    ratio_values
from scripts.term_hierarchy import ELL_MAX, FRACTIONS, N_MIN, OUT, RUN, SMOOTHING_WIDTHS, STEPS, Grid, smooth, \
    snapshot_params

EXPERIMENT = "power_counting"
CENTRE = (128, 512)                                    # run_001's V, L
T_MID = 1445                                           # run_001: first evaluation with accuracy >= 1/2
STAGES = {"mid": 1448, "end": 1552, "last": 3000}       # the Phase 4 snapshots after the transition
ROBUST_FLOOR = 0.1
H0_STAGE = 7.0                                         # a second matching: the first step with h0 >= 7
LEARNED = ("end", "last", f"h0={H0_STAGE:g}")
REFERENCE_CELL = (512, 1)                              # (mu, ell) of the per-term summary at run_001
ORDER_NAMES = ("mu", *RATIOS)
MEAN_PARAMS = ("M_on", "M_off", "Q_on", "Q_T", "G_on", "G_T")
VARIANCE_PARAMS = ("var_M_on", "var_M_off", "var_Q_on", "var_Q_T", "var_Q_N", "var_G_on", "var_G_T", "var_G_N")
CLASS_VALUES = {"out": 1, "plain": 0}
THRESHOLDS = (1e-3, 3e-3, 1e-2, 3e-2)                  # on the estimated share, against the Phase 4 pruning


def key_name(key) -> str:
    return " ".join(name if power == 1 else f"{name}^{power}" for name, power in key) or "1"


# ---- 1. orders ------------------------------------------------------------------------------

def moments_of(pair: str) -> list[tuple[str, int, int, float]]:
    """(moment, s, s2, weight) of a pair: the class values that make each Phase 4 moment."""
    if pair in ("b", "b,b", "on,b", "b,T"):
        return [(f"{pair}|{c}", s, 0, 1.0) for c, s in CLASS_VALUES.items()]
    if pair == "b,b'":
        return [("b,b'|out,out", 1, 1, 1.0), ("b,b'|plain,plain", 0, 0, 1.0),
                ("b,b'|out,plain", 1, 0, 0.5), ("b,b'|out,plain", 0, 1, 0.5)]
    return [(pair, 0, 0, 1.0)]


def orders() -> tuple[dict, pd.DataFrame]:
    """{term: (tags, [(key, coefficient function, coefficient)])} and the monomial table."""
    S, C = SYMBOLS, COUNT_SYMBOLS
    arguments = (S["ell"], S["Lambda"], C["rho"], S["s"], S["s2"])
    table, rows = {}, []
    for name, (tags, expression) in normalized_terms_ell().items():
        entries = []
        for key, coefficient in monomials(expression).items():
            entries.append((key, sp.lambdify(arguments, coefficient, "numpy"), coefficient, count_exponents(coefficient)))
            rows.append({"term": name, **{tag: tags[tag] for tag in ("pair", "channel", "column", "mechanism",
                                                                     "keyset", "detail")},
                         "monomial": key_name(key), **{f"n_{order}": dict(key).get(order, 0) for order in ORDER_NAMES},
                         "coefficient": str(coefficient), "coefficient_latex": sp.latex(coefficient)})
        table[name] = (tags, entries)
    return table, pd.DataFrame(rows)


def count_exponents(coefficient: sp.Expr) -> list[tuple[int, int, int, int]]:
    """The powers (ell, Lambda, s, s2) of the monomials of a coefficient (its factors in rho
    left out): the counting factors of the estimate with counts."""
    S = SYMBOLS
    return list(sp.Poly(coefficient, S["ell"], S["Lambda"], S["s"], S["s2"]).monoms())


def count_values(exponents, ell, Lambda, s, s2) -> np.ndarray:
    """The largest counting factor ell^a Lambda^b (s^c s2^d) of a coefficient, per cell."""
    return np.max([ell ** a * Lambda ** b * (s ** c if c else 1) * (s2 ** d if d else 1)
                   for a, b, c, d in exponents], axis=0)


def monomial_values(key, ratios: dict, mu: np.ndarray) -> np.ndarray:
    values = np.ones_like(mu, dtype=float)
    for name, power in key:
        values = values * (mu if name == "mu" else ratios[name]) ** power
    return values


# ---- 2. run_001 -----------------------------------------------------------------------------

def run001_orders(table: dict, steps) -> tuple[pd.DataFrame, pd.DataFrame]:
    run = RunData(*RUN, base_dir="data")
    V, L, K = run.config.model_args.vocab_size, run.config.model_args.seq_len, run.config.data_args.K
    beta = run.config.model_args.beta
    grid = Grid([math.ceil(f * L) for f in FRACTIONS])
    mu, ell = grid.mu.numpy().astype(float), grid.ell.numpy().astype(float)
    ratio_rows, order_rows = [], []
    for step in steps:
        op, _ = snapshot_params(run, step)
        window = {name: value.numpy() for name, value in window_moments(op, grid.mu, L).items()}
        ratios = {name: np.array([ratio_values(op, {k: window[k][c] for k in window}, mu[c], V, K)[name]
                                  for c in range(len(grid))]) for name in (*RATIOS, "rho", "Lambda")}
        h0 = beta * float(op["G_on"]) * float(op["Q_on"]) * window["Phi1"] / L
        for c in range(0, len(grid), ELL_MAX + 1):
            ratio_rows.append({"step": step, "mu": int(mu[c]), "h0": h0[c], **{k: v[c] for k, v in ratios.items()},
                               **{k: window[k][c] for k in window}})
        for name, (tags, entries) in table.items():
            power = 1 if tags["channel"] == "mean" else 2
            for moment, s, s2, weight in moments_of(tags["pair"]):
                sizes = np.stack([monomial_values(key, ratios, mu) for key, _, _, _ in entries])
                coefficients = np.stack([np.broadcast_to(np.asarray(f(ell, ratios["Lambda"], ratios["rho"], s, s2),
                                                                    dtype=float), mu.shape) for _, f, _, _ in entries])
                counts = np.stack([np.broadcast_to(count_values(exponents, ell, ratios["Lambda"], s, s2), mu.shape)
                                   for _, _, _, exponents in entries])
                parts = weight * coefficients * sizes
                for c in range(len(grid)):
                    order_rows.append((step, int(mu[c]), int(ell[c]), moment, name, power, h0[c], parts[:, c],
                                       sizes[:, c], coefficients[:, c], counts[:, c], [key for key, _, _, _ in entries]))
    orders_frame = _collect(order_rows)
    return pd.DataFrame(ratio_rows), orders_frame


def _collect(rows) -> pd.DataFrame:
    """Sum the class halves of b,b'|out,plain, then the per-term columns."""
    merged = {}
    for step, mu, ell, moment, name, power, h0, parts, sizes, coefficients, counts, keys in rows:
        index = (step, mu, ell, moment, name)
        if index in merged:
            merged[index][1] = merged[index][1] + parts
            merged[index][3] = merged[index][3] + coefficients / 2
            merged[index][4] = np.maximum(merged[index][4], counts)
        else:
            merged[index] = [power, parts, sizes, coefficients / (2 if moment == "b,b'|out,plain" else 1), counts, keys, h0]
    out = []
    for (step, mu, ell, moment, name), (power, parts, sizes, coefficients, counts, keys, h0) in merged.items():
        alive = coefficients != 0
        value = parts.sum()
        if not alive.any():
            continue
        estimate = np.abs(sizes[alive]).max()
        with_counts = (np.abs(sizes) * counts)[alive]
        estimate_counts = with_counts.max()
        dominant = int(np.argmax(np.abs(parts)))
        leading = int(np.flatnonzero(alive)[np.argmax(np.abs(sizes[alive]))])
        out.append({"step": step, "mu": mu, "ell": ell, "moment": moment, "term": name,
                    "value_h0": value, "estimate": estimate, "estimate_monomial": key_name(keys[leading]),
                    "dominant_monomial": key_name(keys[dominant]), "dominant_part": parts[dominant],
                    "coefficient": abs(value) / estimate if estimate > 0 else math.nan,
                    "estimate_counts": estimate_counts,
                    "estimate_counts_monomial": key_name(keys[int(np.flatnonzero(alive)[np.argmax(with_counts)])]),
                    "coefficient_counts": abs(value) / estimate_counts if estimate_counts > 0 else math.nan,
                    "cancellation": abs(value) / np.abs(parts).sum() if np.abs(parts).sum() > 0 else math.nan,
                    "value": value * h0 ** power})
    return pd.DataFrame(out)


# ---- 3. scaling -----------------------------------------------------------------------------

def stage_params(run: RunData, step: int) -> tuple[dict, dict]:
    """Means and variances from the logged metrics, the profile from the MProfile artifact at
    `step`, smoothed as in Phase 4 (residual variance = var_M_on)."""
    metrics = run.metrics.loc[step]
    op = {name: float(metrics[name]) for name in (*MEAN_PARAMS, *VARIANCE_PARAMS)}
    raw = run.profile(step)
    residuals = {width: ((raw - smooth(raw, width)) ** 2).mean().item() for width in SMOOTHING_WIDTHS}
    width = min(residuals, key=lambda w: abs(residuals[w] - op["var_M_on"]))
    op["M_profile"] = smooth(raw, width)
    return op, {"width": width}


def stage_ratios() -> pd.DataFrame:
    runs = [RunData(*RUN, base_dir="data")]
    runs += [RunData(EXPERIMENT, rid, base_dir="data") for rid in ExperimentReader(EXPERIMENT, base_dir="data").list_runs()]
    rows = []
    for run in runs:
        config = run.config
        V, L, K = config.model_args.vocab_size, config.model_args.seq_len, config.data_args.K
        metrics = run.metrics
        accuracy = metrics["top1_accuracy"].dropna()
        reached = accuracy[accuracy >= 0.5]
        if reached.empty or "M_on" not in metrics:
            print(f"{run}: no transition, skipped")
            continue
        t_mid = int(reached.index[0])
        profiles = run.steps("profile")
        # the signal per earlier a at mu = L, with M_on for Phi1 (to locate the h0-matched stage)
        signal = (config.model_args.beta * metrics["G_on"] * metrics["Q_on"] * metrics["M_on"] / L).dropna()
        after = signal[(signal.index > t_mid) & (signal >= H0_STAGE)]
        targets = {stage: t_mid + snapshot - T_MID for stage, snapshot in STAGES.items()}
        targets[f"h0={H0_STAGE:g}"] = int(after.index[0]) if len(after) else None
        for stage, target in targets.items():
            if target is None:
                continue
            step = min(profiles, key=lambda s: abs(s - target))
            if abs(step - target) > 10 or step not in metrics.index:
                continue
            op, smoothing = stage_params(run, step)
            mu = L                                                          # the full window
            window = {name: float(value) for name, value in window_moments(op, torch.tensor([mu]), L).items()}
            ratios = ratio_values(op, window, mu, V, K)
            h0 = config.model_args.beta * op["G_on"] * op["Q_on"] * window["Phi1"] / L
            rows.append({"run": f"{run.reader.experiment_name}/{run.run_id}", "V": V, "L": L, "K": K,
                         "seed": config.extra_args.seed, "t_mid": t_mid, "stage": stage, "step": step,
                         "accuracy": float(metrics.loc[step, "top1_accuracy"]), "loss": float(metrics.loc[step, "loss"]),
                         "width": smoothing["width"], "h0": h0, **window, **ratios,
                         **{name: op[name] for name in (*MEAN_PARAMS, *VARIANCE_PARAMS)}})
    return pd.DataFrame(rows)


def fit(frame: pd.DataFrame, column: str, by: str) -> dict:
    """Least-squares slope of log|column| against log(by), with its s.e.; nan when the sign
    varies or fewer than three points / two sizes."""
    values = frame[column].to_numpy(float)
    x = np.log(frame[by].to_numpy(float))
    if len(values) < 3 or len(set(x)) < 2 or not np.all(np.isfinite(values)) or np.any(values == 0):
        return {"exponent": math.nan, "se": math.nan, "n": len(values), "sign": "n/a"}
    signs = np.sign(values)
    y = np.log(np.abs(values))
    A = np.stack([x, np.ones_like(x)], 1)
    coef, residual, *_ = np.linalg.lstsq(A, y, rcond=None)
    dof = len(y) - 2
    sigma2 = float(((y - A @ coef) ** 2).sum() / dof) if dof > 0 else math.nan
    se = math.sqrt(sigma2 * np.linalg.inv(A.T @ A)[0, 0]) if dof > 0 else math.nan
    return {"exponent": coef[0], "se": se, "n": len(values),
            "sign": "+" if np.all(signs > 0) else "-" if np.all(signs < 0) else "varies"}


def scaling(stages: pd.DataFrame) -> pd.DataFrame:
    rows = []
    V0, L0 = CENTRE
    for stage, frame in stages.groupby("stage"):
        for column in (*RATIOS, "h0", *MEAN_PARAMS, *VARIANCE_PARAMS, "Phi1"):
            in_L = fit(frame[frame["V"] == V0], column, "L")
            in_V = fit(frame[frame["L"] == L0], column, "V")
            diagonal = fit(frame[(frame["L"] / frame["V"]) == L0 / V0], column, "L")    # fixed lambda, directly
            lam = in_L["exponent"] + in_V["exponent"]
            rows.append({"stage": stage, "quantity": column, "a_L": in_L["exponent"], "a_L_se": in_L["se"],
                         "n_L": in_L["n"], "sign_L": in_L["sign"], "a_V": in_V["exponent"], "a_V_se": in_V["se"],
                         "n_V": in_V["n"], "sign_V": in_V["sign"], "a_lambda": lam,
                         "a_lambda_se": math.hypot(in_L["se"], in_V["se"]), "a_diagonal": diagonal["exponent"],
                         "a_diagonal_se": diagonal["se"], "n_diagonal": diagonal["n"]})
    return pd.DataFrame(rows)


# ---- 4. asymptotic orders -------------------------------------------------------------------

def robust_exponents(exponents: pd.DataFrame) -> pd.DataFrame:
    """The "last" exponents along fixed lambda, set to 0 for the ratios whose sign is not the
    same across the learned stages and the two estimates (a_lambda = a_L + a_V and the direct
    fixed-lambda a_diagonal); exponents below ROBUST_FLOOR in size count as either sign."""
    rows = []
    for name in RATIOS:
        values = exponents[(exponents["quantity"] == name) & exponents["stage"].isin(LEARNED)]
        signs = np.sign(np.concatenate([values["a_lambda"], values["a_diagonal"]]))
        sizes = np.abs(np.concatenate([values["a_lambda"], values["a_diagonal"]]))
        signs = signs[sizes > ROBUST_FLOOR]
        robust = len(signs) > 0 and (np.all(signs > 0) or np.all(signs < 0))
        last = values[values["stage"] == "last"].iloc[0]
        rows.append({"stage": "robust", "quantity": name, "robust": robust,
                     "a_lambda": last["a_lambda"] if robust else 0.0,
                     "a_lambda_se": last["a_lambda_se"] if robust else 0.0})
    return pd.DataFrame(rows)

def asymptotic(table: dict, exponents: pd.DataFrame, stage: str) -> dict:
    """{term: (exponent, s.e., monomial)} along fixed lambda: the largest over the term's
    monomials of (power of mu) + sum_i n_i a_i (mu ~ L)."""
    a = exponents[exponents["stage"] == stage].set_index("quantity")
    out = {}
    for name, (tags, entries) in table.items():
        best = None
        for key, _, coefficient, _ in entries:
            powers = dict(key)
            value = powers.get("mu", 0) + sum(power * a.loc[n, "a_lambda"] for n, power in powers.items() if n != "mu")
            se = math.sqrt(sum((power * a.loc[n, "a_lambda_se"]) ** 2 for n, power in powers.items() if n != "mu"))
            if best is None or value > best[0]:
                best = (value, se, key_name(key))
        out[name] = best
    return out


# ---- 5. against the Phase 4 pruning ---------------------------------------------------------

def pruning_check(orders_frame: pd.DataFrame, prunable: dict) -> tuple[pd.DataFrame, dict]:
    """The estimated share of each term (its estimate with counts over the sum of the estimates
    of its moment, largest over the valid cells) against its Phase 4 share and the Phase 4
    pruned sets: rank correlation, AUC, and the terms below thresholds."""
    moments = pd.read_csv(OUT / "phase4_moments.csv")[["step", "mu", "ell", "moment", "n_ansatz"]]
    frame = orders_frame.merge(moments, on=["step", "mu", "ell", "moment"])
    frame = frame[frame["n_ansatz"] >= N_MIN].copy()
    cell = ["step", "mu", "ell", "moment"]
    frame["estimated_share"] = frame["estimate_counts"] / frame.groupby(cell)["estimate_counts"].transform("sum")
    rows, checks = [], {}
    for scope, steps in [*((str(step), (step,)) for step in STAGES.values()), ("all", tuple(STAGES.values()))]:
        group = frame[frame["step"].isin(steps)]
        estimated = group.groupby("term")["estimated_share"].max()
        measured = group.groupby("term")["phase4_share"].max()
        pruned = prunable[scope]
        rows += [{"scope": scope, "term": term, "estimated_share": estimated[term], "phase4_share": measured[term],
                  "prunable": term in pruned} for term in estimated.index]
        low, high = estimated[[t in pruned for t in estimated.index]], estimated[[t not in pruned for t in estimated.index]]
        auc = float((low.to_numpy()[:, None] < high.to_numpy()[None, :]).mean())
        checks[scope] = {"spearman": float(estimated.rank().corr(measured.rank())), "auc": auc,
                         "prunable": len(low), "below": {}}
        for threshold in THRESHOLDS:
            below = estimated.index[estimated < threshold]
            checks[scope]["below"][str(threshold)] = {"terms": len(below), "prunable": int(sum(t in pruned for t in below))}
    return pd.DataFrame(rows), checks


def local_exponents(table: dict, ratios: pd.DataFrame, exponents: pd.DataFrame, stage: str) -> dict:
    """{term: d log|t| / d log L} along fixed lambda at run_001's reference cell (step 3000,
    plain columns): the exponents of its monomials averaged with weights |c_m monomial_m|,
    i.e. how the term changes near run_001 (the largest exponent of `asymptotic` may belong
    to a monomial that dominates only at far larger sizes)."""
    mu0, ell0 = REFERENCE_CELL
    here = ratios[(ratios["step"] == 3000) & (ratios["mu"] == mu0)].iloc[0]
    values = {name: np.array([here[name]]) for name in (*RATIOS, "rho", "Lambda")}
    a = exponents[exponents["stage"] == stage].set_index("quantity")["a_lambda"]
    out = {}
    for name, (tags, entries) in table.items():
        weights, rates = [], []
        for key, function, _, _ in entries:
            size = monomial_values(key, values, np.array([float(mu0)]))[0]
            part = float(np.asarray(function(float(ell0), values["Lambda"][0], values["rho"][0], 0, 0))) * size
            powers = dict(key)
            weights.append(abs(part))
            rates.append(powers.get("mu", 0) + sum(p * a.get(n, 0.0) for n, p in powers.items() if n != "mu"))
        out[name] = float(np.dot(weights, rates) / sum(weights)) if sum(weights) > 0 else math.nan
    return out


# ---- main -----------------------------------------------------------------------------------

def per_term_summary(table, orders_frame, phase4_terms, prunable, asymptotic_orders, local=None) -> pd.DataFrame:
    mu0, ell0 = REFERENCE_CELL
    rows = []
    reference = orders_frame[(orders_frame["mu"] == mu0) & (orders_frame["ell"] == ell0)]
    shares = phase4_terms.groupby(["step", "term"])["share"].max()
    for name, (tags, entries) in table.items():
        row = {"term": name, **{tag: tags[tag] for tag in ("pair", "channel", "column", "mechanism", "keyset", "detail")},
               "monomials": len(entries)}
        for step in (1448, 3000):
            here = reference[(reference["step"] == step) & (reference["term"] == name)]
            if len(here):
                first = here.iloc[0]
                row |= {f"{step}/estimate": first["estimate"], f"{step}/estimate_monomial": first["estimate_monomial"],
                        f"{step}/dominant_monomial": first["dominant_monomial"],
                        f"{step}/estimate_counts": first["estimate_counts"],
                        f"{step}/coefficient_counts": first["coefficient_counts"],
                        f"{step}/value_h0": first["value_h0"], f"{step}/coefficient": first["coefficient"],
                        f"{step}/cancellation": first["cancellation"]}
            row[f"{step}/max_share"] = shares.get((step, name), math.nan)
        for stage, orders_by_term in asymptotic_orders.items():
            exponent, se, key = orders_by_term[name]
            row |= {f"{stage}/exponent": exponent, f"{stage}/exponent_se": se, f"{stage}/exponent_monomial": key}
        for stage, rates in (local or {}).items():
            row[f"{stage}/local_exponent"] = rates[name]
        row |= {"prunable/all": name in prunable["all"], "prunable/3000": name in prunable["3000"]}
        rows.append(row)
    frame = pd.DataFrame(rows)
    leaders = frame.groupby("pair")["3000/value_h0"].transform(lambda v: v.abs().idxmax())
    for stage in asymptotic_orders:      # relative to the largest term of each pair at run_001 (3000, reference cell)
        frame[f"{stage}/relative_exponent"] = frame[f"{stage}/exponent"] - frame.loc[leaders, f"{stage}/exponent"].to_numpy()
    for stage in (local or {}):
        frame[f"{stage}/relative_local_exponent"] = (frame[f"{stage}/local_exponent"]
                                                     - frame.loc[leaders, f"{stage}/local_exponent"].to_numpy())
    return frame


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-runs", action="store_true", help="skip the scaling from the power_counting runs")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    table, monomial_frame = orders()
    print(f"{len(table)} terms, {len(monomial_frame)} monomials")
    ratios, orders_frame = run001_orders(table, STEPS)
    phase4_terms = pd.read_csv(OUT / "phase4_terms.csv", keep_default_na=False, na_values={"value": [""]})
    orders_frame = orders_frame.merge(phase4_terms[["step", "mu", "ell", "moment", "term", "value", "share"]]
                                      .rename(columns={"value": "phase4_value", "share": "phase4_share"}),
                                      on=["step", "mu", "ell", "moment", "term"], how="left")
    ratios.to_csv(OUT / "phase5_ratios.csv", index=False)
    orders_frame.to_csv(OUT / "phase5_orders.csv", index=False)
    summary = {"reference_cell": REFERENCE_CELL, "stages": STAGES, "t_mid_run001": T_MID, "ratios": RATIOS}

    asymptotic_orders, local = {}, {}
    if not args.no_runs:
        stages = stage_ratios()
        stages.to_csv(OUT / "phase5_stage_ratios.csv", index=False)
        exponents = scaling(stages)
        exponents.to_csv(OUT / "phase5_scaling.csv", index=False)
        robust = robust_exponents(exponents)
        summary["robust_exponents"] = {row.quantity: row.a_lambda for row in robust.itertuples() if row.robust}
        exponents = pd.concat([exponents, robust], ignore_index=True)
        for stage in (*LEARNED, "robust"):
            asymptotic_orders[stage] = asymptotic(table, exponents, stage)
            local[stage] = local_exponents(table, ratios, exponents, stage)
            for name, (exponent, _, _) in asymptotic_orders[stage].items():
                monomial_frame.loc[monomial_frame["term"] == name, f"{stage}/term_exponent"] = exponent
        a = exponents.set_index(["stage", "quantity"])["a_lambda"]
        for stage in (*LEARNED, "robust"):
            monomial_frame[f"{stage}/exponent"] = monomial_frame["n_mu"] + sum(
                monomial_frame[f"n_{name}"] * a.get((stage, name), math.nan) for name in RATIOS)
    monomial_frame.to_csv(OUT / "phase5_monomials.csv", index=False)

    prunable = json.loads((OUT / "phase4_summary.json").read_text())["prunable"]
    prunable = {scope: set(prunable[scope]["combined"]) for scope in ("all", *map(str, STEPS))}
    estimated, summary["pruning_check"] = pruning_check(orders_frame, prunable)
    estimated.to_csv(OUT / "phase5_estimated_shares.csv", index=False)
    terms = per_term_summary(table, orders_frame, phase4_terms, prunable, asymptotic_orders, local)
    terms.to_csv(OUT / "phase5_terms.csv", index=False)

    # how well the estimate orders the terms: Spearman correlation per (step, cell, moment)
    correlations = []
    for (step, mu, ell, moment), group in orders_frame.groupby(["step", "mu", "ell", "moment"]):
        group = group[group["value_h0"] != 0]
        if len(group) >= 4:
            size = group["value_h0"].abs().rank()
            correlations.append({"step": step, "mu": mu, "ell": ell, "moment": moment,
                                 "spearman": group["estimate"].rank().corr(size),
                                 "spearman_counts": group["estimate_counts"].rank().corr(size)})
    correlations = pd.DataFrame(correlations)
    correlations.to_csv(OUT / "phase5_spearman.csv", index=False)
    summary["spearman_median"] = {column: {int(step): float(group[column].median())
                                           for step, group in correlations.groupby("step")}
                                  for column in ("spearman", "spearman_counts")}
    (OUT / "phase5_summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print(json.dumps({key: summary[key] for key in ("spearman_median", "pruning_check")}, indent=1))


if __name__ == "__main__":
    main()
