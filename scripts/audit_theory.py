"""Phase 2 of paper/scratch/2026-10-08-1103_plan-pruned-effective-model.md: audit the public
functions of icl.theory against the term layer (icl.theory.terms) and the exact covariance.

    python -m scripts.audit_theory --part noise   [--device cuda] [--sequences 20000]
    parts: means, sampler, noise, nontrigger, loss (default: all)

"code" is the package before Phase 3 (legacy_noise_variances, legacy_non_trigger_loss below:
the numbers of the audit); since Phase 3 noise.py delegates to icl.theory.terms.

"appendix" is the term layer with as_written=True (the appendix as written).

Two points at the size of full_ansatz/run_001 (V=128, L=512, K=25, beta=0.25): "trained" =
run_001's last step (means, profile, block variances), "mid" = the golden mid point (bulk
and previous-token attention comparable). Every number is written to
data/audit/phase2_<part>.json; a summary is printed.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import torch

from icl import (EffectiveLoss, RunData, ansatz_logits, ansatz_matrices, closure_cross_entropy, generate_icl_batch,
                 logit_blocks, mean_variables, measure_variables, noise_variances, non_trigger_loss, sample_variables)
from icl.theory.ansatz import _previous_token_sums
from icl.theory.ansatz import ORDER_PARAMS, VARIANCES, _brackets, block_variance
from icl.theory.noise import _diagonal, _key_sums, _parameters, _witness_sums
from icl.theory.terms import (exact_nontrigger_terms, exact_trigger_terms, nontrigger_terms_mu, trigger_terms,
                              trigger_terms_ell)
from icl.theory.terms.forms import GivenEll
from icl.theory.terms.trigger import _ell_setup, coincidence_rates, trigger_statistics

V, L, K, BETA = 128, 512, 25, 0.25
NB = V - K - 1
OUT = Path("data/audit")


def points() -> dict:
    run = RunData("full_ansatz", "run_001", base_dir="data")
    trained = run.order_params("last", profile=True, variances=True)
    trained.pop("var_M_on_raw")
    for name in ("var_M", "var_Q", "var_G"):                  # the eight blocks are given: the pooled ones are unused
        trained.pop(name)
    u = torch.arange(2, L + 1, dtype=torch.float64) / L
    sds = {"M_on": 0.3, "M_off": 0.05, "Q_on": 0.8, "Q_T": 0.6, "Q_N": 0.4, "G_on": 0.5, "G_T": 0.6, "G_N": 0.3}
    mid = {"M_on": 1.0, "M_off": 0.01, "Q_on": 3.0, "Q_T": 0.5, "G_on": 2.0, "G_T": -0.7,
           "M_profile": 1.6 - 1.2 * u, **{"var_" + name: sd ** 2 for name, sd in sds.items()}}
    return {"mid": mid, "trained": trained}


def to(op: dict, device) -> dict:
    return {name: (value.to(device) if torch.is_tensor(value) else value) for name, value in op.items()}


def save(part: str, results: dict):
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"phase2_{part}.json"
    path.write_text(json.dumps(results, indent=1))
    print(f"wrote {path}")


def stats(difference: torch.Tensor, reference: torch.Tensor) -> dict:
    """Relative bias of the mean (to the mean of the reference) and its z-score."""
    difference, reference = difference.reshape(-1).double(), reference.reshape(-1).double()
    n = len(difference)
    if n < 2:
        return {"n": n}
    scale = reference.mean().abs().item()
    mean, error = difference.mean().item(), difference.std().item() / n ** 0.5
    return {"n": n, "reference": reference.mean().item(), "rel": mean / scale if scale else math.nan,
            "z": mean / error if error else math.nan}


# ---- noise_variances ---------------------------------------------------------------------------

def code_token_coherent(variables, op) -> dict:
    """The token-coherent parts of C_phi, C_b, C(I_phi, all) as noise.py computes them (the
    count-only kappa_2 rule): what differs from the appendix's corrected rule."""
    dtype, device = variables["N"].dtype, variables["N"].device
    p = _parameters(op, dtype, device)
    V_ = int(variables["K"]) + 1 + variables["U_bar"].size(1)
    kappa_2 = (V_ + 3 * K) / (V_ + K) ** 2
    n_keys = (variables["mu"] - 2).clamp(min=0)
    n = n_keys.to(dtype).clamp(min=1)
    sums = _key_sums(_diagonal(op, L, dtype, device), p["M_off"], n_keys, p["var_M_on"], p["var_M_off"])
    witness = _previous_token_sums(variables, op, L, dtype, device)["witness"]
    K1, K2 = sums["K1"], sums["K2"]
    share = lambda c, N, K1, K2: c * K2 / N + c * (c - 1) * (K1 ** 2 - K2) / N ** 2
    c_phi = variables["N"] + variables["F"]
    return {"C_phi": p["var_Q_T"] * kappa_2 * share(c_phi, n, K1, K2),
            "C_b": p["var_Q_T"] * (kappa_2 * share(variables["U_bar"], n[:, None], K1[:, None], K2[:, None])
                                   + variables["Y_bar"] * sums["mean"][:, None] ** 2),
            "C_phi_all": p["var_Q_T"] * kappa_2 * sums["K1"] * (c_phi / n * sums["K1"] - witness)}


def keyset_mechanism(table, pair, keyset, mechanism, factor):
    """A key-set sum's mechanism, read back from the attention terms of a pair."""
    value = table.aggregate(("pair", "channel", "keyset", "mechanism")).get((pair, "attention", keyset, mechanism))
    return value / factor


def audit_noise(ops, device, sequences, mus=(128, 256, 512)):
    results = {}
    for point, op in ops.items():
        op = to(op, device)
        p = _parameters(op, torch.float64, device)
        pref = (BETA / L) ** 2
        batch = generate_icl_batch(sequences, V, L, K, generator=torch.Generator(device).manual_seed(0), device=device)
        for mu in mus:
            t0 = time.time()
            variables = measure_variables(batch, [mu])
            code = legacy_noise_variances(variables, op, BETA, L)
            appendix = trigger_terms(variables, op, BETA, L, as_written=True)
            exact = exact_trigger_terms(batch, [mu], op, BETA, L, chunk=32)
            a, x = appendix.moments(), exact.moments()
            entries = {"target": ("on,on", lambda m: m["on,on"]),
                       "trigger_common": ("T,T'", lambda m: m["T,T'"]),
                       "trigger_individual": ("T,T", lambda m: m["T,T"] - m["T,T'"]),
                       "non_target": ("b,b", lambda m: m["b,b"]),
                       "target_trigger": ("on,T", lambda m: m["on,T"])}
            ell, f, c_b = variables["ell"], variables["F"], variables["U_bar"]
            outputs = (torch.arange(NB, device=device) < K - 1).expand_as(c_b)
            bins = {"all": torch.ones_like(ell, dtype=torch.bool), "ell=0": ell == 0, "ell=1": ell == 1,
                    "ell=2": ell == 2, "ell>=3": ell >= 3, "f=0": f == 0, "f>=1": f >= 1}
            for name, (pair, get) in entries.items():
                reference, mine = get(x), get(a)
                for label, mask in bins.items():
                    if name == "non_target":
                        for cls, cls_mask in (("output", outputs), ("plain", ~outputs)):
                            for c_label, c_mask in (("all", torch.ones_like(c_b, dtype=torch.bool)), ("c=0", c_b == 0),
                                                    ("c=1", c_b == 1), ("c=2", c_b == 2), ("c>=3", c_b >= 3)):
                                if label != "all" and c_label != "all":
                                    continue
                                m = mask[:, None] & cls_mask & c_mask
                                key = f"{point}/mu={mu}/{name}/{label}/{cls}/{c_label}"
                                results[key] = {"code": stats((code[name] - reference)[m], reference[m]),
                                                "appendix": stats((mine - reference)[m], reference[m])}
                    else:
                        key = f"{point}/mu={mu}/{name}/{label}"
                        results[key] = {"code": stats((code[name] - reference)[mask], reference[mask]),
                                        "appendix": stats((mine - reference)[mask], reference[mask])}
            # token-coherent parts: code (kappa_2 by counts), appendix (corrected rule), exact
            tc_code = code_token_coherent(variables, op)
            for keyset, pair, factor in (("C_phi", "on,on", pref * p["G_on"] ** 2), ("C_b", "b,b", pref * p["G_on"] ** 2),
                                         ("C_phi_all", "on,T", pref * p["G_on"] * p["G_T"])):
                tc_exact = keyset_mechanism(exact, pair, keyset, "token_coherent", factor)
                tc_appendix = keyset_mechanism(appendix, pair, keyset, "token_coherent", factor)
                masks = {"all": torch.ones_like(tc_exact, dtype=torch.bool)}
                if keyset == "C_b":
                    masks = {"output": outputs, "plain": ~outputs}
                for cls, m in masks.items():
                    results[f"{point}/mu={mu}/tc/{keyset}/{cls}"] = {
                        "code": stats((tc_code[keyset] - tc_exact)[m], tc_exact[m]),
                        "appendix": stats((tc_appendix - tc_exact)[m], tc_exact[m]),
                        "share_of_keyset": (tc_exact[m].mean() / keyset_total(exact, pair, keyset, factor)[m].mean()).item()}
            # the rest of the code's key-set sums is the appendix's: check the identity
            for keyset, pair, factor in (("C_phi", "on,on", pref * p["G_on"] ** 2), ("C_b", "b,b", pref * p["G_on"] ** 2)):
                code_total = (code_keyset(variables, op, code, keyset, p, pref))
                rest_code = code_total - tc_code[keyset]
                rest_appendix = (keyset_total(appendix, pair, keyset, factor)
                                 - keyset_mechanism(appendix, pair, keyset, "token_coherent", factor))
                results[f"{point}/mu={mu}/identity/{keyset}"] = ((rest_code - rest_appendix).abs().max()
                                                                  / rest_appendix.abs().max()).item()
            # appendix variants: (A1) without the sigma = sigma' pairs of kappa_NN g^N_I g^N_I in T(I, I);
            # (A2) also without the sigma = sigma' pairs of kappa_2 K2 in C_d and of kappa_2 K_all^2 in C_all
            for variant, deltas in diagonal_variants(variables, op, p).items():
                for name, value in variant_entries(a, deltas, p, pref).items():
                    reference = entries[name][1](x)
                    if name == "non_target":
                        for cls, cls_mask in (("output", outputs), ("plain", ~outputs)):
                            results[f"{point}/mu={mu}/{name}/all/{cls}/all"][variant] = stats((value - reference)[cls_mask],
                                                                                            reference[cls_mask])
                    else:
                        results[f"{point}/mu={mu}/{name}/all"][variant] = stats(value - reference, reference)
            # target_trigger at ell >= 1: the code with and without witness_in_background
            witness = p["var_Q_T"] * _witness_in_background(variables, op, p)
            m = ell >= 1
            reference = x["on,T"][m]
            results[f"{point}/mu={mu}/witness/ell>=1"] = {
                "code": stats(code["target_trigger"][m] - reference, reference),
                "code_without_correction": stats((code["target_trigger"] + pref * p["G_on"] * p["G_T"] * witness)[m]
                                                 - reference, reference),
                "appendix": stats(a["on,T"][m] - reference, reference)}
            print(f"{point} mu={mu}: {variables.num_rows} rows, {time.time() - t0:.1f}s", flush=True)
    return results


def diagonal_variants(variables, op, p) -> dict:
    """Changes of the key-set sums and of the energy when the sigma = sigma' pairs are taken
    out of the token-coherent terms of the appendix."""
    from icl.theory.ansatz import _take
    from icl.theory.terms.trigger import coincidence_rates
    rates = coincidence_rates(V, K)
    dtype, device = variables["N"].dtype, variables["N"].device
    diagonal = _diagonal(op, L, dtype, device)
    n_keys = (variables["mu"] - 2).clamp(min=0)
    sums = _key_sums(diagonal, p["M_off"], n_keys, p["var_M_on"], p["var_M_off"])
    free = (_take(diagonal, variables["F_keys"]) ** 2).sum(1)
    tokens = (_take(diagonal, variables["U_keys"]) ** 2 * (1 - variables["U_triggered"].to(dtype))).sum(-1)
    nn = {"C_phi": -p["var_Q_T"] * rates["kappa_NN"] * free, "C_b": -p["var_Q_T"] * rates["kappa_NN"] * tokens}
    same = {"E": -p["var_Q_T"] * rates["kappa_2"] * sums["G1"], "C_all": -p["var_Q_T"] * rates["kappa_2"] * sums["J2"]}
    return {"appendix_A1": nn, "appendix_A2": {**nn, **same}}


def variant_entries(moments, deltas, p, pref) -> dict:
    """The five entries of noise_variances from the appendix's moments, with key-set sums and
    energy changed by `deltas` (eq. pairs_trigger is linear in them)."""
    zero = 0.0
    dphi, db = deltas.get("C_phi", zero), deltas.get("C_b", zero)
    dE, dall = deltas.get("E", zero), deltas.get("C_all", zero)
    g2 = p["G_on"] ** 2 + p["var_G_on"]
    dE_b = dE[:, None] if torch.is_tensor(dE) else dE
    return {"target": moments["on,on"] + pref * ((g2 - p["var_G_N"]) * dphi + p["var_G_N"] * dE),
            "trigger_common": moments["T,T'"] + pref * p["G_T"] ** 2 * dall,
            "trigger_individual": moments["T,T"] - moments["T,T'"] + pref * p["var_G_T"] * dE,
            "non_target": moments["b,b"] + pref * ((g2 - p["var_G_N"]) * db + p["var_G_N"] * dE_b),
            "target_trigger": moments["on,T"]}


def keyset_total(table, pair, keyset, factor):
    groups = table.aggregate(("pair", "channel", "keyset"))
    return groups[(pair, "attention", keyset)] / factor


def code_keyset(variables, op, code, keyset, p, pref):
    """C_phi or C_b as noise.py builds them (recomputed from its pieces)."""
    dtype, device = variables["N"].dtype, variables["N"].device
    q2_T, q2_on = p["Q_T"] ** 2 + p["var_Q_T"], p["Q_on"] ** 2 + p["var_Q_on"]
    n_keys = (variables["mu"] - 2).clamp(min=0)
    n = n_keys.to(dtype).clamp(min=1)
    diagonal = _diagonal(op, L, dtype, device)
    sums = _key_sums(diagonal, p["M_off"], n_keys, p["var_M_on"], p["var_M_off"])
    kappa_2 = (V + 3 * K) / (V + K) ** 2
    C_d = q2_T * sums["D"] + p["var_Q_T"] * (sums["G1"] + kappa_2 * sums["K2"])
    C_o = p["var_Q_T"] * (sums["J2"] - sums["G1"] + kappa_2 * (sums["K1"] ** 2 - sums["K2"]))
    on = _previous_token_sums(variables, op, L, dtype, device)
    if keyset == "C_phi":
        witness = _witness_sums(variables["N_keys"], diagonal, p["M_off"], n_keys)
        upsilon_P = on["witness"] + p["M_off"] * variables["P"]
        a_pairs = (q2_on - q2_T) * (p["var_M_on"] * variables["N"] + p["var_M_off"] * variables["P"])
        return (legacy_background(variables["N"] + variables["F"], C_d, C_o, n) - p["var_Q_T"] * witness["witness_squared"]
                + a_pairs + p["var_Q_on"] * upsilon_P ** 2)
    P_bar = variables["P_bar"]
    return (legacy_background(variables["U_bar"], C_d, C_o, n) + p["var_Q_T"] * variables["Y_bar"] * sums["mean"][:, None] ** 2
            + (q2_on - q2_T) * p["var_M_off"] * P_bar + p["var_Q_on"] * (p["M_off"] * P_bar) ** 2)


def _witness_in_background(variables, op, p):
    dtype, device = variables["N"].dtype, variables["N"].device
    n_keys = (variables["mu"] - 2).clamp(min=0)
    diagonal = _diagonal(op, L, dtype, device)
    sums = _key_sums(diagonal, p["M_off"], n_keys, p["var_M_on"], p["var_M_off"])
    witness = _witness_sums(variables["N_keys"], diagonal, p["M_off"], n_keys)
    on = _previous_token_sums(variables, op, L, dtype, device)
    kappa_2 = (V + 3 * K) / (V + K) ** 2
    return witness["witness_sent"] + kappa_2 * sums["K1"] * on["witness"]


def print_noise(results):
    print(f"{'key':60s} {'n':>7s} {'code rel':>9s} {'z':>7s} {'app rel':>9s} {'z':>7s}")
    for key, value in results.items():
        if "/identity/" in key:
            print(f"{key:60s} identity of the non-tc parts: {value:.1e}")
            continue
        code, app = value["code"], value["appendix"]
        if code.get("n", 0) < 2:
            continue
        extra = f"  share {value['share_of_keyset']:.3f}" if "share_of_keyset" in value else ""
        extra += "".join(f"  {name[-2:]} {value[name]['rel']:+.4f} ({value[name]['z']:+.1f})"
                         for name in ("appendix_A1", "appendix_A2") if name in value)
        extra += (f"  no-corr {value['code_without_correction']['rel']:+.4f} ({value['code_without_correction']['z']:+.1f})"
                  if "code_without_correction" in value else "")
        print(f"{key:60s} {code['n']:7d} {code['rel']:+9.4f} {code['z']:+7.1f} {app['rel']:+9.4f} {app['z']:+7.1f}{extra}")


# ---- means and sampler ----------------------------------------------------------------------------

def audit_means(ops, device):
    """ansatz_logits / _brackets / mean_variables against the term layer's means (eq. mean_logits
    given S, eq. mean_logits_ell given ell): equal to round-off."""
    results = {}
    batch = generate_icl_batch(2000, V, L, K, generator=torch.Generator(device).manual_seed(1), device=device)
    for point, op in ops.items():
        op = to(op, device)
        variables = measure_variables(batch, [64, 256, 512])
        logits = ansatz_logits(variables, op, BETA, L)
        means = trigger_terms(variables, op, BETA, L).moments(keep="means")
        mus, ells = torch.tensor([64, 256, 512, 512]).repeat_interleave(4), torch.arange(4).repeat(4)
        ell_logits = ansatz_logits(mean_variables(mus, ells, V, K, device=device), op, BETA, L)
        ell_means = trigger_terms_ell(mus, ells, op, BETA, L, V, K, device=device).moments(keep="means")
        for name, (mine, theirs) in {"given_S": (means, logits), "given_ell": (ell_means, ell_logits)}.items():
            for cls, block in (("on", slice(V - 1, V)), ("T", slice(0, 1)), ("b", slice(K, V - 1))):
                value = mine[cls] if mine[cls].dim() == 2 else mine[cls][:, None]
                results[f"{point}/{name}/{cls}"] = ((value - theirs[:, block]).abs().max()
                                                    / theirs[:, block].abs().max()).item()
    return results


SAMPLER_STATS_T = ("Nm", "Fm", "W", "P", "R", "f")
SAMPLER_STATS_B = ("Ub", "Utr", "Wb", "Pb", "c")
NOISE_ONLY = ("Nm2", "Rm", "X1", "X2", "pairs_phi", "Yb", "pairs_b")


def audit_sampler(ops, device, samples=200_000, grid=((64, 0), (64, 2), (256, 1), (256, 3), (512, 2), (512, 5))):
    """The moments of sample_variables (both count laws) against tables moments, cov and eq.
    noise_moments (the means and covariances of GivenEll), with the run_001 profile."""
    results = {}
    op = to(ops["trained"], device)
    for law in ("poisson", "multinomial"):
        for mu, ell in grid:
            generator = torch.Generator(device).manual_seed(mu + ell)
            variables = sample_variables(mu, ell, V, K, num_samples=samples, counts=law, generator=generator,
                                         triggered=True, device=device)
            values = trigger_statistics(variables, op, L)
            *_, evaluator = _ell_setup(mu, ell, op, L, V, K, NB, K - 1, "mu", "exact", torch.float64, device)
            outputs = torch.arange(NB, device=device) < K - 1
            for cls, mask in (("output", outputs), ("plain", ~outputs)):
                b = lambda name: values[name][:, mask].reshape(-1)
                theory_b = lambda table: table[0, mask][0].item() if torch.is_tensor(table) and table.dim() == 2 else table
                for name in (*SAMPLER_STATS_T, "Nm2", "Rm", "X1", "X2", "pairs_phi", *SAMPLER_STATS_B, "Yb", "pairs_b"):
                    is_b = name in SAMPLER_STATS_B or name in ("Yb", "pairs_b")
                    if not is_b and cls == "plain":
                        continue
                    x = b(name) if is_b else values[name]
                    expected = evaluator.means[name]
                    expected = (expected[0, mask][0] if is_b else expected[0]).item()
                    error = x.std().item() / len(x) ** 0.5
                    results[f"{law}/mu={mu}/ell={ell}/mean/{name}{'/' + cls if is_b else ''}"] = {
                        "sampled": x.mean().item(), "table": expected,
                        "rel": (x.mean().item() - expected) / abs(expected) if expected else math.nan,
                        "z": (x.mean().item() - expected) / error if error else math.nan}
                names = [*SAMPLER_STATS_T, *SAMPLER_STATS_B]
                for i, first in enumerate(names):
                    for second in names[i:]:
                        if first in SAMPLER_STATS_T and second in SAMPLER_STATS_T and cls == "plain":
                            continue
                        x = b(first) if first in SAMPLER_STATS_B else values[first].repeat_interleave(int(mask.sum()))
                        y = b(second) if second in SAMPLER_STATS_B else values[second].repeat_interleave(int(mask.sum()))
                        product = (x - x.mean()) * (y - y.mean())
                        table = evaluator._cov(first, second)
                        expected = 0.0 if table is None else (table[0, mask][0] if table.dim() == 2 else table[0]).item()
                        n_rows = samples
                        error = product.std().item() / n_rows ** 0.5      # rows are the independent units
                        sampled = product.mean().item()
                        scale = (x.var() * y.var()).sqrt().item()
                        results[f"{law}/mu={mu}/ell={ell}/cov/{first},{second}/{cls}"] = {
                            "sampled": sampled, "table": expected, "rel_to_sd": (sampled - expected) / scale if scale else 0.0,
                            "z": (sampled - expected) / error if error else math.nan}
            print(f"sampler {law} mu={mu} ell={ell}", flush=True)
    return results


def print_sampler(results):
    for law in ("poisson", "multinomial"):
        rows = {k: v for k, v in results.items() if k.startswith(law)}
        worst = sorted(rows.items(), key=lambda kv: -abs(kv[1]["z"]) if not math.isnan(kv[1]["z"]) else 0)
        print(f"{law}: {len(rows)} moments; max |z| = {abs(worst[0][1]['z']):.1f}")
        for key, value in worst[:12]:
            rel = value.get("rel", value.get("rel_to_sd"))
            print(f"   {key:50s} sampled {value['sampled']:+.5g} table {value['table']:+.5g} rel {rel:+.4f} z {value['z']:+.1f}")


# ---- non_trigger_loss ----------------------------------------------------------------------------

def second_order(table) -> torch.Tensor:
    """L_N - log V to second order in xi, per row: (1/2)[(1/V) sum_c Var xi_c - Var xi_bar],
    from the classes of a non-trigger TermTable (T: K triggers, rho: the V - K others)."""
    var_T, cov_T = table.total("T,T"), table.total("T,T'")
    var_rho = table.total("rho,rho").sum(1)
    cov_T_rho = table.total("T,rho").sum(1)
    cov_rho = table.total("rho,rho'").class_sums(table.b_classes.to(var_T.device)).sum((1, 2))
    trace = K * var_T + var_rho
    total = K * var_T + K * (K - 1) * cov_T + 2 * K * cov_T_rho + var_rho + cov_rho
    return 0.5 * (trace / V - total / V ** 2)


def exact_full_covariance(batch, mu, op, device, rows):
    """(rows, V, V) exact covariance of the logits given the sequence at non-trigger queries
    (eq. cov_exact), in token order, and the sequence index of each row."""
    from icl.theory.terms.exact import _keyset_matrices
    from icl.theory.terms.trigger import _tensors
    from icl.theory.variables import nontrigger_queries
    p = _tensors(op, torch.float64, device)
    inputs = batch["sequence"][:, :-1]
    index, _ = nontrigger_queries(inputs, K, [mu])
    index = index[:rows]
    tokens = inputs[index]
    matrices, _ = _keyset_matrices(tokens, tokens[:, mu - 1], mu, _diagonal(op, L, torch.float64, device), p, False, V)
    C = sum(matrices.values())
    is_trigger = torch.arange(V, device=device) < K
    gamma = torch.where(is_trigger[:, None], p["G_T"], p["G_on"] * torch.eye(V, dtype=torch.float64, device=device))
    var_gamma = torch.where(is_trigger[:, None], p["var_G_T"],
                            torch.where(torch.eye(V, dtype=torch.bool, device=device), p["var_G_on"], p["var_G_N"]))
    energy = torch.diagonal(C, dim1=1, dim2=2)
    return (BETA / L) ** 2 * (gamma @ C @ gamma.T + torch.diag_embed(energy @ var_gamma.T))


def audit_nontrigger(ops, device, mus=(64, 128, 256, 512), networks=128, sequences=256):
    results = {}
    run = RunData("full_ansatz", "run_001", base_dir="data")
    from icl import ReducedTransformer
    from icl.theory.variables import measure_nontrigger_variables
    for point, op in ops.items():
        op = to(op, device)
        batch = generate_icl_batch(sequences, V, L, K, generator=torch.Generator(device).manual_seed(2), device=device)
        inputs = batch["sequence"][:, :-1]
        losses = []
        for seed in range(networks):                              # the model on draws of the variance ansatz
            matrices = ansatz_matrices(to(op, "cpu"), L, V, K, noise=torch.Generator().manual_seed(seed))
            model = ReducedTransformer.from_matrices(matrices, run.config.model_args, device=device, dtype=torch.float64)
            logits = model(inputs)
            per_query = torch.logsumexp(logits, -1) - logits.mean(-1) - math.log(V)
            losses.append(torch.stack([per_query[:, mu - 1][inputs[:, mu - 1] >= K].mean() for mu in mus]))
        losses = torch.stack(losses)
        for i, mu in enumerate(mus):
            code = (legacy_non_trigger_loss(mu, op, BETA, L, V, K, device=device) - math.log(V)).item()
            appendix = second_order(nontrigger_terms_mu(mu, op, BETA, L, V, K, device=device, as_written=True)).item()
            exact = second_order(exact_nontrigger_terms(batch, [mu], op, BETA, L, chunk=32))
            covariance = exact_full_covariance(batch, mu, op, device, rows=200)
            values, vectors = torch.linalg.eigh(covariance)
            normals = torch.randn(len(covariance), 4000, V, dtype=torch.float64, device=device,
                                  generator=torch.Generator(device).manual_seed(mu))
            xi = (normals * values.clamp(min=0).sqrt()[:, None, :]) @ vectors.transpose(1, 2)
            gaussian = (torch.logsumexp(xi, -1) - xi.mean(-1) - math.log(V)).mean(1)
            results[f"{point}/mu={mu}"] = {
                "code": code, "appendix_second_order": appendix,
                "exact_second_order": exact.mean().item(), "exact_second_order_se": exact.std().item() / len(exact) ** 0.5,
                "gaussian_mc_of_exact": gaussian.mean().item(), "gaussian_mc_se": gaussian.std().item() / len(gaussian) ** 0.5,
                "exact_second_order_same_rows": exact[:len(gaussian)].mean().item(),
                "networks": losses[:, i].mean().item(), "networks_se": losses[:, i].std().item() / networks ** 0.5}
            print(f"nontrigger {point} mu={mu}", flush=True)
    return results


def print_nontrigger(results):
    print(f"{'':16s} {'code':>10s} {'appendix':>10s} {'exact 2nd':>16s} {'Gauss MC (2nd)':>24s} {'networks':>18s}")
    for key, r in results.items():
        print(f"{key:16s} {r['code']:10.3e} {r['appendix_second_order']:10.3e} {r['exact_second_order']:10.3e}"
              f"+-{r['exact_second_order_se']:.0e} {r['gaussian_mc_of_exact']:10.3e}+-{r['gaussian_mc_se']:.0e}"
              f" ({r['exact_second_order_same_rows']:.3e}) {r['networks']:10.3e}+-{r['networks_se']:.0e}")


# ---- EffectiveLoss -------------------------------------------------------------------------------

def class_variances(table) -> dict:
    """The five variances of the closure (noise_variances' names) from a given-ell TermTable,
    total (noise + spread) and noise only."""
    out = {}
    for name, keep in (("total", None), ("noise", "no_spread")):
        m = table.moments(keep=keep)
        out[name] = {"target": m["on,on"], "trigger_common": m["T,T'"], "trigger_individual": m["T,T"] - m["T,T'"],
                     "non_target": m["b,b"], "target_trigger": m["on,T"]}
    return out


def full_covariance(table) -> torch.Tensor:
    """(rows, V, V) covariance of the logit vector given ell in the canonical layout, every pair."""
    m = table.moments()
    rows = table.rows.num_rows
    blocks = logit_blocks(K, V)
    T, b, on = blocks["triggers"], blocks["non_target"], V - 1
    C = torch.zeros(rows, V, V, dtype=torch.float64, device=m["on,on"].device)
    C[:, T, T] = m["T,T'"][:, None, None]
    idx = torch.arange(K)
    C[:, idx, idx] = m["T,T"][:, None]
    C[:, on, on] = m["on,on"]
    C[:, on, T] = C[:, T, on] = m["on,T"][:, None]
    C[:, b, b] = m["b,b'"].dense()
    bi = torch.arange(K, V - 1)
    C[:, bi, bi] = m["b,b"]
    C[:, on, b] = C[:, b, on] = m["on,b"]
    C[:, b, T] = m["b,T"][:, :, None]
    C[:, T, b] = m["b,T"][:, None, :]
    return C


def closure_structure(C: torch.Tensor) -> torch.Tensor:
    """The covariance the closure assumes: drop (on, b), (b, T), (b, b')."""
    blocks = logit_blocks(K, V)
    D = C.clone()
    b, T, on = blocks["non_target"], blocks["triggers"], V - 1
    diagonal = torch.diagonal(C[:, b, b], dim1=1, dim2=2).clone()
    D[:, b, :] = 0
    D[:, :, b] = 0
    bi = torch.arange(K, V - 1)
    D[:, bi, bi] = diagonal
    return D


def gaussian_cross_entropy(means, C, draws, seed) -> torch.Tensor:
    values, vectors = torch.linalg.eigh(C)
    normals = torch.randn(len(C), draws, C.size(-1), dtype=torch.float64, device=C.device,
                          generator=torch.Generator(C.device).manual_seed(seed))
    h = means[:, None, :] + (normals * values.clamp(min=0).sqrt()[:, None, :]) @ vectors.transpose(1, 2)
    return (torch.logsumexp(h, -1) - h[..., -1]).mean(1)


def audit_loss(ops, device, mus=(32, 64, 128, 256, 512), sequences=40000):
    results = {}
    # Pi_mu(ell) on the true chain against Poisson(p_T mu)
    from scipy.stats import poisson
    from icl import trigger_queries
    batch = generate_icl_batch(sequences, V, L, K, generator=torch.Generator(device).manual_seed(3), device=device)
    inputs = batch["sequence"][:, :-1]
    empirical = {}
    for mu in mus:
        index, position = trigger_queries(inputs, K, [mu])
        tokens = inputs[index]
        ell = (tokens[:, :mu - 1] == tokens[:, mu - 1:mu]).sum(1).cpu()
        counts = torch.bincount(ell, minlength=40).double()
        pi = counts / counts.sum()
        empirical[mu] = pi
        for label, mean in (("p_T mu", mu / (V + K)), ("p_T (mu-2)", (mu - 2) / (V + K))):
            pois = torch.as_tensor(poisson.pmf(range(len(pi)), mean))
            results[f"Pi/mu={mu}/{label}"] = {"total_variation": 0.5 * (pi - pois).abs().sum().item(),
                                              "mean_ell": (pi * torch.arange(len(pi))).sum().item(), "poisson_mean": mean}
    for point, op in ops.items():
        op = to(op, device)
        for method in ("mean", "mc"):
            loss = EffectiveLoss(V, L, K, BETA, method=method, mus=mus, device=device)
            parts = loss.breakdown(op)
            clusters = parts["clusters"]
            reweighted = 0.0
            for mu in mus:
                rows = clusters["mu"] == mu
                ce, ell = clusters["cross_entropy"][rows], clusters["ell"][rows]
                pi = empirical[mu][ell].to(ce.dtype)
                reweighted += ((pi * ce).sum() / pi.sum()).item()
            reweighted /= len(mus)
            results[f"{point}/EffectiveLoss/{method}"] = {
                "loss": parts["loss"], "loss_trigg": parts["loss_trigg"], "loss_non_trigg": parts["loss_non_trigg"],
                "loss_trigg_with_empirical_Pi": reweighted}
        # the closure at given ell: spread of the means, and the covariances it drops
        grid_mu, grid_ell = torch.tensor([64, 128, 256, 512]).repeat_interleave(4), torch.tensor([0, 1, 2, 4]).repeat(4)
        table = trigger_terms_ell(grid_mu, grid_ell, op, BETA, L, V, K, device=device)
        means = torch.cat([table.total("T")[:, None].expand(-1, K), table.total("b"), table.total("on")[:, None]], 1)
        variances = class_variances(table)
        ce_noise = closure_cross_entropy(means, variances["noise"], K)
        ce_total = closure_cross_entropy(means, variances["total"], K)
        C = full_covariance(table)
        ce_gauss_full = gaussian_cross_entropy(means, C, 20000, 0)
        ce_gauss_closure = gaussian_cross_entropy(means, closure_structure(C), 20000, 0)
        ce_mean = torch.logsumexp(means, 1) - means[:, -1]
        for i in range(len(grid_mu)):
            m = table.moments()
            results[f"{point}/closure/mu={grid_mu[i].item()}/ell={grid_ell[i].item()}"] = {
                "ce_no_noise": ce_mean[i].item(), "ce_closure_noise_only": ce_noise[i].item(),
                "ce_closure_noise_and_spread": ce_total[i].item(),
                "ce_gaussian_closure_structure": ce_gauss_closure[i].item(), "ce_gaussian_full": ce_gauss_full[i].item(),
                "spread_share_on": (variances["total"]["target"][i] - variances["noise"]["target"][i]).item()
                / variances["total"]["target"][i].item(),
                "spread_share_T_common": (variances["total"]["trigger_common"][i] - variances["noise"]["trigger_common"][i]).item()
                / variances["total"]["trigger_common"][i].item(),
                "spread_share_b": ((variances["total"]["non_target"][i] - variances["noise"]["non_target"][i]).sum()
                                   / variances["total"]["non_target"][i].sum()).item(),
                "var_on": m["on,on"][i].item(), "sum_cov_on_b": m["on,b"][i].sum().item(),
                "sum_var_b": m["b,b"][i].sum().item(),
                "sum_cov_bb": m["b,b'"].class_sums(table.b_classes.to(device))[i].sum().item(),
                "sum_cov_bT": m["b,T"][i].sum().item()}
        print(f"loss {point}", flush=True)
    return results


def print_loss(results):
    for key, r in results.items():
        if key.startswith("Pi/"):
            print(f"{key:28s} TV {r['total_variation']:.4f}  E ell {r['mean_ell']:.3f} vs {r['poisson_mean']:.3f}")
        elif "/EffectiveLoss/" in key:
            print(f"{key:28s} " + "  ".join(f"{name} {value:.5f}" for name, value in r.items()))
        else:
            print(f"{key:28s} CE: no noise {r['ce_no_noise']:.4f} | closure noise {r['ce_closure_noise_only']:.4f}"
                  f" +spread {r['ce_closure_noise_and_spread']:.4f} | Gauss closure-structure {r['ce_gaussian_closure_structure']:.4f}"
                  f" full {r['ce_gaussian_full']:.4f} | spread share on {r['spread_share_on']:.2f}"
                  f" Tc {r['spread_share_T_common']:.2f} b {r['spread_share_b']:.2f}")


# ---- the code before Phase 3 (commit 8a826f4), kept to reproduce the audit -------------------------------
# noise.noise_variances and noise.non_trigger_loss as they were: the count-only kappa_2 rule.

def legacy_coincidence_rates(V: int, K: int) -> tuple[float, float]:
    """kappa_2, two keys hold the same token, and kappa_bi, the same token after the
    same predecessor (eqs. kappa2, kbi)."""
    p_T = 1 / (V + K)
    return (V + 3 * K) * p_T ** 2, (K + 1 + 2 * K / V) * p_T ** 2


def legacy_background(counts: torch.Tensor, C_d, C_o, n) -> torch.Tensor:
    """Sigma^alpha(c) (eq. Sigma_alpha): the background attention noise on c keys at
    uniform positions, for counts (num_rows,) or (num_rows, tokens)."""
    if counts.dim() == 2:
        C_d, C_o, n = C_d[:, None], C_o[:, None], n[:, None]
    return counts * C_d / n + counts * (counts - 1) * C_o / n ** 2


def legacy_noise_variances(variables: QueryTable, order_params: dict, beta: float, L: int) -> dict[str, torch.Tensor]:
    """The variances of the logit fluctuation xi at the trigger queries of `variables`
    (measured or sampled: the key columns are needed), eq. noise_trigger:

        "target"              V_on (num_rows,)
        "trigger_common"      V_c^T, the part shared by the K triggers
        "trigger_individual"  V_i^T, i.i.d. over the triggers
        "non_target"          V_b^off (num_rows, V-K-1), in the `non_target` block order
        "target_trigger"      C^{on,T}, the covariance of the target and the common part

    Differentiable in the order parameters, the profile and the variances. Their mean
    given (mu, ell) is the average over `sample_variables` rows: they are not linear in
    the variables, so `mean_variables` would not give it."""
    dtype, device = variables["N"].dtype, variables["N"].device
    K = int(variables["K"])
    V = K + 1 + variables["U_bar"].size(1)
    kappa_2, kappa_bi = legacy_coincidence_rates(V, K)
    p = _parameters(order_params, dtype, device)
    M_off, Q_T, G_on, G_T = (p[name] for name in ("M_off", "Q_T", "G_on", "G_T"))
    delta_Q = p["Q_on"] - Q_T
    q2_T, q2_on = Q_T ** 2 + p["var_Q_T"], p["Q_on"] ** 2 + p["var_Q_on"]
    g2_on = G_on ** 2 + p["var_G_on"]

    n_keys = (variables["mu"] - 2).clamp(min=0)
    n = n_keys.to(dtype).clamp(min=1)
    diagonal = _diagonal(order_params, L, dtype, device)
    sums = _key_sums(diagonal, M_off, n_keys, p["var_M_on"], p["var_M_off"])
    witness = _witness_sums(variables["N_keys"], diagonal, M_off, n_keys)
    on = _previous_token_sums(variables, order_params, L, dtype, device)
    brackets = _brackets(variables, order_params, L)
    B_phi, B_T, B_b = brackets["target"], brackets["trigger"], brackets["non_target"]
    ell, target_keys, P, R, P_bar = (variables["N"], variables["N"] + variables["F"], variables["P"], variables["R"],
                                     variables["P_bar"])

    # the mean attention on the a-sources over the target keys and over all keys (eq. Upsilon), and the
    # incoherent noise of the pairs whose source is an a (eq. AX)
    upsilon_P, upsilon_R = on["witness"] + M_off * P, on["witness"] + M_off * R
    a_pairs = lambda X: (q2_on - q2_T) * (p["var_M_on"] * ell + p["var_M_off"] * X)
    # the background on all keys, diagonal and off-diagonal (eq. background)
    C_d = q2_T * sums["D"] + p["var_Q_T"] * (sums["G1"] + kappa_2 * sums["K2"])
    C_o = p["var_Q_T"] * (sums["J2"] - sums["G1"] + kappa_2 * (sums["K1"] ** 2 - sums["K2"]))
    # the attention noise summed over all keys, the target keys, the keys of b (eqs. Call, Cphi, Cb)
    C_all = (q2_T * sums["D"] + p["var_Q_T"] * (sums["J2"] + kappa_2 * (sums["K1"] - upsilon_R) ** 2)
             + a_pairs(R) + p["var_Q_on"] * upsilon_R ** 2)
    C_phi = (legacy_background(target_keys, C_d, C_o, n) - p["var_Q_T"] * witness["witness_squared"] + a_pairs(P)
             + p["var_Q_on"] * upsilon_P ** 2)
    output_pairs = variables["Y_bar"] * sums["mean"][:, None] ** 2           # (M_on)^2 Y_b, the profile at its mean
    C_b = (legacy_background(variables["U_bar"], C_d, C_o, n) + p["var_Q_T"] * output_pairs
           + (q2_on - q2_T) * p["var_M_off"] * P_bar + p["var_Q_on"] * (M_off * P_bar) ** 2)
    # the readout energy of all keys (eq. E)
    energy = (B_phi ** 2 + p["var_Q_on"] * upsilon_P ** 2 + Q_T ** 2 * sums["K2"]
              + 2 * Q_T * delta_Q * M_off * witness["X1"] + (delta_Q * M_off) ** 2 * witness["X2"]
              + kappa_2 * (B_T - B_phi) ** 2 + C_d + kappa_2 * C_o + a_pairs(R)
              + p["var_Q_T"] * (kappa_bi - kappa_2 ** 2) * sums["all"] ** 2)

    # the row sums of C^{on,T}: on a witness key the previous-token source is an a, already in the var_Q_on term, so
    # what the background counts for it (its weight to all keys and its coincidences) comes out (missing in the note)
    witness_in_background = p["var_Q_T"] * (witness["witness_sent"] + kappa_2 * sums["K1"] * on["witness"])

    prefactor = (beta / L) ** 2
    return {
        "target": prefactor * (p["var_G_on"] * B_phi ** 2 + g2_on * C_phi + p["var_G_N"] * (energy - B_phi ** 2 - C_phi)),
        "trigger_common": prefactor * G_T ** 2 * C_all,
        "trigger_individual": prefactor * p["var_G_T"] * energy,
        "non_target": prefactor * (p["var_G_on"] * B_b ** 2 + g2_on * C_b
                                   + p["var_G_N"] * (energy[:, None] - B_b ** 2 - C_b)),
        "target_trigger": prefactor * G_on * G_T * (target_keys / n * (C_d + C_o) + a_pairs(P)
                                                    + p["var_Q_on"] * upsilon_P * upsilon_R - witness_in_background),
    }


def legacy_non_trigger_loss(mu, order_params: dict, beta: float, L: int, V: int, K: int,
                     dtype=torch.float64, device=None) -> torch.Tensor:
    """The expected cross-entropy at a non-trigger query at position mu (int or 1-D),
    to second order in the fluctuation (eq. LN): log V plus half the variance of the
    logit vector about its own mean, the common modes of the trigger and non-trigger
    entries included. Exactly log V without variances."""
    mu = torch.as_tensor(mu, device=device).reshape(-1)
    kappa_2, kappa_bi = legacy_coincidence_rates(V, K)
    p_T, theta = 1 / (V + K), K / V
    p = _parameters(order_params, dtype, device)
    G_on, G_T, var_Q = p["G_on"], p["G_T"], p["var_Q_N"]
    sums = _key_sums(_diagonal(order_params, L, dtype, device), p["M_off"], (mu - 2).clamp(min=0),
                     p["var_M_on"], p["var_M_off"])
    # the noise of a query row of Q with zero mean (eq. noise_nontrigger)
    C_d = var_Q * (sums["D"] + sums["G1"] + kappa_2 * sums["K2"])
    C_o = var_Q * (sums["J2"] - sums["G1"] + kappa_2 * (sums["K1"] ** 2 - sums["K2"]))
    energy = C_d + kappa_2 * C_o + var_Q * (kappa_bi - kappa_2 ** 2) * sums["all"] ** 2
    # the weights of eq. omegas: own variances minus the common trigger mode and the trigger / non-trigger covariance
    readout = G_on ** 2 + p["var_G_on"] - p["var_G_N"]
    shared = theta * (1 - theta) * G_T ** 2 - 2 * theta * p_T * G_T * G_on
    omega_d = shared + p_T * (readout - G_on ** 2 / V)
    omega_o = shared + p_T ** 2 * ((1 + 2 * theta) * readout - G_on ** 2)
    omega_energy = (1 - 1 / V) * (theta * p["var_G_T"] + (1 - theta) * p["var_G_N"])
    output_pairs = readout * var_Q * theta * (p_T * mu.to(dtype) * sums["mean"]) ** 2
    return math.log(V) + 0.5 * (beta / L) ** 2 * (omega_d * C_d + omega_o * C_o + omega_energy * energy + output_pairs)


PARTS = ("means", "sampler", "noise", "nontrigger", "loss")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--part", choices=PARTS, nargs="*", default=list(PARTS))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--sequences", type=int, default=20000)
    args = parser.parse_args()
    torch.set_grad_enabled(False)
    ops = points()
    if "means" in args.part:
        results = audit_means(ops, args.device)
        save("means", results)
        print("means: max relative difference", max(results.values()))
    if "sampler" in args.part:
        results = audit_sampler(ops, args.device)
        save("sampler", results)
        print_sampler(results)
    if "nontrigger" in args.part:
        results = audit_nontrigger(ops, args.device)
        save("nontrigger", results)
        print_nontrigger(results)
    if "loss" in args.part:
        results = audit_loss(ops, args.device)
        save("loss", results)
        print_loss(results)
    if "noise" in args.part:
        results = audit_noise(ops, args.device, args.sequences)
        save("noise", results)
        print_noise(results)


if __name__ == "__main__":
    main()
