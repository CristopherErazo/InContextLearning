"""The term layer icl.theory.terms against the appendix and the exact covariance.

- Algebra (machine precision): the sums of the terms given S and given ell equal the
  literal transcriptions of the appendix (tests/terms_oracle.py, a port of the 2026-10-07
  check); the means equal `ansatz_logits`; the exact level equals the brute second moments.
- Approximations (Monte Carlo, with standard errors): given S vs the exact covariance
  (R1-R3: the variances within ~2%, the covariances within MC error); given ell vs the mean
  of the given-S terms over the A1-A3 sampler (term by term; the O(1/mu) of A3) and vs the
  true chain (2-7% documented).
- The table: keep / presets / select, Bilinear, gradients, GPU.
"""
from __future__ import annotations

import math

import pytest
import torch

import terms_oracle as O
from icl import (ansatz_logits, canonical_permutation, generate_icl_batch, mean_variables, measure_variables,
                 sample_variables, trigger_queries)
from icl.theory.terms import (PRESETS, Bilinear, exact_nontrigger_terms, exact_trigger_terms, nontrigger_terms,
                              nontrigger_terms_mu, trigger_terms, trigger_terms_ell, window_moments)
from icl.theory.variables import (measure_nontrigger_variables, nontrigger_permutation, nontrigger_queries,
                                  sample_nontrigger_variables)

V, K, L, BETA = 64, 12, 128, 0.25
NB = V - K - 1
_u = torch.arange(2, L + 1, dtype=torch.float64) / L
POINTS = {   # mid: bulk and previous-token attention comparable at mu ~ L; trained: previous-token head dominant
    "mid": ({"M_off": 0.02, "Q_on": 1.5, "Q_T": 0.3, "G_on": 1.0, "G_T": 0.4}, 1.0 * (0.4 + 1.2 * _u),
            {"M_on": .09, "M_off": .0025, "Q_on": .16, "Q_T": .09, "Q_N": .12, "G_on": .09, "G_T": .0625, "G_N": .04}),
    "trained": ({"M_off": 0.006, "Q_on": 4.0, "Q_T": -0.2, "G_on": 3.0, "G_T": -1.0},
                3.0 * (1 + 0.6 * torch.sin(2 * math.pi * _u) + 0.3 * _u),
                {"M_on": .25, "M_off": .0009, "Q_on": .36, "Q_T": .0625, "Q_N": .09, "G_on": .16, "G_T": .09,
                 "G_N": .0625}),
}


def order_params(point: str) -> dict:
    means, profile, variances = POINTS[point]
    return {**means, "M_on": float(profile.mean()), "M_profile": profile,
            **{"var_" + name: value for name, value in variances.items()}}


def oracle_params(op):
    return O.Params.from_order_params(op, V, K, L, BETA)


def tok1_of(batch, index):
    sequences = batch["sequence"][index]
    return torch.cat([torch.full((len(index), 1), -1), sequences], 1), sequences


def pick_columns(rows: int, seed: int):
    """Two distinct non-target columns per row."""
    generator = torch.Generator().manual_seed(seed)
    first = torch.randint(0, NB, (rows,), generator=generator)
    second = (first + 1 + torch.randint(0, NB - 1, (rows,), generator=generator)) % NB
    return first, second


def per_row(moments: dict, table, col, col2) -> dict:
    """The class entries of a trigger-query TermTable in the names of the oracle."""
    rows = torch.arange(len(col))
    out = {"var_on": moments["on,on"], "var_T": moments["T,T"], "VTc": moments["T,T'"], "cov_on_T": moments["on,T"],
           "var_b": moments["b,b"][rows, col], "cov_on_b": moments["on,b"][rows, col],
           "cov_b_T": moments["b,T"][rows, col]}
    if isinstance(moments["b,b'"], Bilinear):
        out["cov_b_b2"] = moments["b,b'"].dense()[rows, col, col2]
    return out


def relative(a, b):
    a, b = torch.as_tensor(a, dtype=torch.float64), torch.as_tensor(b, dtype=torch.float64)
    return ((a - b).abs().max() / b.abs().max().clamp(min=1e-300)).item()


# ---- algebra -----------------------------------------------------------------------------------

@pytest.mark.parametrize("point", list(POINTS))
def test_given_S_is_the_appendix(point):
    """Every pair of eq. pairs_trigger (corrected keyset_sums, E) and the means, row by row."""
    op = order_params(point)
    p = oracle_params(op)
    batch = generate_icl_batch(600, V, L, K, generator=torch.Generator().manual_seed(0))
    for mu in (40, 128):
        variables = measure_variables(batch, [mu])
        table = trigger_terms(variables, op, BETA, L, as_written=True)
        index, _ = trigger_queries(batch["sequence"][:, :-1], K, [mu])
        tok1, sequences = tok1_of(batch, index)
        outputs = batch["output_set"][index]
        stats = O.trigger_stats(p, tok1, outputs, mu)
        permutation = canonical_permutation(sequences[:, mu - 1], outputs, K, V)
        col, col2 = pick_columns(len(index), mu)
        rows = torch.arange(len(index))
        F = O.given_S(p, stats, O.background_sums(p, mu), permutation[rows, K + col], permutation[rows, K + col2])
        expected = O.pairs(p, F)
        for name, value in per_row(table.moments(), table, col, col2).items():
            assert relative(value, expected[name]) < 1e-12, (point, mu, name)
        moments = table.moments()
        assert relative(moments["on"], BETA / L * p.Gon * F["Bon"]) < 1e-12
        assert relative(moments["T"], BETA / L * p.GT * F["BT"]) < 1e-12
        assert relative(moments["b"][rows, col], BETA / L * p.Gon * F["Bb"]) < 1e-12


@pytest.mark.parametrize("point", list(POINTS))
def test_means_are_the_ansatz_logits(point):
    op = order_params(point)
    batch = generate_icl_batch(200, V, L, K, generator=torch.Generator().manual_seed(1))
    variables = measure_variables(batch, [50, 128])
    for table, logits in ((trigger_terms(variables, op, BETA, L), ansatz_logits(variables, op, BETA, L)),
                          (trigger_terms_ell([50, 128, 128], [0, 1, 3], op, BETA, L, V, K),
                           ansatz_logits(mean_variables([50, 128, 128], [0, 1, 3], V, K), op, BETA, L))):
        moments = table.moments()
        torch.testing.assert_close(moments["on"], logits[:, -1], rtol=1e-12, atol=1e-15)
        torch.testing.assert_close(moments["T"], logits[:, 0], rtol=1e-12, atol=1e-15)
        torch.testing.assert_close(moments["b"], logits[:, K:V - 1], rtol=1e-12, atol=1e-15)


@pytest.mark.parametrize("point", list(POINTS))
def test_given_ell_is_the_appendix(point):
    """Mean noise (eqs. keyset_means, E_mean, pairs_trigger_ell) and spread of the means
    (eqs. var_logits, cov_logits), per class of non-target, both Lambda and background
    options, against the literal transcriptions with the same window moments and sums."""
    op = order_params(point)
    p = oracle_params(op)
    for background in ("exact", "leading"):
        for Lambda in ("mu", "n-2ell"):
            for mu, ell in ((40, 0), (96, 2), (128, 4)):
                table = trigger_terms_ell(mu, ell, op, BETA, L, V, K, Lambda=Lambda, background=background,
                                          as_written=True)
                w = {name: value.item() for name, value in window_moments(op, mu, L).items()}
                bg = O.background_sums(p, mu)
                if background == "leading":
                    P1, P2, Psi = w["Phi1"], w["Phi2"], w["Psi"]
                    bg.update(D=p.sMon ** 2 * mu + p.sMoff ** 2 * mu ** 2 / 2, K1=mu * (P1 + p.Moff * mu / 2),
                              J2=mu * (P2 + 2 * p.Moff * mu * (P1 - Psi) + p.Moff ** 2 * mu ** 2 / 3),
                              G1=mu * (P2 + p.Moff ** 2 * mu / 2), K2=mu * (P2 + 2 * p.Moff * mu * Psi + p.Moff ** 2 * mu ** 2 / 3))
                lam = (mu if Lambda == "mu" else mu - 2 - 2 * ell) / (V + K)
                noise = table.moments(keep="no_spread")
                spread = table.moments(keep=lambda term: term.channel == "spread")
                for sb, sb2, i, j in ((1, 1, 0, 0), (0, 0, 1, 1), (1, 0, 0, 1)):
                    for mine, theirs in ((noise, O.pairs(p, O.ell_fixed(p, mu, ell, lam, sb, sb2, w, bg))),
                                         (spread, O.spread(p, mu, ell, lam, sb, sb2, w, bg))):
                        view = {pair: table.class_mean(value, pair) for pair, value in mine.items() if "," in pair}
                        values = {"var_on": view["on,on"], "var_T": view["T,T"], "VTc": view["T,T'"],
                                  "cov_on_T": view["on,T"], "var_b": view["b,b"][:, i], "cov_on_b": view["on,b"][:, i],
                                  "cov_b_T": view["b,T"][:, i], "cov_b_b2": view["b,b'"][:, i, j]}
                        for name, value in values.items():
                            expected = torch.tensor([theirs[name]], dtype=torch.float64)
                            torch.testing.assert_close(value, expected, rtol=1e-11, atol=1e-30,
                                                       msg=f"{point} {background} {Lambda} {mu} {ell} {sb}{sb2} {name}")


def test_exact_level_is_exact():
    """The exact terms equal the brute second moments sum E[M M] E[Q Q] E[Gamma Gamma] of one
    sequence (route 2 of the check), every pair, at trigger and non-trigger queries."""
    op = order_params("trained")
    p = oracle_params(op)
    batch = generate_icl_batch(60, V, L, K, generator=torch.Generator().manual_seed(2))
    mu = 70
    table = exact_trigger_terms(batch, [mu], op, BETA, L)
    moments = table.moments()
    index, _ = trigger_queries(batch["sequence"][:, :-1], K, [mu])
    tok1, sequences = tok1_of(batch, index)
    permutation = canonical_permutation(sequences[:, mu - 1], batch["output_set"][index], K, V)
    nt = slice(K, V - 1)
    for row in range(min(len(index), 6)):
        brute = O.brute_cov(p, tok1[row], mu)
        C = brute[permutation[row]][:, permutation[row]]
        block = C[nt, nt] - torch.diag(torch.diagonal(C[nt, nt]))
        classes = table.b_classes.double()
        expected = {"on,on": C[-1, -1], "T,T": C[0, 0], "T,T'": C[0, 1], "on,T": C[-1, 0], "b,b": torch.diagonal(C)[nt],
                    "on,b": C[-1, nt], "b,T": C[nt, 0], "b,b'": classes @ block @ classes.T}
        for pair, value in expected.items():
            mine = moments[pair].class_sums()[row] if pair == "b,b'" else moments[pair][row]
            assert relative(mine, value) < 1e-11, (row, pair)
    table = exact_nontrigger_terms(batch, [mu], op, BETA, L)
    moments = table.moments()
    index, _ = nontrigger_queries(batch["sequence"][:, :-1], K, [mu])
    tok1, _ = tok1_of(batch, index)
    permutation = nontrigger_permutation(batch["output_set"][index], K, V)
    rho = slice(K, V)
    for row in range(6):
        C = O.brute_cov(p, tok1[row], mu)[permutation[row]][:, permutation[row]]
        block = C[rho, rho] - torch.diag(torch.diagonal(C[rho, rho]))
        classes = table.b_classes.double()
        expected = {"T,T": C[0, 0], "T,T'": C[0, 1], "rho,rho": torch.diagonal(C)[rho], "T,rho": C[rho, 0],
                    "rho,rho'": classes @ block @ classes.T}
        for pair, value in expected.items():
            mine = moments[pair].class_sums()[row] if pair == "rho,rho'" else moments[pair][row]
            assert relative(mine, value) < 1e-11, (row, pair)


def test_nontrigger_is_the_appendix():
    """eq. pairs_nontrigger given S and eq. nontrigger_means at mu, against the transcriptions."""
    op = order_params("mid")
    p = oracle_params(op)
    pref, g2 = (BETA / L) ** 2, p.Gon ** 2 + p.sGon ** 2
    batch = generate_icl_batch(300, V, L, K, generator=torch.Generator().manual_seed(3))
    for mu in (40, 128):
        moments = nontrigger_terms(measure_nontrigger_variables(batch, [mu]), op, BETA, L, as_written=True).moments()
        index, _ = nontrigger_queries(batch["sequence"][:, :-1], K, [mu])
        tok1, _ = tok1_of(batch, index)
        stats = O.nontrigger_per_token(p, tok1, batch["output_set"][index], mu)
        bg = O.background_sums(p, mu)
        oracle = O.nontrigger_given_S(p, stats, bg)
        permutation = nontrigger_permutation(batch["output_set"][index], K, V)[:, K:]
        C_rho, C_rho_all = oracle["C_rho"].gather(1, permutation), oracle["C_rho_all"].gather(1, permutation)
        assert relative(moments["rho,rho"], pref * (g2 * C_rho + p.sGN ** 2 * (oracle["EN"] - C_rho))) < 1e-12
        assert relative(moments["T,rho"], pref * p.Gon * p.GT * C_rho_all) < 1e-12
        assert relative(moments["T,T"], pref * (p.GT ** 2 * oracle["CN"] + p.sGT ** 2 * oracle["EN"])) < 1e-12
        table = nontrigger_terms_mu(mu, op, BETA, L, V, K, as_written=True)
        w = {name: value.item() for name, value in window_moments(op, mu, L).items()}
        for s_rho, i in ((1, 0), (0, 1)):
            expected = O.nontrigger_mu(p, mu, mu / (V + K), s_rho, w, bg)
            assert relative(table.class_mean(table.total("T,rho"), "T,rho")[0, i], pref * p.Gon * p.GT * expected["C_rho_all"]) < 1e-12
            var_rho = table.class_mean(table.total("rho,rho"), "rho,rho")[0, i]
            assert var_rho.item() == pytest.approx(pref * (g2 * expected["C_rho"]
                                                           + p.sGN ** 2 * (oracle["EN"] - expected["C_rho"])), rel=1e-12)


def test_same_source_pairs_are_not_token_coherent():
    """Audit items b1, b2 (approved 2026-10-08): the default leaves out of the token-coherent
    terms the pairs sigma = sigma' that the appendix as written counts: kappa_NN sum m^2 over
    the keys with a non-trigger predecessor in T(I, I), kappa_2 G1 in C_d, kappa_2 J2 in C_all.
    Exactly that difference; given ell the NN self-term is kappa_NN (E g^N)^2; and the
    token-coherent parts agree with the exact ones within MC error (as written they are off by
    up to +185%, archive/scratch/2026-10-08-1119_pruned-effective-model.tex, section Audit)."""
    from icl.theory.ansatz import _take
    from icl.theory.noise import _diagonal
    from icl.theory.terms import coincidence_rates
    op = order_params("trained")
    rates = coincidence_rates(V, K)
    pref = (BETA / L) ** 2
    batch = generate_icl_batch(6000, V, L, K, generator=torch.Generator().manual_seed(20))
    mu = 96
    variables = measure_variables(batch, [mu])
    fixed, written = trigger_terms(variables, op, BETA, L), trigger_terms(variables, op, BETA, L, as_written=True)
    diag = _diagonal(op, L, torch.float64, "cpu")
    free = (_take(diag, variables["F_keys"]) ** 2).sum(1)
    tokens = (_take(diag, variables["U_keys"]) ** 2 * ~variables["U_triggered"]).sum(-1)
    factor = pref * op["G_on"] ** 2 * op["var_Q_T"] * rates["kappa_NN"]
    name = "on,on/attention/C_phi.token_coherent.NN"
    torch.testing.assert_close(written[name].value - fixed[name].value, factor * free, rtol=1e-10, atol=1e-30)
    name = "b,b/attention/C_b.token_coherent.NN"
    torch.testing.assert_close(written[name].value - fixed[name].value, factor * tokens, rtol=1e-10, atol=1e-30)
    G1 = (diag[1:mu - 1] ** 2).sum() + op["M_off"] ** 2 * (mu - 2) * (mu - 3) / 2
    name = "T,T/readout.all/E.token_coherent.same_key"
    torch.testing.assert_close(written[name].value - fixed[name].value,
                               pref * op["var_G_T"] * op["var_Q_T"] * rates["kappa_2"] * G1.expand(variables.num_rows))
    # given ell
    table = trigger_terms_ell(mu, 2, op, BETA, L, V, K)
    lam, Phi1 = mu / (V + K), window_moments(op, mu, L)["Phi1"]
    torch.testing.assert_close(table["on,on/attention/C_phi.token_coherent.NN"].value, factor * (lam * Phi1) ** 2)
    # against the exact token-coherent parts
    exact = exact_trigger_terms(batch, [mu], op, BETA, L)
    for keyset, pair in (("C_phi", "on,on"), ("C_b", "b,b")):
        key = (pair, "attention", keyset, "token_coherent")
        mine = fixed.class_mean(fixed.aggregate(("pair", "channel", "keyset", "mechanism"))[key], pair)
        truth = exact.class_mean(exact.aggregate(("pair", "channel", "keyset", "mechanism"))[key], pair)
        mine, truth = mine.reshape(len(mine), -1), truth.reshape(len(truth), -1)
        difference = mine - truth
        z = difference.mean(0) / (difference.std(0) / len(difference) ** 0.5)
        assert (z.abs() < 4).all(), (keyset, z, difference.mean(0) / truth.mean(0))


# ---- approximations ----------------------------------------------------------------------------

def _bias(table, mine: dict, exact: dict):
    """Per pair and class: relative bias of the class means and its z-score."""
    out = {}
    for pair in exact:
        a = table.class_mean(mine[pair], pair).reshape(len(table.rows["mu"]), -1)
        b = table.class_mean(exact[pair], pair).reshape(len(table.rows["mu"]), -1)
        difference = a - b
        scale = b.mean(0).abs()
        out[pair] = (difference.mean(0) / scale, difference.mean(0) / (difference.std(0) / len(a) ** 0.5))
    return out


@pytest.mark.parametrize("point", list(POINTS))
def test_given_S_agrees_with_the_exact_covariance(point):
    """R1-R3 against eq. cov_exact over ~2600 trigger queries per mu: the variances within
    2.5% (the R3 truncation, significant and falling with mu), every covariance within
    4 standard errors or 2%. Same at non-trigger queries."""
    op = order_params(point)
    batch = generate_icl_batch(16000, V, L, K, generator=torch.Generator().manual_seed(4))
    for mu in (64, 128):
        table = trigger_terms(measure_variables(batch, [mu]), op, BETA, L)
        exact = exact_trigger_terms(batch, [mu], op, BETA, L).moments()
        for pair, (bias, z) in _bias(table, table.moments(), exact).items():
            if pair in ("on,on", "T,T", "T,T'", "b,b"):
                assert bias.abs().max() < 0.025, (point, mu, pair, bias)
            else:
                assert ((z.abs() < 4) | (bias.abs() < 0.02)).all(), (point, mu, pair, bias, z)
    small = generate_icl_batch(3000, V, L, K, generator=torch.Generator().manual_seed(5))
    for mu in (64, 128):
        table = nontrigger_terms(measure_nontrigger_variables(small, [mu]), op, BETA, L)
        exact = exact_nontrigger_terms(small, [mu], op, BETA, L).moments()
        for pair, (bias, z) in _bias(table, table.moments(), exact).items():
            if pair in ("T,T", "T,T'", "rho,rho"):
                assert bias.abs().max() < 0.025, (point, mu, pair, bias)
            else:
                assert ((z.abs() < 4) | (bias.abs() < 0.02)).all(), (point, mu, pair, bias, z)


def test_given_ell_is_the_mean_of_given_S():
    """Each term given ell is the mean of the term of the same name given S over the A1-A3
    sampler: within 1.5% except those reading X1, X2, R^phi (exact discrete sums given S,
    continuum moments given ell: the O(1/mu) of A3, 3% at mu = 128), or within 5 standard
    errors. The spread of the means is the covariance of the given-S means (4 s.e.)."""
    op = order_params("mid")
    generator = torch.Generator().manual_seed(6)
    mu, ell, N = 128, 2, 30000
    table = trigger_terms(sample_variables(mu, ell, V, K, num_samples=N, generator=generator, triggered=True),
                          op, BETA, L)
    theory = trigger_terms_ell(mu, ell, op, BETA, L, V, K)
    assert set(theory.names(keep="no_spread")) == set(table.names())
    for term in theory:
        if term.channel == "spread":
            continue
        sampled = theory.class_mean(table[term.name].value, term.pair).reshape(N, -1)
        expected = theory.class_mean(term.value, term.pair).reshape(-1)
        difference = sampled.mean(0) - expected
        error = sampled.std(0) / N ** 0.5
        tolerance = 0.04 if term.detail in ("X1", "X2", "witness_bulk") else 0.015
        ok = (difference.abs() <= tolerance * expected.abs()) | (difference.abs() <= 5 * error)
        assert ok.all(), (term.name, difference / expected, difference / error)
    centred = {cls: table.total(cls) - table.total(cls).mean(0) for cls in ("on", "T", "b")}
    for pair, (X, Y) in {"on,on": ("on", "on"), "T,T": ("T", "T"), "on,T": ("on", "T"), "b,b": ("b", "b"),
                         "on,b": ("on", "b"), "b,T": ("b", "T")}.items():
        x, y = centred[X], centred[Y]
        if x.dim() < y.dim():
            x = x[:, None]
        if y.dim() < x.dim():
            y = y[:, None]
        products = theory.class_mean(x * y, pair).reshape(N, -1)
        expected = theory.class_mean(theory.total(pair, keep=lambda term: term.channel == "spread"), pair).reshape(-1)
        assert ((products.mean(0) - expected).abs() <= 4 * products.std(0) / N ** 0.5 + 1e-3 * expected.abs()).all(), pair


def test_given_ell_against_the_true_chain():
    """Cov(h | ell) = spread of the means + mean noise (eq. cov_total) against the true chain:
    the covariance over generated sequences at fixed ell of the exact means plus the mean of
    the exact covariance. The documented accuracy of A1-A3 is 2-7% on the variances at
    mu ~ 100; checked here within 8% for the variances of the target and the triggers."""
    op = order_params("mid")
    batch = generate_icl_batch(30000, V, L, K, generator=torch.Generator().manual_seed(7))
    mu = 128
    variables = measure_variables(batch, [mu])
    means = trigger_terms(variables, op, BETA, L).moments(keep="means")
    exact = exact_trigger_terms(batch, [mu], op, BETA, L).moments()
    for ell in (1, 2):
        rows = variables["ell"] == ell
        theory = trigger_terms_ell(mu, ell, op, BETA, L, V, K, Lambda="n-2ell")
        for pair, X, Y in (("on,on", "on", "on"), ("T,T", "T", "T"), ("on,T", "on", "T")):
            x, y = means[X][rows], means[Y][rows]
            chain = ((x - x.mean()) * (y - y.mean())).mean() + exact[pair][rows].mean()
            assert theory.total(pair).item() == pytest.approx(chain.item(), rel=0.08), (ell, pair)


def test_nontrigger_mu_is_the_mean_of_given_S():
    op = order_params("trained")
    generator = torch.Generator().manual_seed(8)
    N = 30000
    for mu in (64, 128):
        sampled = nontrigger_terms(sample_nontrigger_variables(mu, V, K, num_samples=N, generator=generator),
                                   op, BETA, L)
        theory = nontrigger_terms_mu(mu, op, BETA, L, V, K)
        for term in theory:
            values = theory.class_mean(sampled[term.name].value, term.pair).reshape(N, -1)
            expected = theory.class_mean(term.value, term.pair).reshape(-1)
            difference = values.mean(0) - expected
            ok = (difference.abs() <= 0.01 * expected.abs()) | (difference.abs() <= 5 * values.std(0) / N ** 0.5)
            assert ok.all(), (mu, term.name)


# ---- variables ---------------------------------------------------------------------------------

def test_triggered_keys():
    """U_triggered: measured, the keys of an output that follow its trigger (none for a plain
    token); sampled, half of the occurrences of the outputs, drawn after everything else."""
    p = O.Params(V=V, K=K, L=L, m=torch.ones(L - 1, dtype=torch.float64))
    batch = generate_icl_batch(400, V, L, K, generator=torch.Generator().manual_seed(9))
    mu = 128
    variables = measure_variables(batch, [mu])
    index, _ = trigger_queries(batch["sequence"][:, :-1], K, [mu])
    tok1, sequences = tok1_of(batch, index)
    stats = O.trigger_stats(p, tok1, batch["output_set"][index], mu)
    permutation = canonical_permutation(sequences[:, mu - 1], batch["output_set"][index], K, V)
    expected = stats["Utr"].gather(1, permutation)[:, K:V - 1]
    assert torch.equal(variables["U_triggered"].sum(-1).double(), expected)
    assert not variables["U_triggered"][:, K - 1:].any()
    assert (variables["U_triggered"] <= (variables["U_keys"] > 0)).all()
    plain = sample_variables(mu, 2, V, K, num_samples=4000, generator=torch.Generator().manual_seed(10))
    flagged = sample_variables(mu, 2, V, K, num_samples=4000, generator=torch.Generator().manual_seed(10),
                               triggered=True)
    for name, value in plain.items():
        if torch.is_tensor(value):
            assert torch.equal(value, flagged[name]), name
    counts = flagged["U_triggered"].sum(-1).double()
    assert counts[:, K - 1:].sum() == 0
    assert counts[:, :K - 1].sum() / flagged["U_bar"][:, :K - 1].sum() == pytest.approx(0.5, abs=0.01)


def test_nontrigger_variables():
    p = O.Params(V=V, K=K, L=L, m=torch.ones(L - 1, dtype=torch.float64))
    batch = generate_icl_batch(300, V, L, K, generator=torch.Generator().manual_seed(11))
    mu = 100
    variables = measure_nontrigger_variables(batch, [mu])
    index, position = nontrigger_queries(batch["sequence"][:, :-1], K, [mu])
    tok1, _ = tok1_of(batch, index)
    stats = O.nontrigger_per_token(p, tok1, batch["output_set"][index], mu)
    permutation = nontrigger_permutation(batch["output_set"][index], K, V)[:, K:]
    for mine, theirs in (("U_bar", "c"), ("W_bar", "Wb")):
        assert torch.equal(variables[mine], stats[theirs].gather(1, permutation)), mine
    assert torch.equal(variables["U_triggered"].sum(-1).double(), stats["Utr"].gather(1, permutation))
    assert torch.equal((variables["U_keys"] > 0).sum(-1).double(), variables["U_bar"])
    sampled = sample_nontrigger_variables(mu, V, K, num_samples=20000, generator=torch.Generator().manual_seed(12))
    rate = mu / (V + K)
    assert sampled["U_bar"][:, :K].mean().item() == pytest.approx(2 * rate, rel=0.02)
    assert sampled["U_bar"][:, K:].mean().item() == pytest.approx(rate, rel=0.02)
    assert sampled["U_triggered"][:, K:].sum() == 0


# ---- the table ---------------------------------------------------------------------------------

def _tables():
    op = order_params("mid")
    batch = generate_icl_batch(100, V, L, K, generator=torch.Generator().manual_seed(13))
    return (trigger_terms(measure_variables(batch, [96]), op, BETA, L), trigger_terms_ell([64, 96], [1, 2], op, BETA, L, V, K),
            nontrigger_terms_mu([64, 96], op, BETA, L, V, K))


def _equal(a, b):
    if isinstance(a, Bilinear):
        return torch.equal(a.dense(), b.dense())
    return torch.equal(a, b)


def test_keep_and_presets():
    for table in _tables():
        names = set(table.names())
        for pair in table.pairs():
            full = table.total(pair)
            for keep in (None, "full", names, lambda term: True):
                assert _equal(table.total(pair, keep=keep), full), (pair, keep)
            assert _equal(table.select(names).total(pair), full)
        # presets nest: means within no_noise within full; no_noise and no_spread cover everything
        means, quiet, no_spread = (table.resolve(name) for name in ("means", "no_noise", "no_spread"))
        assert means <= quiet <= names and quiet | no_spread == names
        assert set(PRESETS) >= {"full", "means", "no_noise", "no_spread"}
        # dropping everything gives zeros of the right shape
        for pair in table.pairs():
            zero = table.total(pair, keep=set())
            assert (zero.dense() == 0).all() if isinstance(zero, Bilinear) and zero.parts else True
        # aggregate is a regrouping of the terms
        groups = table.aggregate(("pair",))
        for pair in table.pairs():
            assert _equal(groups[(pair,)], table.total(pair))
        with pytest.raises(ValueError):
            table.total(table.pairs()[0], keep={"no such term"})
        with pytest.raises(ValueError):
            table.resolve("no such preset")


def test_drop_all_noise_is_no_variances():
    """keep="no_noise" is the model without variances; without variances the noise terms vanish."""
    op = order_params("mid")
    quiet = {name: value for name, value in op.items() if not name.startswith("var_")}
    batch = generate_icl_batch(100, V, L, K, generator=torch.Generator().manual_seed(14))
    variables = measure_variables(batch, [96])
    noisy, silent = trigger_terms(variables, op, BETA, L), trigger_terms(variables, quiet, BETA, L)
    for pair in silent.pairs():
        value = silent.total(pair)
        if isinstance(value, Bilinear):
            assert (value.dense() == 0).all()
        elif "," in pair:
            assert (value == 0).all(), pair
        else:
            assert torch.equal(value, noisy.total(pair, keep="no_noise"))


def test_bilinear():
    generator = torch.Generator().manual_seed(15)
    u, v, w = (torch.randn(3, NB, generator=generator, dtype=torch.float64) for _ in range(3))
    coef = torch.randn(3, generator=generator, dtype=torch.float64)
    matrix = Bilinear(((coef, u, v), (2.0, w, u)))
    dense = coef[:, None, None] * u[:, :, None] * v[:, None, :] + 2 * w[:, :, None] * u[:, None, :]
    dense = dense * (1 - torch.eye(NB, dtype=torch.float64))
    torch.testing.assert_close(matrix.dense(), dense)
    classes = torch.stack([torch.arange(NB) < K - 1, torch.arange(NB) >= K - 1])
    masks = classes.double()
    torch.testing.assert_close(matrix.class_sums(classes), masks @ dense @ masks.T)
    torch.testing.assert_close((3 * matrix + matrix).dense(), 4 * dense)


def test_gradients():
    """Differentiable in the means, the variances and the profile, at both levels."""
    v_, k_, l_ = 12, 3, 16
    batch = generate_icl_batch(40, v_, l_, k_, generator=torch.Generator().manual_seed(16))
    variables = measure_variables(batch, [12, 16])
    names = ("M_off", "Q_T", "G_on", "var_Q_T", "var_M_on", "var_G_N")
    base = {"M_on": 0.8, "Q_on": 1.2, "G_T": -0.3, "var_Q_on": 0.1, "var_Q_N": 0.2, "var_M_off": 0.05,
            "var_G_on": 0.1, "var_G_T": 0.05}

    def function(level):
        def f(*values):
            op = {**base, **dict(zip(names, values[:-1])), "M_profile": values[-1]}
            table = (trigger_terms(variables, op, 0.5, l_) if level == "S"
                     else trigger_terms_ell([12, 16], [1, 2], op, 0.5, l_, v_, k_))
            moments = table.moments()
            return torch.cat([moments["on,on"], moments["on,T"], moments["b,b"].sum(1), moments["b,T"].sum(1),
                              moments["b,b'"].class_sums(table.b_classes).reshape(-1)])
        return f
    inputs = [torch.tensor(value, dtype=torch.float64, requires_grad=True) for value in (0.1, 0.2, 1.1, 0.3, 0.2, 0.1)]
    inputs.append(torch.linspace(0.5, 1.5, l_ - 1, dtype=torch.float64, requires_grad=True))
    for level in ("S", "ell"):
        assert torch.autograd.gradcheck(function(level), inputs), level


@pytest.mark.skipif(not torch.cuda.is_available(), reason="no GPU")
def test_gpu_path():
    op = order_params("trained")
    cuda_op = {name: (value.cuda() if torch.is_tensor(value) else value) for name, value in op.items()}
    batch = generate_icl_batch(400, V, L, K, generator=torch.Generator().manual_seed(17))
    cuda_batch = {name: (value.cuda() if torch.is_tensor(value) else value) for name, value in batch.items()}
    pairs = []
    pairs.append((trigger_terms(measure_variables(batch, [96]), op, BETA, L),
                  trigger_terms(measure_variables(cuda_batch, [96]), cuda_op, BETA, L)))
    pairs.append((trigger_terms_ell([64, 96], [1, 2], op, BETA, L, V, K),
                  trigger_terms_ell([64, 96], [1, 2], cuda_op, BETA, L, V, K, device="cuda")))
    pairs.append((exact_trigger_terms(batch, [96], op, BETA, L), exact_trigger_terms(cuda_batch, [96], cuda_op, BETA, L)))
    for cpu, gpu in pairs:
        for pair in cpu.pairs():
            a, b = cpu.total(pair), gpu.total(pair)
            if isinstance(a, Bilinear):
                a, b = a.dense(), b.dense()
            elif not torch.is_tensor(a):
                a, b = a.sums, b.sums
            torch.testing.assert_close(b.cpu(), a, rtol=1e-10, atol=1e-18)


def test_predecessor_types_averaged_given_the_keys():
    """Without "U_triggered" the given-S terms are their mean over the predecessor types
    given the keys (each key of an output follows its trigger with probability 1/2)."""
    from icl import QueryTable
    from icl.theory.variables import _draw_triggered
    op = order_params("trained")
    rows, draws = 8, 3000
    variables = sample_variables(96, 1, V, K, num_samples=rows, generator=torch.Generator().manual_seed(21))
    averaged = trigger_terms(variables, op, BETA, L).moments()
    repeated = QueryTable({name: (value.repeat(draws, *[1] * (value.dim() - 1)) if torch.is_tensor(value) else value)
                           for name, value in variables.items()})
    repeated["U_triggered"] = _draw_triggered(repeated["U_keys"], K - 1, torch.Generator().manual_seed(22))
    flagged = trigger_terms(repeated, op, BETA, L, pairs=("on,on", "b,b", "b,T")).moments()
    for pair in ("on,on", "b,b", "b,T"):
        values = flagged[pair].reshape(draws, rows, *flagged[pair].shape[1:])
        mean, error = values.mean(0), values.std(0) / draws ** 0.5
        assert ((mean - averaged[pair]).abs() <= 4.5 * error + 1e-12 * averaged[pair].abs()).all(), pair
