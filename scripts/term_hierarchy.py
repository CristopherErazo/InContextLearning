"""Phase 4 of archive/scratch/2026-10-08-1103_plan-pruned-effective-model.md: the empirical
hierarchy of the terms of the logit moments at full_ansatz/run_001.

    python -m scripts.term_hierarchy [--steps 1138 1345 1448 1552 3000] [--device cuda]
        [--networks 128] [--sequences 4096] [--trained-sequences 262144] [--s-sequences 32768]

At each snapshot (plateau, onset, middle and end of the transition, last step), for every
class moment at trigger queries given (mu, ell), mu in MUS, ell <= ELL_MAX:

  (i)   trained: the run's network on fresh sequences (one quenched network; s.e. by a
        jackknife over blocks of sequences, and the scatter of the moment over the K
        query tokens);
  (ii)  ansatz: `networks` draws of `ansatz_matrices(..., noise=)` at the snapshot's order
        parameters, each on its own sequences (the annealed moments; s.e. by a jackknife
        over the networks);
  (iii) theory: `trigger_terms_ell` with the default options (the closure of
        EffectiveLoss), term by term; and, as a diagnostic, the given-S level averaged
        over the measured variables of fresh sequences (the noise terms averaged per
        (mu, ell), the spread as the covariance of the exact given-S means), so that
        (ii) - (iii) splits into R1-R3 (ii vs S) and A1-A3 (S vs ell).

The order parameters are the measured means and the eight block variances; the profile of
M is the measured sub-diagonal smoothed by a Gaussian kernel whose width makes the residual
variance equal to var_M_on (the measured sub-diagonal is one draw: using it raw would count
the sub-diagonal noise twice).

Class moments (MOMENTS): means of h^on, h_tau, h_b (b an output of another trigger "out"
or a plain token "plain"); Var h^on, Var h_tau ("T,T"), Cov(h_tau, h_tau') ("T,T'"),
Cov(h^on, h_tau), Var h_b, Cov(h^on, h_b), Cov(h_b, h_tau), Cov(h_b, h_b') per class pair,
each averaged over the columns of its class.

Pruning (tolerance agreed 2026-10-08): dropping a set of terms is allowed if every class
moment changes, in every cell (snapshot, mu, ell) with at least N_MIN ansatz rows, by less
than TOL_REL of its full value or by less than TOL_SE of the s.e. of the ansatz moment.
Terms are ranked by their largest share |term| / sum |terms| of a moment over the cells,
and dropped greedily in that order, separately for the means, the spread of the means and
the noise (a term whose drop breaks the tolerance is kept and the next one is tried).

A last greedy pass ("combined") drops noise, then spread, then means with one shared budget,
so that its set of dropped terms is within the tolerance as a whole.

Writes data/hierarchy/phase4_{moments,terms,ranking,pruning,observations}.csv and
phase4_summary.json; prints a summary. --from-csv redoes only the ranking and the pruning
from the CSVs (e.g. after changing the tolerance).
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import pandas as pd
import torch

from icl import ReducedTransformer, RunData, ansatz_matrices, logit_table, measure_variables
from icl.theory.terms import trigger_terms, trigger_terms_ell
from icl.theory.terms.table import Bilinear, ClassPairSums

RUN = ("full_ansatz", "run_001")
STEPS = (1138, 1345, 1448, 1552, 3000)        # plateau, onset, middle, end of the transition, last step
FRACTIONS = (1 / 16, 1 / 8, 1 / 4, 1 / 2, 3 / 4, 1)
ELL_MAX = 12
N_MIN = 400                                   # ansatz rows for a cell to enter the tolerance
TOL_REL, TOL_SE = 0.02, 0.2
SMOOTHING_WIDTHS = (2, 3, 4, 6, 8, 12, 16, 24, 32)
OUT = Path("data/hierarchy")

MEANS = ("on", "T", "b|out", "b|plain")
COVARIANCES = ("on,on", "T,T", "T,T'", "on,T", "b,b|out", "b,b|plain", "on,b|out", "on,b|plain", "b,T|out",
               "b,T|plain", "b,b'|out,out", "b,b'|out,plain", "b,b'|plain,plain")
MOMENTS = MEANS + COVARIANCES
CLASSES = ("out", "plain")
GROUPS = {"means": ("mean",), "spread": ("spread",), "noise": ("attention", "readout")}


# ---- order parameters -----------------------------------------------------------------------

def smooth(profile: torch.Tensor, width: float) -> torch.Tensor:
    """Nadaraya-Watson smoothing with a Gaussian kernel of `width` positions."""
    x = torch.arange(len(profile), dtype=torch.float64)
    kernel = torch.exp(-0.5 * ((x[:, None] - x[None, :]) / width) ** 2)
    return kernel @ profile / kernel.sum(1)


def snapshot_params(run: RunData, step: int) -> tuple[dict, dict]:
    """The snapshot's means, eight block variances and smoothed profile, and how the
    profile was smoothed."""
    op = run.order_params(step, profile=True, variances=True)
    for name in ("var_M", "var_Q", "var_G", "var_M_on_raw"):
        op.pop(name)
    raw = op["M_profile"]
    residuals = {width: ((raw - smooth(raw, width)) ** 2).mean().item() for width in SMOOTHING_WIDTHS}
    width = min(residuals, key=lambda w: abs(residuals[w] - op["var_M_on"]))
    op["M_profile"] = smooth(raw, width)
    return op, {"width": width, "residual_variance": residuals[width], "var_M_on": op["var_M_on"]}


def ratio(a: float, b: float) -> float:
    return a / b if b else math.nan


def to(op: dict, device) -> dict:
    return {name: (value.to(device) if torch.is_tensor(value) else value) for name, value in op.items()}


# ---- empirical class moments ----------------------------------------------------------------

class Accumulator:
    """Sufficient statistics of the logit vectors (canonical layout) per (cell, group):
    the count, per-column sums and sums of squares, and the first two moments of the four
    aggregates z = (h^on, sum of the triggers, sum of the outputs, sum of the plain tokens)."""

    def __init__(self, cells: int, groups: int, K: int, V: int):
        self.cells, self.groups, self.K, self.V = cells, groups, K, V
        size = cells * groups
        self.n = torch.zeros(size, dtype=torch.float64)
        self.s1 = torch.zeros(size, V, dtype=torch.float64)
        self.s2 = torch.zeros(size, V, dtype=torch.float64)
        self.zs = torch.zeros(size, 4, dtype=torch.float64)
        self.zz = torch.zeros(size, 4, 4, dtype=torch.float64)

    def add(self, logits: torch.Tensor, cell: torch.Tensor, group: torch.Tensor):
        keep = cell >= 0
        x, index = logits[keep].double().cpu(), (cell[keep] * self.groups + group[keep]).cpu()
        K, V = self.K, self.V
        z = torch.stack([x[:, -1], x[:, :K].sum(1), x[:, K:2 * K - 1].sum(1), x[:, 2 * K - 1:V - 1].sum(1)], 1)
        self.n.index_add_(0, index, torch.ones(len(x), dtype=torch.float64))
        self.s1.index_add_(0, index, x)
        self.s2.index_add_(0, index, x ** 2)
        self.zs.index_add_(0, index, z)
        self.zz.index_add_(0, index, z[:, :, None] * z[:, None, :])

    def _stats(self):
        shape = (self.cells, self.groups)
        return (self.n.view(shape), self.s1.view(*shape, -1), self.s2.view(*shape, -1), self.zs.view(*shape, -1),
                self.zz.view(*shape, 4, 4))

    def pooled(self) -> tuple[dict, dict, torch.Tensor]:
        """{moment: (cells,)} over all groups, its jackknife s.e. over the groups, and the
        counts (cells,)."""
        stats = self._stats()
        total = [s.sum(1) for s in stats]
        estimate = class_moments(*total, self.K, self.V)
        left_out = class_moments(*[t[:, None] - s for t, s in zip(total, stats)], self.K, self.V)
        used = stats[0] > 0
        groups = used.sum(1).clamp(min=2).double()
        se = {}
        for name, values in left_out.items():
            values = torch.where(used, values, estimate[name][:, None])
            mean = values.sum(1) / groups
            spread = torch.where(used, (values - mean[:, None]) ** 2, torch.zeros_like(values)).sum(1)
            se[name] = ((groups - 1) / groups * spread).sqrt()
        return estimate, se, total[0]

    def per_group(self, n_min: int) -> tuple[dict, dict]:
        """The mean and standard deviation over the groups of the per-group moments, using
        the groups with at least n_min rows in the cell (the quenched scatter)."""
        stats = self._stats()
        values = class_moments(*stats, self.K, self.V)
        used = stats[0] >= n_min
        count = used.sum(1).double()
        mean, sd = {}, {}
        for name, value in values.items():
            value = torch.where(used, value, torch.zeros_like(value))
            mean[name] = value.sum(1) / count
            sd[name] = (torch.where(used, (value - mean[name][:, None]) ** 2, torch.zeros_like(value)).sum(1)
                        / (count - 1)).sqrt()
        return mean, sd


def class_moments(n, s1, s2, zs, zz, K: int, V: int) -> dict:
    """The class moments from the sufficient statistics (any leading shape), with unbiased
    (n - 1) covariances."""
    n = n.clamp(min=2)
    mean = s1 / n[..., None]
    var = (s2 - s1 ** 2 / n[..., None]) / (n[..., None] - 1)
    cov = (zz - zs[..., :, None] * zs[..., None, :] / n[..., None, None]) / (n[..., None, None] - 1)
    T, out, plain = slice(0, K), slice(K, 2 * K - 1), slice(2 * K - 1, V - 1)
    sizes = {"out": K - 1, "plain": V - 2 * K}
    columns = {"out": out, "plain": plain}
    z = {"out": 2, "plain": 3}
    moments = {"on": mean[..., -1], "T": mean[..., T].mean(-1), "on,on": var[..., -1], "T,T": var[..., T].mean(-1),
               "T,T'": (cov[..., 1, 1] - var[..., T].sum(-1)) / (K * (K - 1)), "on,T": cov[..., 0, 1] / K}
    for c in CLASSES:
        moments[f"b|{c}"] = mean[..., columns[c]].mean(-1)
        moments[f"b,b|{c}"] = var[..., columns[c]].mean(-1)
        moments[f"on,b|{c}"] = cov[..., 0, z[c]] / sizes[c]
        moments[f"b,T|{c}"] = cov[..., z[c], 1] / (sizes[c] * K)
        moments[f"b,b'|{c},{c}"] = ((cov[..., z[c], z[c]] - var[..., columns[c]].sum(-1))
                                    / (sizes[c] * (sizes[c] - 1)))
    moments["b,b'|out,plain"] = cov[..., 2, 3] / (sizes["out"] * sizes["plain"])
    return {name: moments[name] for name in MOMENTS}


class Grid:
    """The cells (mu, ell), mu in `mus`, ell = 0..ELL_MAX, flattened mu-major."""

    def __init__(self, mus):
        self.mus = torch.as_tensor(mus)
        self.mu = self.mus.repeat_interleave(ELL_MAX + 1)
        self.ell = torch.arange(ELL_MAX + 1).repeat(len(self.mus))

    def __len__(self):
        return len(self.mu)

    def cell(self, mu: torch.Tensor, ell: torch.Tensor) -> torch.Tensor:
        """The cell of each row, -1 if ell > ELL_MAX or mu not in the grid."""
        mu, ell = mu.cpu(), ell.cpu()
        where = (mu[:, None] == self.mus[None, :]).long()
        index = where.argmax(1)
        ok = (where.sum(1) > 0) & (ell <= ELL_MAX)
        return torch.where(ok, index * (ELL_MAX + 1) + ell.clamp(max=ELL_MAX), torch.full_like(index, -1))


def trained_moments(run, step, grid, K, V, sequences, device, chunk=4096):
    """(i): the trained network; groups are blocks of sequences (s.e.) and query tokens."""
    model = run.model(step, device=device, dtype=torch.float64)
    blocks = max(sequences // chunk, 2)
    by_block, by_token = Accumulator(len(grid), blocks, K, V), Accumulator(len(grid), K, K, V)
    for block in range(blocks):
        batch = run.batch(chunk, seed=10_000 + block, device=device)
        table = logit_table(model, batch, mus=grid.mus.tolist(), chunk=chunk)
        cell = grid.cell(table["mu"], table["ell"])
        query = batch["sequence"][:, :-1].cpu()[table["sequence_index"], table["mu"] - 1]
        by_block.add(table["logits"], cell, torch.full_like(cell, block))
        by_token.add(table["logits"], cell, query)
    return by_block, by_token


def ansatz_moments(run, op, grid, K, V, L, networks, sequences, device, chunk=4096):
    """(ii): `networks` draws of the variance ansatz, each on its own sequences."""
    accumulator = Accumulator(len(grid), networks, K, V)
    for network in range(networks):
        matrices = ansatz_matrices(op, L, V, K, noise=torch.Generator().manual_seed(network))
        model = ReducedTransformer.from_matrices(matrices, run.config.model_args, device=device, dtype=torch.float64)
        for start in range(0, sequences, chunk):
            batch = run.batch(min(chunk, sequences - start), seed=100_000 * (network + 1) + start, device=device)
            table = logit_table(model, batch, mus=grid.mus.tolist(), chunk=chunk)
            cell = grid.cell(table["mu"], table["ell"])
            accumulator.add(table["logits"], cell, torch.full_like(cell, network))
    return accumulator


# ---- theory: term values per class moment ---------------------------------------------------

def term_moments(table, term) -> dict:
    """{moment: (rows,)} of one term (class means of its b-indexed values)."""
    rows = table.rows.num_rows
    value = term.value
    pair = term.pair
    if isinstance(value, (Bilinear, ClassPairSums)):
        mean = table.class_mean(value, pair)                      # (rows, 2, 2)
        return {"b,b'|out,out": mean[:, 0, 0], "b,b'|plain,plain": mean[:, 1, 1],
                "b,b'|out,plain": 0.5 * (mean[:, 0, 1] + mean[:, 1, 0])}
    value = torch.as_tensor(value, dtype=torch.float64)
    if pair in ("b", "b,b", "on,b", "b,T"):
        if value.dim() == 2:
            mean = table.class_mean(value, pair)
            columns = (mean[:, 0], mean[:, 1])
        else:
            columns = (value.expand(rows),) * 2
        return {f"{pair}|{c}": column for c, column in zip(CLASSES, columns)}
    return {pair: value.expand(rows)}


def theory_ell(op, grid, V, K, L, beta, device):
    """(iii) at ell: {term name: {moment: (cells,)}} and the terms' tags."""
    table = trigger_terms_ell(grid.mu, grid.ell, op, beta, L, V, K, device=device)
    values = {term.name: {moment: value.detach().cpu() for moment, value in term_moments(table, term).items()}
              for term in table}
    tags = {term.name: term.tags() for term in table}
    return values, tags


class RowMeans:
    """Per-cell means and s.e. of per-row values (rows of one cell are independent)."""

    def __init__(self, cells: int):
        self.cells = cells
        self.sums = {}

    def add(self, key, values: torch.Tensor, cell: torch.Tensor):
        keep = cell >= 0
        values, cell = values[keep].double().cpu(), cell[keep]
        if key not in self.sums:
            self.sums[key] = torch.zeros(3, self.cells, dtype=torch.float64)
        s = self.sums[key]
        s[0].index_add_(0, cell, torch.ones_like(values))
        s[1].index_add_(0, cell, values)
        s[2].index_add_(0, cell, values ** 2)

    def get(self, key) -> tuple[torch.Tensor, torch.Tensor]:
        n, s1, s2 = self.sums[key]
        n = n.clamp(min=2)
        mean = s1 / n
        return mean, ((s2 / n - mean ** 2).clamp(min=0) / (n - 1)).sqrt()


def theory_S(run, op, grid, V, K, L, beta, sequences, device, chunk=4096):
    """The given-S level averaged per cell: per-term means of the mean and noise terms, the
    noise totals, and the covariance of the given-S means over the rows (the spread)."""
    blocks = max(sequences // chunk, 2)
    spread = Accumulator(len(grid), blocks, K, V)
    rows = RowMeans(len(grid))
    for block in range(blocks):
        batch = run.batch(chunk, seed=20_000 + block, device=device)
        variables = measure_variables(batch, grid.mus.tolist())
        table = trigger_terms(variables, op, beta, L)
        cell = grid.cell(variables["mu"], variables["ell"])
        means = torch.cat([table.total("T")[:, None].expand(-1, K), table.total("b"), table.total("on")[:, None]], 1)
        spread.add(means.detach(), cell, torch.full_like(cell, block))
        noise = {}
        for term in table:
            for moment, value in term_moments(table, term).items():
                value = value.detach()
                rows.add(("term", term.name, moment), value, cell)
                if term.channel != "mean":
                    noise[moment] = noise.get(moment, 0) + value
        for moment, value in noise.items():
            rows.add(("noise", moment), value, cell)
    return spread, rows


# ---- hierarchy ------------------------------------------------------------------------------

def hierarchy(theory: dict, tags: dict, ansatz_se: dict, valid: dict, width: int, empirical: dict | None = None):
    """Shares, the individual drop test, the ranking and the greedy pruning curves.

    theory: {step: {term: {moment: (cells,)}}}; ansatz_se, valid: {step: {moment: (cells,)}};
    empirical: {"ansatz" | "trained": ({step: {moment: value}}, {step: {moment: s.e.}})}, for
    the rms z-score of the pruned theory against them along the curves.
    The cells of the tolerance are (step, moment, mu, ell) with valid ansatz counts."""
    steps = list(theory)
    names = list(tags)
    cells = [(step, moment) for step in steps for moment in MOMENTS]
    D = torch.zeros(len(names), len(cells), width, dtype=torch.float64)
    for t, name in enumerate(names):
        for c, (step, moment) in enumerate(cells):
            if moment in theory[step][name]:
                D[t, c] = theory[step][name][moment]
    full = D.sum(0)
    se = torch.stack([ansatz_se[step][moment] for step, moment in cells])
    mask = torch.stack([valid[step][moment] for step, moment in cells])
    absolute = D.abs().sum(0)
    share = D.abs() / absolute.clamp(min=1e-300)
    allowed = torch.maximum(TOL_REL * full.abs(), TOL_SE * se)

    def passes(change, where):
        return ((change.abs() <= allowed) | ~where).all().item()

    targets = {kind: (torch.stack([values[step][moment] for step, moment in cells]),
                      torch.stack([errors[step][moment] for step, moment in cells]))
               for kind, (values, errors) in (empirical or {}).items()}

    def rms_z(change, where) -> dict:
        out = {}
        for kind, (value, error) in targets.items():
            z = ((full - change - value) / error.clamp(min=1e-300))[where]
            out[f"rms_z_vs_{kind}"] = z.pow(2).mean().sqrt().item() if where.any() else math.nan
        return out

    step_of = torch.tensor([steps.index(step) for step, _ in cells])
    rows = []
    for t, name in enumerate(names):
        entry = {"term": name, **tags[name]}
        touched = (D[t] != 0) & mask
        for scope, scope_mask in [("all", mask)] + [(str(step), mask & (step_of == i)[:, None])
                                                    for i, step in enumerate(steps)]:
            where = scope_mask & touched
            entry[f"max_share/{scope}"] = share[t][where].max().item() if where.any() else 0.0
            rel = (D[t].abs() / full.abs().clamp(min=1e-300))[where]
            over_se = (D[t].abs() / se.clamp(min=1e-300))[where]
            entry[f"max_rel_change/{scope}"] = rel.max().item() if where.any() else 0.0
            entry[f"max_change_over_se/{scope}"] = over_se.max().item() if where.any() else 0.0
            entry[f"prunable/{scope}"] = passes(D[t], scope_mask)
        rows.append(entry)
    ranking = pd.DataFrame(rows)

    curves = []
    for scope, scope_mask in [("all", mask)] + [(str(step), mask & (step_of == i)[:, None])
                                                for i, step in enumerate(steps)]:
        order = {group: list(ranking[ranking["channel"].isin(channels)].sort_values(f"max_share/{scope}")["term"])
                 for group, channels in GROUPS.items()}
        # each group alone (the others in full), then all together: noise, spread, means, one dropped budget
        order["combined"] = order["noise"] + order["spread"] + order["means"]
        for group, members in order.items():
            dropped = torch.zeros_like(full)
            cumulative = torch.zeros_like(full)
            for k, name in enumerate(members):
                t = names.index(name)
                cumulative = cumulative + D[t]
                trial = dropped + D[t]
                accepted = passes(trial, scope_mask)
                if accepted:
                    dropped = trial
                where = scope_mask & (cumulative != 0)
                curves.append({
                    "scope": scope, "group": group, "rank": k, "term": name, "channel": tags[name]["channel"],
                    "max_share": ranking.loc[ranking["term"] == name, f"max_share/{scope}"].item(),
                    "accepted_greedy": accepted,
                    "in_order/max_rel_change": ((cumulative.abs() / full.abs().clamp(min=1e-300))[where].max().item()
                                                if where.any() else 0.0),
                    "in_order/max_change_over_se": ((cumulative.abs() / se.clamp(min=1e-300))[where].max().item()
                                                    if where.any() else 0.0),
                    "in_order/within_tolerance": passes(cumulative, scope_mask),
                    **{f"in_order/{key}": value for key, value in rms_z(cumulative, scope_mask).items()},
                    **{f"greedy/{key}": value for key, value in rms_z(dropped, scope_mask).items()}})
    return ranking, pd.DataFrame(curves), D, names, cells


# ---- main -----------------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, nargs="+", default=list(STEPS))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--networks", type=int, default=128)
    parser.add_argument("--sequences", type=int, default=4096, help="sequences per ansatz network")
    parser.add_argument("--trained-sequences", type=int, default=262144)
    parser.add_argument("--s-sequences", type=int, default=32768)
    parser.add_argument("--chunk", type=int, default=4096)
    parser.add_argument("--from-csv", action="store_true", help="only redo the ranking and pruning from the CSVs")
    args = parser.parse_args()
    if args.from_csv:
        return analyse()

    run = RunData(*RUN, base_dir="data")
    model_args = run.config.model_args
    V, L, K, beta = model_args.vocab_size, model_args.seq_len, run.config.data_args.K, model_args.beta
    grid = Grid([math.ceil(f * L) for f in FRACTIONS])
    OUT.mkdir(parents=True, exist_ok=True)

    moment_rows, term_rows, observation_rows = [], [], []
    theory, tags, ansatz_se, valid, summary = {}, {}, {}, {}, {"settings": vars(args), "snapshots": {}}
    for step in args.steps:
        t0 = time.time()
        op, smoothing = snapshot_params(run, step)
        op_device = to(op, args.device)
        by_block, by_token = trained_moments(run, step, grid, K, V, args.trained_sequences, args.device, args.chunk)
        trained, trained_se, trained_n = by_block.pooled()
        token_mean, token_sd = by_token.per_group(n_min=20)
        ansatz = ansatz_moments(run, op, grid, K, V, L, args.networks, args.sequences, args.device, args.chunk)
        annealed, annealed_se, annealed_n = ansatz.pooled()
        values, tags = theory_ell(op_device, grid, V, K, L, beta, args.device)
        spread_S, rows_S = theory_S(run, op_device, grid, V, K, L, beta, args.s_sequences, args.device, args.chunk)
        spread_S, spread_S_se, _ = spread_S.pooled()
        theory[step] = values
        ansatz_se[step] = annealed_se
        valid[step] = {moment: annealed_n >= N_MIN for moment in MOMENTS}
        summary["snapshots"][step] = {"order_params": {k: (float(v) if not torch.is_tensor(v) else None)
                                                       for k, v in op.items()},
                                      "profile_smoothing": smoothing, "seconds": None}

        totals = {moment: sum(v[moment] for v in values.values() if moment in v) for moment in MOMENTS}
        by_channel = {}
        for name, v in values.items():
            for moment, value in v.items():
                key = (moment, tags[name]["channel"])
                by_channel[key] = by_channel.get(key, 0) + value
        for c in range(len(grid)):
            mu, ell = grid.mu[c].item(), grid.ell[c].item()
            for moment in MOMENTS:
                noise_S, noise_S_se = (rows_S.get(("noise", moment)) if ("noise", moment) in rows_S.sums
                                       else (torch.zeros(len(grid)),) * 2)
                mean_terms = [name for name in values if tags[name]["channel"] == "mean" and moment in values[name]]
                if moment in MEANS:
                    s_value = sum(rows_S.get(("term", name, moment))[0][c].item() for name in mean_terms)
                    s_se = math.sqrt(sum(rows_S.get(("term", name, moment))[1][c].item() ** 2 for name in mean_terms))
                else:
                    s_value = noise_S[c].item() + spread_S[moment][c].item()
                    s_se = math.hypot(noise_S_se[c].item(), spread_S_se[moment][c].item())
                moment_rows.append({
                    "step": step, "mu": mu, "ell": ell, "moment": moment,
                    "n_trained": int(trained_n[c]), "trained": trained[moment][c].item(),
                    "trained_se": trained_se[moment][c].item(),
                    "trained_token_mean": token_mean[moment][c].item(), "trained_token_sd": token_sd[moment][c].item(),
                    "n_ansatz": int(annealed_n[c]), "ansatz": annealed[moment][c].item(),
                    "ansatz_se": annealed_se[moment][c].item(),
                    "theory_S": s_value, "theory_S_se": s_se,
                    "theory_S_noise": noise_S[c].item() if moment in COVARIANCES else 0.0,
                    "theory_S_spread": spread_S[moment][c].item() if moment in COVARIANCES else 0.0,
                    "theory_ell": totals[moment][c].item(),
                    **{f"theory_ell_{channel}": (by_channel[(moment, channel)][c].item()
                                                 if (moment, channel) in by_channel else 0.0)
                       for channel in ("mean", "spread", "attention", "readout")}})
            absolute = {moment: sum(v[moment].abs() for v in values.values() if moment in v) for moment in MOMENTS}
            for name, v in values.items():
                for moment, value in v.items():
                    key = ("term", name, moment)
                    S_mean = rows_S.get(key) if key in rows_S.sums else (None, None)
                    term_rows.append({
                        "step": step, "mu": mu, "ell": ell, "moment": moment, "term": name, **tags[name],
                        "value": value[c].item(), "share": (value[c].abs() / absolute[moment][c]).item()
                        if absolute[moment][c] > 0 else 0.0,
                        "signed": (value[c] / totals[moment][c]).item() if totals[moment][c] != 0 else math.nan,
                        "S_average": S_mean[0][c].item() if S_mean[0] is not None else math.nan,
                        "S_average_se": S_mean[1][c].item() if S_mean[1] is not None else math.nan})
            # the two observations: the signal term of E[h^on | ell]; noise vs spread per moment
            signal = values["on/mean/signal"]["on"][c].item()
            observation = {"step": step, "mu": mu, "ell": ell, "n_ansatz": int(annealed_n[c]),
                           "signal": signal, "signal_share_theory": ratio(signal, totals["on"][c].item()),
                           "signal_share_ansatz": ratio(signal, annealed["on"][c].item()),
                           "signal_share_trained": ratio(signal, trained["on"][c].item())}
            for moment in COVARIANCES:
                parts = {channel: (by_channel[(moment, channel)][c].item() if (moment, channel) in by_channel else 0.0)
                         for channel in ("spread", "attention", "readout")}
                total = totals[moment][c].item()
                observation |= {f"{moment}/spread_share": parts["spread"] / total if total else math.nan,
                                f"{moment}/attention_share": parts["attention"] / total if total else math.nan,
                                f"{moment}/readout_share": parts["readout"] / total if total else math.nan}
                if moment in COVARIANCES:
                    noise_S, _ = rows_S.get(("noise", moment))
                    total_S = noise_S[c].item() + spread_S[moment][c].item()
                    observation[f"{moment}/spread_share_S"] = spread_S[moment][c].item() / total_S if total_S else math.nan
            observation_rows.append(observation)
        summary["snapshots"][step]["seconds"] = time.time() - t0
        print(f"step {step}: {time.time() - t0:.0f} s (profile width {smoothing['width']})", flush=True)

    pd.DataFrame(moment_rows).to_csv(OUT / "phase4_moments.csv", index=False)
    pd.DataFrame(term_rows).to_csv(OUT / "phase4_terms.csv", index=False)
    pd.DataFrame(observation_rows).to_csv(OUT / "phase4_observations.csv", index=False)
    (OUT / "phase4_summary.json").write_text(json.dumps(summary, indent=1, default=str))
    analyse()


TAGS = ("pair", "channel", "column", "mechanism", "keyset", "detail", "label")


def analyse():
    """The ranking and the pruning curves from phase4_moments.csv and phase4_terms.csv."""
    moments = pd.read_csv(OUT / "phase4_moments.csv")
    terms = pd.read_csv(OUT / "phase4_terms.csv", keep_default_na=False, na_values={"value": [""]})
    grid = Grid(sorted(moments["mu"].unique()))
    cell = {(mu, ell): c for c, (mu, ell) in enumerate(zip(grid.mu.tolist(), grid.ell.tolist()))}
    steps = sorted(moments["step"].unique())
    tags = {row.term: {tag: getattr(row, tag) for tag in TAGS}
            for row in terms.drop_duplicates("term").itertuples()}
    theory = {step: {} for step in steps}
    for (step, name, moment), group in terms.groupby(["step", "term", "moment"]):
        values = torch.zeros(len(grid), dtype=torch.float64)
        values[[cell[key] for key in zip(group["mu"], group["ell"])]] = torch.tensor(group["value"].to_numpy(float))
        theory[step].setdefault(name, {})[moment] = values
    columns = {}
    for (step, moment), group in moments.groupby(["step", "moment"]):
        index = [cell[key] for key in zip(group["mu"], group["ell"])]
        for column in ("ansatz", "ansatz_se", "n_ansatz", "trained", "trained_se"):
            values = torch.zeros(len(grid), dtype=torch.float64)
            values[index] = torch.tensor(group[column].to_numpy(float))
            columns.setdefault(column, {}).setdefault(step, {})[moment] = values
    valid = {step: {moment: n >= N_MIN for moment, n in by_moment.items()} for step, by_moment in columns["n_ansatz"].items()}
    empirical = {"ansatz": (columns["ansatz"], columns["ansatz_se"]),
                 "trained": (columns["trained"], columns["trained_se"])}
    ranking, curves, _, _, _ = hierarchy(theory, tags, columns["ansatz_se"], valid, len(grid), empirical)
    ranking.to_csv(OUT / "phase4_ranking.csv", index=False)
    curves.to_csv(OUT / "phase4_pruning.csv", index=False)
    summary = json.loads((OUT / "phase4_summary.json").read_text())
    summary["tolerance"] = {"rel": TOL_REL, "se": TOL_SE, "n_min": N_MIN}
    summary["prunable"] = {scope: {group: sorted(curves[(curves["scope"] == scope) & (curves["group"] == group)
                                                       & curves["accepted_greedy"]]["term"])
                                   for group in (*GROUPS, "combined")}
                           for scope in curves["scope"].unique()}
    (OUT / "phase4_summary.json").write_text(json.dumps(summary, indent=1, default=str))
    sizes = {group: int(ranking["channel"].isin(GROUPS[group]).sum()) for group in GROUPS}
    sizes["combined"] = len(ranking)
    for scope, groups in summary["prunable"].items():
        print(scope, {group: f"{len(names)}/{sizes[group]}" for group, names in groups.items()})
    print(f"wrote {OUT}/phase4_*.csv")


if __name__ == "__main__":
    main()
