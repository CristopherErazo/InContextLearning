"""Phase 6 of archive/scratch/2026-10-08-1103_plan-pruned-effective-model.md: nested presets of
the terms at a trigger query, and their validation.

The presets are sets of dropped term names, each containing the previous one:

    counting        every term whose power-counting estimated share (Phase 5) is < 3e-3 at
                    every snapshot from the middle of the transition on (1448, 1552, 3000)
    pruned          + greedy drops within the tolerance at all five Phase 4 snapshots
    pruned_late     + greedy drops within the tolerance after the transition (1552, 3000)
    compact         + greedy drops within a 5x looser tolerance (10% or one s.e.) at 1552, 3000
    minimal         + greedy drops within a 12.5x looser tolerance (25%, 2.5 s.e.) at 1552, 3000

(a preset within the tolerance from 1448 on adds nothing to "pruned": its budget is used up at
1448 and 3000 already).

The tolerance is Phase 4's: in every cell (snapshot, mu, ell) with >= 400 ansatz rows every
class moment changes by < 2% of its full value or < 0.2 ansatz s.e. The greedy order is Phase
4's: noise, then spread, then means, each by ascending largest share over the scope's cells.

Validation, from the Phase 4 tables (no new networks): per preset and snapshot, the largest
change in units of the allowance, the cells that break it, the rms z-score against the ansatz
networks (ii) and the trained network (i) per group of moments; the power counting of the
dropped terms (largest estimated share, local exponent, Phase 5). For information only, the
effective loss of each preset at run_001's snapshots (EffectiveLoss, methods "mean" and "mc").

    python -m scripts.presets             # build, validate, write data/hierarchy/phase6_*
    python -m scripts.presets --write     # ... and src/icl/theory/terms/presets.py
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from icl import EffectiveLoss, RunData
from scripts.term_hierarchy import N_MIN, OUT, RUN, STEPS, TOL_REL, TOL_SE, snapshot_params

CELL = ["step", "mu", "ell", "moment"]
GROUP_ORDER = {"attention": 0, "readout": 0, "spread": 1, "mean": 2}
COUNTING_THRESHOLD = 3e-3
LATE = (1552, 3000)
CHAIN = [("counting", None, 1.0), ("pruned", STEPS, 1.0), ("pruned_late", LATE, 1.0), ("compact", LATE, 5.0),
         ("minimal", LATE, 12.5)]                   # (name, snapshots of the tolerance, its factor)
MOMENT_GROUPS = {"means on, T": ("on", "T"), "means b": ("b|out", "b|plain"),
                 "Var on, T; Cov T,T'": ("on,on", "T,T", "T,T'"), "Cov on,T": ("on,T",),
                 "Var b": ("b,b|out", "b,b|plain"),
                 "Cov on,b; b,T": ("on,b|out", "on,b|plain", "b,T|out", "b,T|plain"),
                 "Cov b,b'": ("b,b'|out,out", "b,b'|out,plain", "b,b'|plain,plain")}
PRESETS_FILE = Path("src/icl/theory/terms/presets.py")


def load() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    terms = pd.read_csv(OUT / "phase4_terms.csv", keep_default_na=False, na_values={"value": [""]})
    moments = pd.read_csv(OUT / "phase4_moments.csv").set_index(CELL)
    moments["full"] = terms.groupby(CELL)["value"].sum()
    moments["allowed"] = np.maximum(TOL_REL * moments["full"].abs(), TOL_SE * moments["ansatz_se"])
    moments["valid"] = moments["n_ansatz"] >= N_MIN
    D = terms.pivot_table(index=CELL, columns="term", values="value", aggfunc="sum").reindex(moments.index).fillna(0.0)
    return terms, moments, D


def change_ratio(dropped, moments, D, steps=None) -> pd.Series:
    """|change| / allowance of every valid cell (of `steps`) when `dropped` go."""
    change = D[sorted(dropped)].sum(axis=1) if dropped else pd.Series(0.0, index=D.index)
    ratio = (change.abs() / moments["allowed"])[moments["valid"]]
    if steps is not None:
        ratio = ratio[ratio.index.get_level_values("step").isin(steps)]
    return ratio


def greedy(start: set, terms: pd.DataFrame, moments, D, steps, loose: float) -> set:
    tags = terms.drop_duplicates("term").set_index("term")
    in_scope = terms[terms["step"].isin(steps)]
    share = in_scope.groupby("term")["share"].max()
    candidates = sorted((name for name in tags.index if name not in start),
                        key=lambda name: (GROUP_ORDER[tags.loc[name, "channel"]], share.get(name, 0.0)))
    dropped = set(start)
    base = D[sorted(dropped)].sum(axis=1) if dropped else pd.Series(0.0, index=D.index)
    mask = moments["valid"] & moments.index.get_level_values("step").isin(steps)
    allowed = loose * moments["allowed"]
    for name in candidates:
        trial = base + D[name]
        if ((trial.abs() <= allowed) | ~mask).all():
            dropped.add(name)
            base = trial
    return dropped


def build_chain(terms, moments, D) -> dict:
    estimated = pd.read_csv(OUT / "phase5_estimated_shares.csv")
    estimated = estimated[estimated["scope"] != "all"].pivot(index="term", columns="scope", values="estimated_share")
    counting = set(estimated.index[(estimated < COUNTING_THRESHOLD).all(axis=1)])
    chain, dropped = {}, set()
    for name, steps, loose in CHAIN:
        dropped = counting if steps is None else greedy(dropped, terms, moments, D, steps, loose)
        chain[name] = set(dropped)
    return chain


# ---- validation -----------------------------------------------------------------------------

def validate(chain: dict, terms, moments, D) -> pd.DataFrame:
    rows = []
    for preset, dropped in {"full": set(), **chain}.items():
        change = D[sorted(dropped)].sum(axis=1) if dropped else pd.Series(0.0, index=D.index)
        pruned = moments["full"] - change
        ratio = (change.abs() / moments["allowed"])
        for step in STEPS:
            for group, members in {"all": None, **MOMENT_GROUPS}.items():
                where = moments["valid"] & (moments.index.get_level_values("step") == step)
                if members is not None:
                    where &= moments.index.get_level_values("moment").isin(members)
                z_ansatz = ((pruned - moments["ansatz"]) / moments["ansatz_se"])[where]
                z_trained = ((pruned - moments["trained"]) / moments["trained_se"])[where]
                rows.append({"preset": preset, "step": step, "moments": group, "cells": int(where.sum()),
                             "max_change_over_allowed": float(ratio[where].max()),
                             "broken_cells": int((ratio[where] > 1).sum()),
                             "rms_z_ansatz": float(np.sqrt((z_ansatz ** 2).mean())),
                             "rms_z_trained": float(np.sqrt((z_trained ** 2).mean()))})
    return pd.DataFrame(rows)


def composition(chain: dict, terms) -> pd.DataFrame:
    tags = terms.drop_duplicates("term").set_index("term")
    tags["group"] = tags["channel"].where(tags["channel"] != "readout", "readout " + tags["column"])
    phase5 = pd.read_csv(OUT / "phase5_terms.csv").set_index("term")
    estimated = pd.read_csv(OUT / "phase5_estimated_shares.csv").query("scope == 'all'").set_index("term")
    rows = []
    for preset, dropped in chain.items():
        kept = set(tags.index) - dropped
        row = {"preset": preset, "dropped": len(dropped), "kept": len(kept)}
        for group, members in tags.groupby("group").groups.items():
            row[f"kept/{group}"] = f"{len(kept & set(members))}/{len(members)}"
        dropped_list = sorted(dropped)
        row["max_estimated_share_dropped"] = float(estimated.loc[dropped_list, "estimated_share"].max())
        row["median_local_exponent_dropped"] = float(
            phase5.loc[dropped_list, "robust/relative_local_exponent"].median())
        row["median_local_exponent_kept"] = float(phase5.loc[sorted(kept), "robust/relative_local_exponent"].median())
        rows.append(row)
    return pd.DataFrame(rows)


def losses(chain: dict, device) -> pd.DataFrame:
    """For information: the effective loss of each preset at run_001's snapshots."""
    from icl.theory.terms import PRESETS
    run = RunData(*RUN, base_dir="data")
    rows = []
    for step in STEPS:
        op, _ = snapshot_params(run, step)
        for method in ("mean", "mc"):
            for preset in ("full", *chain):
                loss = EffectiveLoss.from_config(run.config, method=method, keep=None if preset == "full" else preset,
                                                 num_samples=64, device=device)
                parts = loss.breakdown({k: (v.to(device) if torch.is_tensor(v) else v) for k, v in op.items()})
                rows.append({"step": step, "method": method, "preset": preset,
                             "run_loss": float(run.metrics["loss"].dropna().asof(step)),
                             **{k: float(v) for k, v in parts.items() if k != "clusters"}})
    return pd.DataFrame(rows)


def write_module(chain: dict) -> None:
    lines = ['"""The term sets of the pruned presets (generated by scripts/presets.py, Phase 6 of',
             'archive/scratch/2026-10-08-1103_plan-pruned-effective-model.md; do not edit by hand).',
             '',
             'ADDED[preset]: the trigger-query terms a preset drops on top of the previous one;',
             'DROPPED[preset]: all the terms it drops (nested: counting < pruned < pruned_late <',
             'compact < minimal). Validated at run_001 (full_ansatz/run_001, V=128, L=512, K=25,',
             'beta=0.25) against the trained network and 128 ansatz networks, see the report',
             'archive/scratch/2026-10-08-1119_pruned-effective-model.tex."""',
             '', 'ADDED = {']
    previous = set()
    for preset, dropped in chain.items():
        lines.append(f'    "{preset}": (')
        lines += [f'        "{name}",' for name in sorted(dropped - previous)]
        lines.append('    ),')
        previous = dropped
    lines += ['}', '', 'DROPPED, _dropped = {}, frozenset()', 'for _preset, _names in ADDED.items():',
              '    _dropped = DROPPED[_preset] = _dropped | frozenset(_names)']
    PRESETS_FILE.write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true", help=f"write {PRESETS_FILE}")
    parser.add_argument("--no-loss", action="store_true", help="skip the effective losses")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    terms, moments, D = load()
    chain = build_chain(terms, moments, D)
    if args.write:
        write_module(chain)
    else:
        from icl.theory.terms.presets import DROPPED
        if {k: set(v) for k, v in DROPPED.items()} != chain:
            print(f"warning: {PRESETS_FILE} differs from the chain built now (run with --write)")
    for preset, dropped in chain.items():
        print(f"{preset}: {len(dropped)} dropped")
    validation = validate(chain, terms, moments, D)
    validation.to_csv(OUT / "phase6_validation.csv", index=False)
    composition(chain, terms).to_csv(OUT / "phase6_presets.csv", index=False)
    summary = {"chain": {name: {"steps": steps, "tolerance_factor": loose} for name, steps, loose in CHAIN},
               "counting_threshold": COUNTING_THRESHOLD,
               "dropped": {name: sorted(dropped) for name, dropped in chain.items()}}
    if not args.no_loss:
        loss = losses(chain, args.device)
        loss.to_csv(OUT / "phase6_loss.csv", index=False)
    (OUT / "phase6_summary.json").write_text(json.dumps(summary, indent=1))
    print(validation[validation["moments"] == "all"].pivot(index="preset", columns="step",
                                                            values="max_change_over_allowed").round(2))


if __name__ == "__main__":
    main()
