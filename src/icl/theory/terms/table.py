"""Terms and term tables: the container shared by every level of `icl.theory.terms`.

A `Term` is one named summand of a boxed equation of the appendix, with its tags and its
value (one per row: a query, or a (mu, ell) cluster). A `TermTable` holds all the terms of
one level at one kind of query; `total(pair, keep)` sums them into a class moment and
`keep` prunes them (a set of names, a predicate on the term, or a preset of PRESETS).
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Callable, Iterable

import torch

from ..query_table import QueryTable
from .presets import DROPPED


class Bilinear:
    """A (rows, nb, nb) matrix over pairs of distinct non-targets b != b', stored as
    sum_k coef_k[row] u_k[row, b] v_k[row, b'] (every b-b' covariance of the appendix is
    a sum of such outer products). The diagonal b = b' is excluded."""

    def __init__(self, parts=()):
        self.parts = tuple(parts)

    def __add__(self, other):
        if isinstance(other, (int, float)) and other == 0:
            return self
        return Bilinear(self.parts + other.parts)

    __radd__ = __add__

    def __mul__(self, factor):
        return Bilinear(tuple((factor * coef, u, v) for coef, u, v in self.parts))

    __rmul__ = __mul__

    def __neg__(self):
        return self * -1.0

    @staticmethod
    def _column(coef, u):
        coef = torch.as_tensor(coef, dtype=u.dtype, device=u.device)
        return coef[:, None] if coef.dim() == 1 else coef

    def dense(self) -> torch.Tensor:
        """(rows, nb, nb), zero on the diagonal."""
        total = 0
        for coef, u, v in self.parts:
            total = total + self._column(coef, u)[..., None] * u[:, :, None] * v[:, None, :]
        if not self.parts:
            raise ValueError("an empty Bilinear has no shape")
        return total * (1 - torch.eye(total.size(-1), dtype=total.dtype, device=total.device))

    def class_sums(self, classes: torch.Tensor) -> torch.Tensor:
        """(rows, k, k): the sum over b in class i, b' in class j, b != b', for `classes`
        a (k, nb) boolean mask."""
        masks = classes.to(dtype=self.parts[0][1].dtype) if self.parts else None
        total = 0
        for coef, u, v in self.parts:
            coef = torch.as_tensor(coef, dtype=u.dtype, device=u.device)
            coef = coef[:, None, None] if coef.dim() == 1 else coef
            outer = (u @ masks.T)[:, :, None] * (v @ masks.T)[:, None, :]
            diagonal = torch.einsum("rb,ib,jb->rij", u * v, masks, masks)
            total = total + coef * (outer - diagonal)
        return total


class ClassPairSums:
    """A b-b' covariance known only through its class sums (rows, k, k) (the exact level)."""

    def __init__(self, sums: torch.Tensor):
        self.sums = sums

    def __add__(self, other):
        if isinstance(other, (int, float)) and other == 0:
            return self
        return ClassPairSums(self.sums + other.sums)

    __radd__ = __add__

    def __mul__(self, factor):
        return ClassPairSums(factor * self.sums)

    __rmul__ = __mul__

    def __neg__(self):
        return self * -1.0

    def class_sums(self, classes=None) -> torch.Tensor:
        return self.sums


@dataclass(frozen=True)
class Term:
    """One summand. `pair` is the class (means: "on", "T", "b") or pair of classes
    ("on,on", "T,T", "T,T'", "on,T", "b,b", "on,b", "b,T", "b,b'"; at a non-trigger query
    "T,T", "T,T'", "rho,rho", "T,rho", "rho,rho'"). Tags:

    - channel: "mean", "spread" (spread of the means given ell), "attention" (noise of
      M, Q read through the mean readout), "readout" (noise of Gamma);
    - column: for "readout", "own" (the logit's own column, sigma_Gamma^on), "other" (the
      other columns, sigma_Gamma^N) or "all" (a trigger row, sigma_Gamma^T);
    - mechanism: for the means "signal", "free", "bulk", "bulk_a", "baseline"; for the
      spread the product of two of them, e.g. "signal*bulk"; for the noise "incoherent",
      "source_coherent", "token_coherent", "a_incoherent", "a_coherent", "mean_amplitude";
    - keyset: the key-set sum or energy it comes from ("C_all", "C_phi", "C_b",
      "C_phi_all", "C_b_all", "C_phi_b", "C_bb'", "E", "B_on2", "B_b2"; at a non-trigger
      query "C_rho", "C_rho_all", "C_rhorho'");
    - detail: the sub-mechanism, e.g. "witness", "bigram", "BB", "NB", "same_key";
    - params: the factors it multiplies (readout factor, noise variance, attention).
    """
    name: str
    pair: str
    channel: str
    mechanism: str
    label: str
    value: object = field(repr=False, compare=False)
    column: str = ""
    keyset: str = ""
    detail: str = ""
    params: tuple = ()

    def tags(self) -> dict:
        return {"pair": self.pair, "channel": self.channel, "column": self.column, "mechanism": self.mechanism,
                "keyset": self.keyset, "detail": self.detail, "label": self.label}


def _is_noise(term: Term) -> bool:
    return term.channel in ("attention", "readout")


# the labels of the terms at a non-trigger query (nontrigger.S_LABEL, MU_LABEL), which the
# pruned presets keep
NONTRIGGER_LABELS = ("eq-apx:pairs_nontrigger", "eq-apx:nontrigger_means")


def _pruned(dropped: frozenset) -> Callable[[Term], bool]:
    return lambda term: term.label in NONTRIGGER_LABELS or term.name not in dropped


# preset name -> predicate on a Term. The pruned presets (Phase 6, presets.DROPPED; nested,
# counting < pruned < pruned_late < compact < minimal) drop named terms at a trigger
# query (levels S and ell) and keep every term at a non-trigger query.
PRESETS: dict[str, Callable[[Term], bool]] = {
    "full": lambda term: True,
    "means": lambda term: term.channel == "mean",
    "no_noise": lambda term: not _is_noise(term),
    "no_spread": lambda term: term.channel != "spread",
    **{name: _pruned(dropped) for name, dropped in DROPPED.items()},
}


class TermTable:
    """All the terms of one level at one kind of query, one value per row.

        table = trigger_terms(variables, order_params, beta, L)
        table.names(pair="on,on", channel="attention")
        table.total("on,on")                         # Var h^on given S, all terms
        table.total("on,on", keep="no_noise")        # a preset
        table.total("on,on", keep=lambda t: t.mechanism != "token_coherent")
        table.moments(keep)                          # {pair: total}
        table.aggregate(("pair", "mechanism"))       # {(pair, mechanism): total}

    level: "tau" (exact given the sequence), "S" (given the summary statistics),
    "ell" (given ell), "mu" (non-trigger query, given mu). query: "trigger" or
    "nontrigger". rows: a QueryTable with at least "mu" and "ell". b_classes: (2, nb)
    boolean, the outputs of other triggers and the plain tokens of the b (or rho) block.
    """

    def __init__(self, terms: Iterable[Term], rows: QueryTable, level: str, query: str, K: int, V: int,
                 b_classes: torch.Tensor | None = None):
        self.terms = {}
        for term in terms:
            if term.name in self.terms:
                raise ValueError(f"duplicate term {term.name!r}")
            self.terms[term.name] = term
        self.rows, self.level, self.query, self.K, self.V = rows, level, query, K, V
        nb = V - K - 1 if query == "trigger" else V - K
        outputs = torch.arange(nb) < (K - 1 if query == "trigger" else K)
        self.b_classes = torch.stack([outputs, ~outputs]) if b_classes is None else b_classes

    def __len__(self):
        return len(self.terms)

    def __iter__(self):
        return iter(self.terms.values())

    def __getitem__(self, name: str) -> Term:
        return self.terms[name]

    def __repr__(self):
        return (f"TermTable(level={self.level!r}, query={self.query!r}, rows={self.rows.num_rows}, "
                f"terms={len(self)}, pairs={self.pairs()})")

    def pairs(self) -> list[str]:
        return list(dict.fromkeys(term.pair for term in self))

    def names(self, keep=None, **tags) -> list[str]:
        """The names kept by `keep` whose tags equal `tags` (e.g. pair="on,on")."""
        kept = self.resolve(keep)
        return [name for name, term in self.terms.items() if name in kept
                and all(getattr(term, tag) == value for tag, value in tags.items())]

    def resolve(self, keep=None) -> set[str]:
        """The set of term names that `keep` selects: None or "full" (all), a preset
        name, a predicate Term -> bool, or an iterable of names."""
        if keep is None:
            return set(self.terms)
        if isinstance(keep, str):
            if keep not in PRESETS:
                raise ValueError(f"unknown preset {keep!r}; presets: {list(PRESETS)}")
            keep = PRESETS[keep]
        if callable(keep):
            return {name for name, term in self.terms.items() if keep(term)}
        names = set(keep)
        unknown = names - set(self.terms)
        if unknown:
            raise ValueError(f"unknown terms: {sorted(unknown)[:5]}{' ...' if len(unknown) > 5 else ''}")
        return names

    def select(self, keep) -> "TermTable":
        """A table with only the kept terms (the others count as 0 in every total)."""
        kept = self.resolve(keep)
        table = TermTable([term for term in self if term.name in kept], self.rows, self.level, self.query,
                          self.K, self.V, self.b_classes)
        table._shapes = self._templates()
        return table

    def _templates(self) -> dict:
        templates = getattr(self, "_shapes", {})
        for term in self:
            if term.pair not in templates:
                value = term.value
                templates[term.pair] = (Bilinear() if isinstance(value, Bilinear) else
                                        ClassPairSums(torch.zeros_like(value.sums)) if isinstance(value, ClassPairSums)
                                        else torch.zeros_like(value))
        return templates

    def total(self, pair: str, keep=None):
        """The sum of the kept terms of `pair`, in the order of the table (so keep=None
        is bit for bit the full formula); a zero of the right shape if none is kept."""
        kept = self.resolve(keep)
        total = None
        for term in self:
            if term.pair == pair and term.name in kept:
                total = term.value if total is None else total + term.value
        if total is None:
            templates = self._templates()
            if pair not in templates:
                raise KeyError(f"no pair {pair!r} in this table; pairs: {self.pairs()}")
            return templates[pair]
        return total

    def moments(self, keep=None) -> dict:
        """{pair: total} over every pair of the table."""
        return {pair: self.total(pair, keep) for pair in self.pairs()}

    def aggregate(self, by=("pair", "channel", "mechanism"), keep=None) -> dict:
        """{tuple of the tags in `by`: sum of the kept terms with those tags}."""
        kept = self.resolve(keep)
        groups = {}
        for term in self:
            if term.name in kept:
                key = tuple(getattr(term, tag) for tag in by)
                groups[key] = term.value if key not in groups else groups[key] + term.value
        return groups

    def class_mean(self, value, pair: str):
        """The class view of a total: for a b-indexed pair ("b,b", "on,b", "b,T", ...)
        the mean over the outputs and over the plain tokens, (rows, 2); for "b,b'" the
        mean over pairs b != b' of each pair of classes, (rows, 2, 2); else unchanged."""
        classes = self.b_classes.to(device=_device(value))
        if isinstance(value, (Bilinear, ClassPairSums)):
            sizes = classes.sum(1).to(torch.float64)
            counts = sizes[:, None] * sizes[None, :] - torch.diag(sizes)
            return value.class_sums(classes) / counts.to(device=classes.device)
        if value.dim() == 2:
            weights = classes.to(value.dtype)
            return value @ (weights / weights.sum(1, keepdim=True)).T
        return value


def _device(value):
    if isinstance(value, Bilinear):
        return value.parts[0][1].device if value.parts else "cpu"
    if isinstance(value, ClassPairSums):
        return value.sums.device
    return value.device


def with_value(term: Term, value) -> Term:
    return replace(term, value=value)
