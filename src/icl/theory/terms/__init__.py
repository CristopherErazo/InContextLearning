"""The logit moments of the appendix as explicit sums of named terms, at every level of
conditioning (paper/appendix/appendix.tex, sections Ansatz and Effective Loss):

    level "tau"  exact given the sequence      exact_trigger_terms, exact_nontrigger_terms
    level "S"    given the statistics S_mu     trigger_terms, nontrigger_terms
    level "ell"  given ell (trigger query)     trigger_terms_ell
    level "mu"   given mu (non-trigger query)  nontrigger_terms_mu

Each returns a `TermTable`: every term has a name, the appendix label it comes from, tags
(pair of classes, channel, column, mechanism, key set, detail, parameters) and a value per
row. `total(pair, keep)` sums the kept terms into a class moment; `keep` is a set of names,
a predicate on the terms or a preset name of PRESETS. The S and ell (mu) levels share one
definition of the pieces (`trigger._structure`, `nontrigger._structure`), the ell level
being their conditional mean under A1-A3; the token-coherent noise uses the corrected rule
(eqs. predecessor - tc_all). Everything is differentiable in the order parameters, the
profile and the variances, in float64, on any device.
"""
from .table import PRESETS, Bilinear, ClassPairSums, Term, TermTable
from .forms import B_STATS, T_STATS, Form, GivenEll, GivenS, Piece
from .trigger import (coincidence_rates, trigger_statistics, trigger_terms, trigger_terms_ell, window_moments)
from .nontrigger import nontrigger_second_order, nontrigger_statistics, nontrigger_terms, nontrigger_terms_mu
from .exact import exact_nontrigger_terms, exact_trigger_terms

__all__ = ["PRESETS", "Bilinear", "ClassPairSums", "Term", "TermTable", "coincidence_rates", "trigger_statistics",
           "trigger_terms", "trigger_terms_ell", "window_moments", "nontrigger_statistics", "nontrigger_terms",
           "nontrigger_terms_mu", "nontrigger_second_order", "exact_trigger_terms", "exact_nontrigger_terms"]
