"""Independent oracle for tests/test_terms.py: a port of the checks of the appendix
(code/_scratch/appendix_check/: core.py, noise_formulas.py, s9_fixed_appendix.py,
s9c_nontrigger_fixed.py, paper.py, validated in archive/scratch/2026-10-07-2153_ansatz-appendix-check.tex),
written as literal transcriptions of the appendix formulas, independent of icl.theory.

Adaptations: the profile is a tensor `m` (the mean previous-token entries, entry i <-> key
nu = i + 2) instead of M_on times a function, i.e. Params.Mon = 1 and phi = m, which leaves
every formula unchanged; `ell_fixed` takes the window moments and background sums as inputs
and keeps n(n - 1)/2 exact in E g^B_all (as the package does: deterministic, kept exact).
Positions are the paper's: tok1[:, nu] is the token at position nu (column 0 a dummy).
"""
from __future__ import annotations

from dataclasses import dataclass, replace

import torch

DT = torch.float64


@dataclass(frozen=True)
class Params:
    V: int
    K: int
    L: int
    m: torch.Tensor          # (L-1,) previous-token means, entry i <-> key nu = i + 2
    beta: float = 0.25
    Moff: float = 0.0
    Qon: float = 0.0
    QT: float = 0.0
    Gon: float = 0.0
    GT: float = 0.0
    sMon: float = 0.0
    sMoff: float = 0.0
    sQon: float = 0.0
    sQT: float = 0.0
    sQN: float = 0.0
    sGon: float = 0.0
    sGT: float = 0.0
    sGN: float = 0.0
    Mon: float = 1.0         # phi = m: M_on is folded into the profile

    @property
    def pT(self):
        return 1.0 / (self.V + self.K)

    @property
    def dQ(self):
        return self.Qon - self.QT

    def phi(self, u):
        index = (torch.as_tensor(u, dtype=DT) * self.L).round().long() - 2
        return self.m[index.clamp(min=0, max=len(self.m) - 1)] * (index >= 0)

    def with_(self, **kw):
        return replace(self, **kw)

    @classmethod
    def from_order_params(cls, op: dict, V, K, L, beta):
        get = lambda name: float(op.get(name, 0.0))
        sd = lambda name: get(name) ** 0.5
        m = op["M_profile"].double() if "M_profile" in op else torch.full((L - 1,), get("M_on"), dtype=DT)
        return cls(V=V, K=K, L=L, m=m, beta=beta, Moff=get("M_off"), Qon=get("Q_on"), QT=get("Q_T"), Gon=get("G_on"),
                   GT=get("G_T"), sMon=sd("var_M_on"), sMoff=sd("var_M_off"), sQon=sd("var_Q_on"), sQT=sd("var_Q_T"),
                   sQN=sd("var_Q_N"), sGon=sd("var_G_on"), sGT=sd("var_G_T"), sGN=sd("var_G_N"))


def c(x):
    return torch.tensor(x, dtype=DT)


def mean_matrices(p: Params):
    L, V, K = p.L, p.V, p.K
    nu = torch.arange(L + 1).view(-1, 1)
    sg = torch.arange(L + 1).view(1, -1)
    valid = (nu >= 1) & (sg >= 1)
    prof = p.phi(nu.to(DT) / L).expand(L + 1, L + 1)
    M = torch.where(valid & (sg == nu - 1), p.Mon * prof, c(0.0))
    M = torch.where(valid & (sg <= nu - 2), c(p.Moff), M)
    a, b = torch.arange(V).view(-1, 1), torch.arange(V).view(1, -1)
    trig = a < K
    Q = torch.where(trig, torch.where(a == b, c(p.Qon), c(p.QT)), c(0.0))
    G = torch.where(trig, c(p.GT), torch.where(a == b, c(p.Gon), c(0.0))).expand(V, V).clone()
    return M, Q, G


def var_matrices(p: Params):
    L, V, K = p.L, p.V, p.K
    nu = torch.arange(L + 1).view(-1, 1)
    sg = torch.arange(L + 1).view(1, -1)
    valid = (nu >= 1) & (sg >= 1)
    vM = torch.zeros(L + 1, L + 1, dtype=DT)
    vM[valid & (sg == nu - 1)] = p.sMon ** 2
    vM[valid & (sg <= nu - 2)] = p.sMoff ** 2
    a, b = torch.arange(V).view(-1, 1), torch.arange(V).view(1, -1)
    trig = a < K
    vQ = torch.where(trig, torch.where(a == b, c(p.sQon ** 2), c(p.sQT ** 2)), c(p.sQN ** 2)).expand(V, V).clone()
    vG = torch.where(trig, c(p.sGT ** 2), torch.where(a == b, c(p.sGon ** 2), c(p.sGN ** 2))).expand(V, V).clone()
    return vM, vQ, vG


def onehot(t, V):
    return torch.nn.functional.one_hot(t, V).to(DT)


def trigger_stats(p: Params, tok1, out, mu):
    """Every statistic of tables summary_stats and noise_stats at trigger queries at mu."""
    L, V, K = p.L, p.V, p.K
    B = tok1.shape[0]
    a = tok1[:, mu]
    tgt = out.gather(1, a[:, None]).squeeze(1)
    nu = torch.arange(2, mu)
    tk, prev = tok1[:, 2:mu], tok1[:, 1:mu - 1]
    is_a = tok1[:, 1:mu] == a[:, None]
    ell = is_a.sum(1)
    cum = torch.cumsum(is_a.to(torch.long), 1)
    na2 = torch.cat([torch.zeros(B, 1, dtype=torch.long), cum[:, : mu - 3]], 1).to(DT)
    ph = p.phi(nu.to(DT) / L)
    is_t = tk == tgt[:, None]
    is_w = is_t & (prev == a[:, None])
    is_f = is_t & ~is_w
    nm2 = (nu - 2).to(DT)
    S = {"ell": ell.to(DT), "f": is_f.sum(1).to(DT), "N": (is_w * ph).sum(1), "F": (is_f * ph).sum(1),
         "W": (is_t * nm2).sum(1), "P": (is_t * na2).sum(1), "R": na2.sum(1), "N2": (is_w * ph ** 2).sum(1),
         "Rphi": (is_w * ph * (mu - 1 - nu.to(DT))).sum(1)}
    k = p.Mon * ph + p.Moff * nm2
    S["X1"], S["X2"], S["Phihat"] = (k * na2).sum(1), (na2 ** 2).sum(1), ph.sum().expand(B)

    def per_tok(vals, mask=None):
        v = vals.expand(B, -1) if vals.dim() == 1 else vals
        v = v * mask if mask is not None else v
        return torch.zeros(B, V, dtype=DT).scatter_add_(1, tk, v.to(DT))
    one = torch.ones(B, tk.shape[1], dtype=DT)
    S["c"], S["U"], S["Wb"], S["Pb"] = per_tok(one), per_tok(ph), per_tok(nm2), per_tok(na2)
    trg = (prev < K).to(DT)
    s1, s2 = per_tok(ph, trg), per_tok(ph ** 2, trg)
    S["Y"], S["Utr"] = s1 ** 2 - s2, s1
    cls = torch.full((B, V), 3, dtype=torch.long)
    cls.scatter_(1, out, 2)
    cls[:, :K] = 0
    cls.scatter_(1, tgt[:, None], 1)
    S["cls"], S["a"], S["tgt"] = cls, a, tgt
    return S


def amplitude_moments(p, tok1, mu, Mb, Qb, vM, vQ):
    a, src = tok1[:, mu], tok1[:, 1:mu]
    Ms, vMs = Mb[1:mu, 1:mu], vM[1:mu, 1:mu]
    qb, qv = Qb[a].gather(1, src), vQ[a].gather(1, src)
    return qb @ Ms.T, (qb ** 2 + qv) @ vMs.T, torch.einsum("ns,bsv->bnv", Ms, onehot(src, p.V)), vQ[a]


def exact_classes(p, tok1, mu, S, b_idx, b2_idx, chunk=128):
    """route 1: eqs. cov_exact, Calpha; the class entries at trigger queries."""
    Mb, Qb, Gb = mean_matrices(p)
    vM, vQ, vG = var_matrices(p)
    pref = (p.beta / p.L) ** 2
    acc = {}
    for i in range(0, tok1.shape[0], chunk):
        sl = slice(i, i + chunk)
        t = tok1[sl]
        ar = torch.arange(t.shape[0])
        Abar, d, Snu, qv = amplitude_moments(p, t, mu, Mb, Qb, vM, vQ)
        Ca = torch.diag_embed(d) + torch.einsum("bnc,bc,bmc->bnm", Snu, qv, Snu)
        ok = onehot(t[:, 1:mu], p.V)
        Ctok = ok.transpose(1, 2) @ Ca @ ok
        Bc = torch.einsum("bnc,bn->bc", ok, Abar)
        Ec = Bc ** 2 + torch.diagonal(Ctok, dim1=1, dim2=2)
        Kc = pref * (Gb @ Ctok @ Gb.T + torch.diag_embed(Ec @ vG.T))
        tg, b, b2 = S["tgt"][sl], b_idx[sl], b2_idx[sl]
        rows = Ctok.sum(2)
        r = dict(C_all=Ctok.sum((1, 2)), C_phi=Ctok[ar, tg, tg], C_phi_all=rows[ar, tg], E=Ec.sum(1), Bon=Bc[ar, tg],
                 BT=Bc.sum(1), Bb=Bc[ar, b], C_b=Ctok[ar, b, b], C_b_all=rows[ar, b], C_phi_b=Ctok[ar, tg, b],
                 C_b_b2=Ctok[ar, b, b2], var_on=Kc[ar, tg, tg], var_T=Kc[:, 0, 0], VTc=Kc[:, 0, 1],
                 cov_on_T=Kc[ar, tg, 0], var_b=Kc[ar, b, b], cov_on_b=Kc[ar, tg, b], cov_b_T=Kc[ar, b, 0],
                 cov_b_b2=Kc[ar, b, b2])
        for k, v in r.items():
            acc.setdefault(k, []).append(v)
    return {k: torch.cat(v) for k, v in acc.items()}


def brute_cov(p, tok1_row, mu):
    """route 2: the covariance of ONE sequence from the factorised second moments
    E[M M] E[Q Q] E[G G], no A / alpha decomposition."""
    Mb, Qb, Gb = mean_matrices(p)
    vM, vQ, vG = var_matrices(p)
    a, src = int(tok1_row[mu]), tok1_row[1:mu]
    Ms, vMs = Mb[1:mu, 1:mu], vM[1:mu, 1:mu]
    qb = Qb[a, src]
    same = (src[:, None] == src[None, :]).to(DT)
    EQ = qb[:, None] * qb[None, :] + vQ[a, src][:, None] * same
    F = Ms @ EQ @ Ms.T + torch.diag((vMs * torch.diagonal(EQ)[None, :]).sum(1))
    Gk, vGk = Gb[:, src], vG[:, src]
    diagR = (vGk * (F * same).sum(1)[None, :]).sum(1)
    hbar = Gk @ (Ms @ qb)
    return (p.beta / p.L) ** 2 * (Gk @ F @ Gk.T + torch.diag(diagR) - hbar[:, None] * hbar[None, :])


def background_sums(p, mu):
    """D, K1, J2, G1, K2, Phihat, n: the exact discrete sums."""
    Mb, _, _ = mean_matrices(p)
    vM, _, _ = var_matrices(p)
    Ms, vMs = Mb[1:mu, 1:mu], vM[1:mu, 1:mu]
    k, om = Ms.sum(1), Ms.sum(0)
    nu = torch.arange(2, mu)
    return dict(D=vMs.sum().item(), K1=k.sum().item(), J2=(om ** 2).sum().item(), G1=(Ms ** 2).sum().item(),
                K2=(k ** 2).sum().item(), Phihat=p.phi(nu.to(DT) / p.L).sum().item(), n=mu - 2)


def rates(p):
    return dict(k2=(p.V + 3 * p.K) * p.pT ** 2, kN=(p.V + 2 * p.K) * p.pT / p.V, kNN=(p.V + 2 * p.K) / p.V ** 2,
                kT=p.pT, kbi=p.pT ** 2 * (p.K + 1 + 2 * p.K / p.V))


def T(r, gI, gJ, Y=0.0):
    """eq. tc_rule, g = (N, tr, B)."""
    (aN, aT, aB), (bN, bT, bB) = gI, gJ
    return r["k2"] * aB * bB + r["kN"] * (aN * bB + aB * bN) + r["kT"] * (aT * bB + aB * bT) + r["kNN"] * aN * bN + Y


def T_all(r, gI, Kall, gB_all, Mon2Y):
    """eq. tc_all."""
    aN, aT, aB = gI
    return (r["k2"] * aB + r["kN"] * aN) * Kall + r["kT"] * aT * gB_all + Mon2Y


def given_S(p, S, bg, b, b2):
    """eqs. keyset_sums (corrected rule), E, brackets, given S (s9 given_S_fixed + the energy)."""
    r = rates(p)
    q2T, q2on = p.QT ** 2 + p.sQT ** 2, p.Qon ** 2 + p.sQon ** 2
    dq2, sT2, son2 = q2on - q2T, p.sQT ** 2, p.sQon ** 2
    D, K1, J2, G1, K2, n, Phih = (bg[k] for k in ("D", "K1", "J2", "G1", "K2", "n", "Phihat"))
    J = lambda cIJ, cI, cJ: cIJ / n * G1 + (cI * cJ - cIJ) / n ** 2 * (J2 - G1)
    AX = lambda lx, X: dq2 * (p.sMon ** 2 * lx + p.sMoff ** 2 * X)
    ar = torch.arange(len(b))
    ell, f, N, F, W, P, Rr = (S[k] for k in ("ell", "f", "N", "F", "W", "P", "R"))
    UpsP, UpsR = p.Mon * N + p.Moff * P, p.Mon * N + p.Moff * Rr
    gphi = (p.Mon * F, 0 * F, p.Moff * (W - P))
    gb = lambda i: (p.Mon * (S["U"][ar, i] - S["Utr"][ar, i]), p.Mon * S["Utr"][ar, i],
                    p.Moff * (S["Wb"][ar, i] - S["Pb"][ar, i]))
    g1, g2 = gb(b), gb(b2)
    Yb = p.Mon ** 2 * S["Y"][ar, b]
    gBall = p.Moff * (n * (n - 1) / 2 - Rr)
    Kall = K1 - UpsR
    c1, c2, Pb1, Pb2 = S["c"][ar, b], S["c"][ar, b2], S["Pb"][ar, b], S["Pb"][ar, b2]
    o = {}
    o["C_all"] = q2T * D + sT2 * (J2 + r["k2"] * Kall ** 2) + AX(ell, Rr) + son2 * UpsR ** 2
    o["C_phi"] = (q2T * (ell + f) / n * D + sT2 * (J(ell + f, ell + f, ell + f) - p.Mon ** 2 * S["N2"] + T(r, gphi, gphi))
                  + AX(ell, P) + son2 * UpsP ** 2)
    o["C_b"] = q2T * c1 / n * D + sT2 * (J(c1, c1, c1) + T(r, g1, g1, Yb)) + AX(0, Pb1) + son2 * (p.Moff * Pb1) ** 2
    o["C_phi_all"] = (q2T * (ell + f) / n * D + sT2 * ((ell + f) / n * J2 - p.Mon ** 2 * S["N2"] - p.Mon * p.Moff * S["Rphi"]
                                                     + T_all(r, gphi, Kall, gBall, 0.0)) + AX(ell, P) + son2 * UpsP * UpsR)
    o["C_b_all"] = (q2T * c1 / n * D + sT2 * (c1 / n * J2 + T_all(r, g1, Kall, gBall, Yb)) + AX(0, Pb1)
                    + son2 * p.Moff * Pb1 * UpsR)
    o["C_phi_b"] = (sT2 * (J(0, ell + f, c1) - p.Mon * p.Moff * c1 / n * S["Rphi"] + T(r, gphi, g1))
                    + son2 * UpsP * p.Moff * Pb1)
    o["C_b_b2"] = sT2 * (J(0, c1, c2) + T(r, g1, g2)) + son2 * p.Moff ** 2 * Pb1 * Pb2
    Bon = p.Mon * (p.Qon * N + p.QT * F) + p.Moff * (p.QT * W + p.dQ * P)
    BT = p.QT * (p.Mon * Phih + p.Moff * n * (n - 1) / 2) + p.dQ * UpsR
    Cd = q2T * D + sT2 * (G1 + r["k2"] * K2)
    Co = q2T * D + sT2 * (J2 + r["k2"] * K1 ** 2) - Cd
    o["E"] = (Bon ** 2 + p.QT ** 2 * K2 + 2 * p.QT * p.dQ * p.Moff * S["X1"] + (p.dQ * p.Moff) ** 2 * S["X2"]
              + r["k2"] * (BT - Bon) ** 2 + Cd + r["k2"] * Co + AX(ell, Rr) + son2 * UpsP ** 2
              + sT2 * (r["kbi"] - r["k2"] ** 2) * (p.Mon * Phih) ** 2)
    o["Bon"], o["BT"] = Bon, BT
    o["Bb"] = p.Mon * p.QT * S["U"][ar, b] + p.Moff * (p.QT * S["Wb"][ar, b] + p.dQ * Pb1)
    return o


def pairs(p, F):
    """eq. pairs_trigger from key-set sums and energy F (given S or their means given ell,
    with E(B^2) in place of B^2)."""
    pref = (p.beta / p.L) ** 2
    g2 = p.Gon ** 2 + p.sGon ** 2
    on2 = F["Bon2"] if "Bon2" in F else F["Bon"] ** 2
    b2 = F["Bb2"] if "Bb2" in F else F["Bb"] ** 2
    o = {"var_on": pref * (p.sGon ** 2 * on2 + g2 * F["C_phi"] + p.sGN ** 2 * (F["E"] - on2 - F["C_phi"])),
         "VTc": pref * p.GT ** 2 * F["C_all"]}
    o["var_T"] = o["VTc"] + pref * p.sGT ** 2 * F["E"]
    o["cov_on_T"] = pref * p.Gon * p.GT * F["C_phi_all"]
    o["var_b"] = pref * (p.sGon ** 2 * b2 + g2 * F["C_b"] + p.sGN ** 2 * (F["E"] - b2 - F["C_b"]))
    o["cov_on_b"] = pref * p.Gon ** 2 * F["C_phi_b"]
    o["cov_b_T"] = pref * p.Gon * p.GT * F["C_b_all"]
    o["cov_b_b2"] = pref * p.Gon ** 2 * F["C_b_b2"]
    return o


# ---- given ell (s9 ell_fixed, plus E C_all and E E transcribed from eqs. keyset_means, E_mean) ------

def mean_var_brackets(p, mu, ell, Lam, rb, rb2, w, bg):
    """E B, Var B, Cov B of eqs. mean_logits_ell, var_logits, cov_logits divided by beta/L Gamma
    (paper.py), with Phihat and n(n-1)/2 exact in E B^T."""
    u, v, s, ww, z = p.Mon * p.Qon, p.Mon * p.QT, p.Mon * p.dQ, p.Moff * p.QT, p.Moff * p.dQ
    P1, P2, Psi = w["Phi1"], w["Phi2"], w["Psi"]
    d = Psi - P1 / 2
    n = bg["n"]
    E = {"on": P1 * (u * ell + v * Lam) + ww * mu * (ell + Lam) / 2 + z * (ell * (ell - 1) / 2 + Lam * ell / 2),
         "T": p.QT * (p.Mon * bg["Phihat"] + p.Moff * n * (n - 1) / 2) + ell * (s * P1 + z * mu / 2),
         "b": rb * (v * P1 + ww * mu / 2 + z * ell / 2)}
    Var = {"on": (u ** 2 * ell * (P2 - P1 ** 2) + v ** 2 * Lam * P2 + ww ** 2 * mu ** 2 * (ell / 12 + Lam / 3)
                  + z ** 2 * (Lam * ell * (2 * ell + 1) / 6 + Lam ** 2 * ell / 12) + 2 * u * ww * mu * ell * d
                  - 2 * u * z * Lam * ell * d + 2 * v * ww * mu * Lam * Psi + 2 * v * z * Lam * ell * Psi
                  + 2 * ww * z * mu * Lam * ell / 4),
           "T": ell * (s ** 2 * (P2 - P1 ** 2) - 2 * s * z * mu * d + z ** 2 * mu ** 2 / 12),
           "b": rb * (v ** 2 * P2 + ww ** 2 * mu ** 2 / 3 + z ** 2 * (ell * (2 * ell + 1) / 6 + rb * ell / 12)
                      + 2 * v * ww * mu * Psi + 2 * v * z * ell * Psi + 2 * ww * z * mu * ell / 3)}
    Cov = {"on,T": ell * (s * (u * (P2 - P1 ** 2) + (ww * mu - z * Lam) * d) - z * mu * (u * d + (ww * mu - z * Lam) / 12)),
           "on,b": -z * rb * ell * (u * d + (ww * mu - z * Lam) / 12),
           "T,b": z * rb * ell * (z * mu / 12 - s * d),
           "b,b2": z ** 2 * rb * rb2 * ell / 12}
    return E, Var, Cov


def ell_fixed(p, mu, ell, Lam, sb, sb2, w, bg):
    """E of the key-set sums and of the energy at fixed ell (eqs. g_moments - E_mean)."""
    r = rates(p)
    P1, P2, Psi = w["Phi1"], w["Phi2"], w["Psi"]
    dP = Psi - P1 / 2
    D, K1, J2, G1, K2, n = (bg[k] for k in ("D", "K1", "J2", "G1", "K2", "n"))
    q2T, q2on = p.QT ** 2 + p.sQT ** 2, p.Qon ** 2 + p.sQon ** 2
    dq2, sT2, son2 = q2on - q2T, p.sQT ** 2, p.sQon ** 2
    Mon, Moff = p.Mon, p.Moff
    rb, rb2 = (1 + sb) * Lam, (1 + sb2) * Lam
    VarW = mu ** 2 * (ell / 12 + Lam / 3)
    VarP = Lam * ell * (2 * ell + 1) / 6 + Lam ** 2 * ell / 12
    EP = ell * (ell - 1) / 2 + Lam * ell / 2
    VarPb = lambda rr: rr * ell * (2 * ell + 1) / 6 + rr ** 2 * ell / 12
    E_UpsP = Mon * ell * P1 + Moff * EP
    V_UpsP = Mon ** 2 * ell * (P2 - P1 ** 2) - 2 * Mon * Moff * Lam * ell * dP + Moff ** 2 * VarP
    E_UpsR = ell * (Mon * P1 + Moff * mu / 2)
    V_UpsR = ell * (Mon ** 2 * (P2 - P1 ** 2) - 2 * Mon * Moff * mu * dP + Moff ** 2 * mu ** 2 / 12)
    C_PR = ell * (Mon ** 2 * (P2 - P1 ** 2) - Mon * Moff * (mu + Lam) * dP + Moff ** 2 * mu * Lam / 12)
    C_PPb = lambda rr: rr * ell * (Moff * Lam / 12 - Mon * dP)
    C_RPb = lambda rr: rr * ell * (Moff * mu / 12 - Mon * dP)
    Eg = {"phi": (Mon * Lam * P1, 0.0, Moff * (mu * (ell + Lam) / 2 - EP)),
          "b": (Mon * Lam * P1, sb * Mon * Lam * P1, Moff * rb * (mu - ell) / 2),
          "b2": (Mon * Lam * P1, sb2 * Mon * Lam * P1, Moff * rb2 * (mu - ell) / 2)}
    EgBall = Moff * (n * (n - 1) / 2 - mu * ell / 2)        # the appendix: Moff mu (mu - ell)/2
    cNB = Mon * Moff * Lam * (mu - ell) * Psi
    Vg = {"phi": dict(NN=Mon ** 2 * Lam * P2, TT=0.0, BB=Moff ** 2 * (VarW + VarP - mu * Lam * ell / 2), NB=cNB, TB=0.0),
          "b": dict(NN=Mon ** 2 * Lam * P2, TT=sb * Mon ** 2 * Lam * P2,
                    BB=Moff ** 2 * (mu ** 2 * rb / 3 + VarPb(rb) - 2 * mu * rb * ell / 3), NB=cNB, TB=sb * cNB)}
    covBB_phib = lambda rr: Moff ** 2 * rr * ell * (mu + Lam) / 12
    covBB_bb2 = Moff ** 2 * rb * rb2 * ell / 12
    cov_gBphi_R = Moff * ell * (mu + Lam) * (Mon * dP - Moff * mu / 12)
    cov_gBb_R = -Moff * rb * ell * (Moff * mu / 12 - Mon * dP)
    k2, kN, kNN, kT = r["k2"], r["kN"], r["kNN"], r["kT"]

    def ET_self(I, Y):
        m, v = Eg[I], Vg[I]
        return T(r, m, m) + k2 * v["BB"] + 2 * kN * v["NB"] + 2 * kT * v["TB"] + kNN * v["NN"] + Y
    EY = sb * Mon ** 2 * Lam ** 2 * P1 ** 2
    ET = {"phiphi": ET_self("phi", 0.0), "bb": ET_self("b", EY),
          "phiall": (k2 * Eg["phi"][2] + kN * Eg["phi"][0]) * (K1 - E_UpsR) - k2 * cov_gBphi_R,
          "ball": (k2 * Eg["b"][2] + kN * Eg["b"][0]) * (K1 - E_UpsR) - k2 * cov_gBb_R + kT * Eg["b"][1] * EgBall + EY,
          "phib": T(r, Eg["phi"], Eg["b"]) + k2 * covBB_phib(rb), "bb2": T(r, Eg["b"], Eg["b2"]) + k2 * covBB_bb2}
    o = {}
    o["C_all"] = (q2T * D + sT2 * (J2 + k2 * ((K1 - E_UpsR) ** 2 + V_UpsR))
                  + dq2 * (p.sMon ** 2 * ell + p.sMoff ** 2 * mu * ell / 2) + son2 * (E_UpsR ** 2 + V_UpsR))
    o["C_phi"] = (q2T * (ell + Lam) / n * D + sT2 * ((ell + Lam) / n * G1 + (ell ** 2 - ell + 2 * ell * Lam + Lam ** 2) / n ** 2
                                                    * (J2 - G1) - Mon ** 2 * ell * P2 + ET["phiphi"])
                  + dq2 * (p.sMon ** 2 * ell + p.sMoff ** 2 * EP) + son2 * (E_UpsP ** 2 + V_UpsP))
    o["C_b"] = (q2T * rb / n * D + sT2 * (rb / n * G1 + rb ** 2 / n ** 2 * (J2 - G1) + ET["bb"])
                + dq2 * p.sMoff ** 2 * rb * ell / 2 + son2 * Moff ** 2 * (rb ** 2 * ell ** 2 / 4 + VarPb(rb)))
    o["C_phi_all"] = (q2T * (ell + Lam) / n * D + sT2 * ((ell + Lam) / n * J2 - Mon ** 2 * ell * P2
                                                        - Mon * Moff * mu * ell * (P1 - Psi) + ET["phiall"])
                      + dq2 * (p.sMon ** 2 * ell + p.sMoff ** 2 * EP) + son2 * (E_UpsP * E_UpsR + C_PR))
    o["C_b_all"] = (q2T * rb / n * D + sT2 * (rb / n * J2 + ET["ball"]) + dq2 * p.sMoff ** 2 * rb * ell / 2
                    + son2 * Moff * (rb * ell / 2 * E_UpsR + C_RPb(rb)))
    o["C_phi_b"] = (sT2 * ((ell + Lam) * rb / n ** 2 * (J2 - G1) - Mon * Moff * rb / n * mu * ell * (P1 - Psi) + ET["phib"])
                    + son2 * Moff * (rb * ell / 2 * E_UpsP + C_PPb(rb)))
    o["C_b_b2"] = (sT2 * (rb * rb2 / n ** 2 * (J2 - G1) + ET["bb2"])
                   + son2 * Moff ** 2 * rb * rb2 * ell * (ell / 4 + 1 / 12))
    # eq. E_mean
    E, Var, Cov = mean_var_brackets(p, mu, ell, Lam, rb, rb2, w, bg)
    E_Bon2 = E["on"] ** 2 + Var["on"]
    E_diff2 = (E["T"] - E["on"]) ** 2 + Var["T"] + Var["on"] - 2 * Cov["on,T"]
    Cd = q2T * D + sT2 * (G1 + k2 * K2)
    Co = q2T * D + sT2 * (J2 + k2 * K1 ** 2) - Cd
    o["E"] = (E_Bon2 + p.QT ** 2 * K2 + 2 * p.QT * p.dQ * Moff * mu * ell * (Mon * Psi + Moff * mu / 3)
              + (p.dQ * Moff) ** 2 * mu * ell * (2 * ell + 1) / 6 + k2 * E_diff2 + Cd + k2 * Co
              + dq2 * (p.sMon ** 2 * ell + p.sMoff ** 2 * mu * ell / 2) + son2 * (E_UpsP ** 2 + V_UpsP)
              + sT2 * (r["kbi"] - k2 ** 2) * (Mon * bg["Phihat"]) ** 2)
    o["Bon2"], o["Bb2"] = E_Bon2, E["b"] ** 2 + Var["b"]
    return o


def spread(p, mu, ell, Lam, sb, sb2, w, bg):
    """eqs. var_logits, cov_logits."""
    rb, rb2 = (1 + sb) * Lam, (1 + sb2) * Lam
    _, Var, Cov = mean_var_brackets(p, mu, ell, Lam, rb, rb2, w, bg)
    k2 = (p.beta / p.L) ** 2
    return {"var_on": k2 * p.Gon ** 2 * Var["on"], "var_T": k2 * p.GT ** 2 * Var["T"], "VTc": k2 * p.GT ** 2 * Var["T"],
            "var_b": k2 * p.Gon ** 2 * Var["b"], "cov_on_T": k2 * p.Gon * p.GT * Cov["on,T"],
            "cov_on_b": k2 * p.Gon ** 2 * Cov["on,b"], "cov_b_T": k2 * p.GT * p.Gon * Cov["T,b"],
            "cov_b_b2": k2 * p.Gon ** 2 * Cov["b,b2"]}


# ---- non-trigger query (s9c) -----------------------------------------------------------------

def nontrigger_given_S(p, S, bg):
    """eq. pairs_nontrigger, per token rho (B, V): C_rho, C_rho_all; and C^N, E^N."""
    r = rates(p)
    s2 = p.sQN ** 2
    D, K1, J2, G1, K2, n, Phih = (bg[k] for k in ("D", "K1", "J2", "G1", "K2", "n", "Phihat"))
    J = lambda cIJ, cI, cJ: cIJ / n * G1 + (cI * cJ - cIJ) / n ** 2 * (J2 - G1)
    g = (p.Mon * (S["U"] - S["Utr"]), p.Mon * S["Utr"], p.Moff * S["Wb"])
    cc = S["c"]
    Cd = s2 * (D + G1 + r["k2"] * K2)
    CN = s2 * (D + J2 + r["k2"] * K1 ** 2)
    return {"C_rho": s2 * (cc / n * D + J(cc, cc, cc) + T(r, g, g, p.Mon ** 2 * S["Y"])),
            "C_rho_all": s2 * (cc / n * (D + J2) + T_all(r, g, K1, p.Moff * n * (n - 1) / 2, p.Mon ** 2 * S["Y"])),
            "CN": CN, "EN": Cd + r["k2"] * (CN - Cd) + s2 * (r["kbi"] - r["k2"] ** 2) * (p.Mon * Phih) ** 2}


def nontrigger_per_token(p, tok1, out, mu):
    B, V, K, L = tok1.shape[0], p.V, p.K, p.L
    nu = torch.arange(2, mu)
    tk, prev = tok1[:, 2:mu], tok1[:, 1:mu - 1]
    ph = p.phi(nu.to(DT) / L).expand(B, -1)
    nm2 = (nu - 2).to(DT).expand(B, -1)
    trg = (prev < K).to(DT)
    z = lambda v: torch.zeros(B, V, dtype=DT).scatter_add_(1, tk, v)
    S = dict(c=z(torch.ones_like(ph)), U=z(ph), Wb=z(nm2), Utr=z(ph * trg))
    S["Y"] = S["Utr"] ** 2 - z(ph ** 2 * trg)
    cls = torch.full((B, V), 3, dtype=torch.long)
    cls.scatter_(1, out, 2)
    cls[:, :K] = 0
    S["cls"] = cls
    return S


def nontrigger_mu(p, mu, Lam, s_rho, w, bg):
    """eq. nontrigger_means: the ell = 0 limit of the trigger-query moments with
    q2T, sigma_Q^T -> sigma_Q^N and no a-terms (s9c)."""
    pN = p.with_(QT=0.0, Qon=0.0, sQT=p.sQN, sQon=p.sQN)
    F = ell_fixed(pN, mu, 0, Lam, s_rho, s_rho, w, bg)
    return {"C_rho": F["C_b"], "C_rho_all": F["C_b_all"]}
