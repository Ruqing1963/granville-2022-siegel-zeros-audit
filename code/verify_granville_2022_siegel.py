"""Verification script for the audit note on
A. Granville, "Sieving intervals and Siegel zeros", Acta Arith. 204 (2022), 297-315
(DOI 10.4064/aa201002-25-6).  Page numbers below refer to the Online First version (pp. 1-19).

Run:  python verify_granville_2022_siegel.py
Writes ../data/verify_granville_2022_siegel.txt and ../figures/fig_granville_2022_audit.{pdf,png}

Notation.  delta := sqrt((1-beta) log q) -> 0,  log y_q := (log q/(1-beta))^{1/2},
t := log y / log y_q (the scale),  C = C(u) = (1/2 (1 - log^+(u-1)))^{1/2}.
The proof of Proposition 2 (p. 16) gives, with A = log y/log q and B = 1/((1-beta) log y),
    S(X,y,z) >= 2y/log y * (1 - 1/A - C^2/B + O(1/(A log q) + 1/A^2 + 1/B^2)).
"""
import os
import sys

import numpy as np
import sympy as sp

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUT_TXT = os.path.join(ROOT, "data", "verify_granville_2022_siegel.txt")
OUT_PDF = os.path.join(ROOT, "figures", "fig_granville_2022_audit.pdf")
OUT_PNG = os.path.join(ROOT, "figures", "fig_granville_2022_audit.png")

_lines = []


def say(*a):
    s = " ".join(str(x) for x in a)
    print(s)
    _lines.append(s)


def check(name, ok):
    say(f"[{'PASS' if ok else 'FAIL'}] {name}")
    if not ok:
        check.failures += 1


check.failures = 0
EG = float(np.exp(-np.euler_gamma))


def Cfun(u):
    u = np.asarray(u, dtype=float)
    lp = np.where(u > 2, np.log(np.maximum(u - 1, 1e-300)), 0.0)
    return np.sqrt(0.5 * (1 - lp))


# ---------------------------------------------------------------------------
say("=" * 72)
say("1. Proposition 2 (p. 7) versus its proof (p. 16)")
say("=" * 72)
L, b1, t, C = sp.symbols("L b1 t C", positive=True)      # L = log q, b1 = 1-beta
logyq = sp.sqrt(L / b1)
delta = sp.sqrt(b1 * L)
logy = t * logyq
Aexp = logy / L
Bexp = 1 / (b1 * logy)
loss = sp.simplify(1 / Aexp + C**2 / Bexp)                      # relative loss
check("relative loss 1/A + C^2/B = delta (1/t + C^2 t)", sp.simplify(loss - delta * (1 / t + C**2 * t)) == 0)
# convert to coefficient of y log q/(log y)^2:  (2y/log y)*loss = kappa * y log q/(log y)^2
kappa = sp.simplify(2 * loss * logy / L)
check("coefficient kappa(t) = 2(1 + C^2 t^2)", sp.simplify(kappa - 2 * (1 + C**2 * t**2)) == 0)
check("at t = 1 (stated scale): A = B = 1/delta", sp.simplify(Aexp.subs(t, 1) - 1 / delta) == 0
      and sp.simplify(Bexp.subs(t, 1) - 1 / delta) == 0)
check("at t = 1: 1 - beta = log q/(log y)^2", sp.simplify(b1 - L / logy.subs(t, 1)**2) == 0)
check("at t = 1/C (proof's choice): A = 1/(C delta), B = C/delta",
      sp.simplify(Aexp.subs(t, 1 / C) - 1 / (C * delta)) == 0 and sp.simplify(Bexp.subs(t, 1 / C) - C / delta) == 0)
check("at t = 1/C: 1 - beta = log q/(C log y)^2 (p. 16, as printed)",
      sp.simplify(b1 - L / (C * logy.subs(t, 1 / C))**2) == 0)
check("at t = 1/C: relative loss = 2 C delta (p. 16, as printed)", sp.simplify(loss.subs(t, 1 / C) - 2 * C * delta) == 0)
check("at t = 1/C: delta = log q/(C log y), NOT log q/log y (last line of p. 16)",
      sp.simplify(delta - L / (C * logy.subs(t, 1 / C))) == 0 and sp.simplify(delta - L / logy.subs(t, 1 / C)) != 0)
check("at t = 1/C: kappa = 4 (C cancels), not 4C", sp.simplify(kappa.subs(t, 1 / C) - 4) == 0)
check("t = 1/C minimises the relative loss delta(1/t + C^2 t)",
      sp.solve(sp.diff(1 / t + C**2 * t, t), t) == [1 / C])
check("2(1+C^2) - 4C = 2(1-C)^2 > 0 for C != 1", sp.expand(2 * (1 + C**2) - 4 * C - 2 * (1 - C)**2) == 0)

C12, C3 = float(Cfun(1.5)), float(Cfun(3.0))
say(f"   C(u) = {C12:.4f} on [1,2];  C(3) = {C3:.4f};  range [{C3:.4f}, {C12:.4f}]")
say(f"   u in [1,2]: true 2(1+C^2) = {2*(1+C12**2):.4f}   printed 4C = {4*C12:.4f}")
say(f"   u = 3     : true 2(1+C^2) = {2*(1+C3**2):.4f}   printed 4C = {4*C3:.4f}   ratio = {(1+C3**2)/(2*C3):.4f}")
check("C(3) = sqrt((1-log 2)/2)", abs(C3 - np.sqrt((1 - np.log(2)) / 2)) < 1e-14)
check("ratio true/printed at u=3 equals 1.472 (to 3 d.p.)", round((1 + C3**2) / (2 * C3), 3) == 1.472)
uu = np.linspace(1, 3, 2001)
Cu = Cfun(uu)
check("2(1+C^2) > 4C on all of [1,3]", bool(np.all(2 * (1 + Cu**2) > 4 * Cu)))
say("   first bullet (p. 7): (2y/log y)*2C*delta = 4C delta y/log y -- consistent with the relative-loss form")

# ---------------------------------------------------------------------------
say("=" * 72)
say("2. Fourth bullet (p. 7): 1-beta <= q^{-eps}")
say("=" * 72)
eps = sp.symbols("epsilon", positive=True)
# log log y = log t + 1/2 log log q + 1/2 log(1/(1-beta)) >= (eps/2) log q + ...
say("   log log y = log t + (1/2) log log q + (1/2) log 1/(1-beta) >= (eps/2) log q + O(log log q + |log t|)")
say("   so log q <= (2/eps + o(1)) log log y, and the secondary term")
say("   kappa(t) y log q/(log y)^2 <= (2 kappa(t)/eps + o(1/eps)) y loglog y/(log y)^2,  2 kappa(t) = 4(1 + C^2 t^2).")
K = lambda tt, c: 4 * (1 + c**2 * tt**2)
say(f"   t = 1   : eps*K = 4(1+C^2) = {K(1, C12):.4f} (u in [1,2]),  {K(1, C3):.4f} (u = 3)")
say(f"   t = 1/C : eps*K = 8")
say(f"   t -> 0  : eps*K -> 4 (infimum, not attained)")
say(f"   printed 4C converted: eps*K = 8C in [{8*C3:.4f}, {8*C12:.4f}]")
say(f"   printed: eps*K = 2")

check("numerical: min over t in [1e-6,10], u in [1,3] of 4(1+C^2 t^2) exceeds 4 - 1e-9",
      float(np.min(4 * (1 + np.outer(Cu**2, np.logspace(-6, 1, 200)**2)))) >= 4 - 1e-9)
# check the log log y asymptotics numerically for a concrete Siegel-zero profile 1-beta = q^{-eps}
for epsv, Lq in ((0.2, 1e3), (0.1, 1e4), (0.05, 1e5)):
    lly = 0.5 * (np.log(Lq) + epsv * Lq)  # log log y_q (t = 1), 1-beta = q^{-eps}
    say(f"   eps={epsv}, log q={Lq:g}: log q / log log y_q = {Lq/lly:.4f}  vs 2/eps = {2/epsv:.4f}")
check("log q/log log y_q -> 2/eps from below (concrete profiles)", all(
    0.9 * 2 / e < Lq / (0.5 * (np.log(Lq) + e * Lq)) < 2 / e for e, Lq in ((0.2, 1e3), (0.1, 1e4), (0.05, 1e5))))

# ---------------------------------------------------------------------------
say("=" * 72)
say("3. Section 4 (pp. 17-18): w versus w^2")
say("=" * 72)
w, W, q = sp.symbols("w W q", positive=True)
# Prop. 3: S <~ 4y/(log y)^2 (log(q y/z^2) + (1-beta)(log y)^2/4 ... ) ; with log y = 2 sqrt(log q/(1-beta)),
# (1-beta)(log y)^2 = 4 log q, so (1-beta) y = 4 y log q/(log y)^2.
check("log y = 2 sqrt(log q/(1-beta))  =>  (1-beta)(log y)^2 = 4 log q",
      sp.simplify(b1 * (2 * sp.sqrt(L / b1))**2 - 4 * L) == 0)
check("y = (wz)^2: log(q w^2) + log q = 2 log(q w)  (8 log(qw) as printed)",
      sp.simplify(sp.expand_log(sp.log(q * w**2) + sp.log(q) - 2 * sp.log(q * w), force=True)) == 0)
check("y = W z^2:  log(q W) + log q = 2 log(q sqrt W)",
      sp.simplify(sp.expand_log(sp.log(q * W) + sp.log(q) - 2 * sp.log(q * sp.sqrt(W)), force=True)) == 0)
check("relation for W = w^2 is identical to the w^2-relation: 4 W log(q sqrt W) = 4 w^2 log(q w)",
      sp.simplify(sp.expand_log((4 * W * sp.log(q * sp.sqrt(W))).subs(W, w**2) - 4 * w**2 * sp.log(q * w), force=True)) == 0)
say("   Comparison with the heuristic min S ~ 2e^{-gamma}(log x)^2/log y, z ~ log x, gives")
say("   e^{-gamma} log y <~ 4 (y/z^2) log(q sqrt(y/z^2)) ; printed: e^{-gamma} log y <~ 4 w log(qw) with y = (wz)^2.")


def solve_lin(Lq, d):
    """W from e^{-g} log y = 4 W (log q + 0.5 log W), log y = 2 log q/d."""
    rhs = EG * 2 * Lq / d
    lo, hi = 1e-12, 1e12
    for _ in range(300):
        mid = np.sqrt(lo * hi)
        val = 4 * mid * (Lq + 0.5 * np.log(mid))
        lo, hi = (mid, hi) if val < rhs else (lo, mid)
    return np.sqrt(lo * hi)


def solve_printed(Lq, d):
    """w from e^{-g} log y = 4 w (log q + log w)."""
    rhs = EG * 2 * Lq / d
    lo, hi = 1e-12, 1e12
    for _ in range(300):
        mid = np.sqrt(lo * hi)
        val = 4 * mid * (Lq + np.log(mid))
        lo, hi = (mid, hi) if val < rhs else (lo, mid)
    return np.sqrt(lo * hi)


say(f"   {'log q':>7} {'delta':>8} {'W=y/z^2':>12} {'w printed':>12} {'w^2':>12} {'e^-g/(2 delta)':>15}")
rows3 = []
for Lq in (1e2, 1e4):
    for d in (1e-3, 1e-2, 1e-1):
        Wv, wv = solve_lin(Lq, d), solve_printed(Lq, d)
        rows3.append((Lq, d, Wv, wv))
        say(f"   {Lq:>7g} {d:>8g} {Wv:>12.4f} {wv:>12.4f} {wv**2:>12.2f} {EG/(2*d):>15.4f}")
check("W and printed w agree to within 3% (both ~ e^-gamma/(2 delta)); w^2 differs by orders of magnitude",
      all(abs(r[2] / r[3] - 1) < 0.03 for r in rows3) and all(r[3]**2 / r[2] > 2 for r in rows3))
# predictions: kappa-profile 1-beta = (log q)^{-kappa}: normalized gap >> (log log p)^{1-2/(kappa+1)}
kap = sp.symbols("kappa", positive=True)
d_k = sp.sqrt((L**(-kap)) * L)
loglogp = sp.sqrt(L / L**(-kap))  # log y = 2 sqrt(...) ~ 2 log log p, up to the constant
expo = sp.simplify(sp.log(1 / d_k) / sp.log(loglogp))
check("1-beta = (log q)^{-kappa}: 1/delta = (log log p)^{1-2/(kappa+1)} (prediction on p. 18 holds for W)",
      sp.simplify(expo - (1 - 2 / (kap + 1))) == 0)

# ---------------------------------------------------------------------------
say("=" * 72)
say("4. Theorem 1 (p. 10): uniformity of the induction near x' = z^{k-1}")
say("=" * 72)
say("   For k >= u both sides of Theorem 1 vanish (pi_k(z^u, z) = 0), so no hypothesis k < u is needed.")
N = 40_000_000
sieve = np.ones(N + 1, dtype=bool)
sieve[:2] = False
for p in range(2, int(N**0.5) + 1):
    if sieve[p]:
        sieve[p * p::p] = False
pi = np.cumsum(sieve)
primes = np.flatnonzero(sieve)
z = 2000
say(f"   z = {z}: relative size of an O(pi(x')) error against pi_1(x', z) = pi(x') - pi(z)")
rel = []
for eta in (1.0, 0.3, 0.1, 0.03, 0.01, 0.003):
    xp = int(z**(1 + eta))
    r = pi[xp] / (pi[xp] - pi[z])
    rel.append(r)
    say(f"     x' = z^(1+{eta}): pi(x')/pi_1(x',z) = {r:10.2f}")
check("pi(x')/pi_1(x',z) is unbounded as x' -> z+ (relative form of the k=1 case fails)", rel[-1] > 40 and rel == sorted(rel))
# the induction sum: boundary error sum_{z<p<=x/z} pi(z) is small compared with x/log z
for u in (2.1, 2.2):
    x = int(z**u)
    if x > N:
        continue
    ps = primes[(primes > z) & (primes <= x // z)]
    bsum = len(ps) * pi[z]
    say(f"     u = {u}: sum_(z<p<=x/z) pi(z) / (x/log z) = {bsum / (x / np.log(z)):.4f}   (1/log(x/z) = {1/np.log(x/z):.4f})")
check("boundary contribution sum_p pi(z) <= 1.1 (x/log z)/log(x/z), i.e. o(x/log z)",
      all(len(primes[(primes > z) & (primes <= int(z**u) // z)]) * pi[z] / (int(z**u) / np.log(z))
          * np.log(z**(u - 1)) < 1.1 for u in (2.1, 2.2) if int(z**u) <= N))

say("=" * 72)
say(f"TOTAL FAILURES: {check.failures}")

# ---------------------------------------------------------------------------
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({"font.family": "serif", "font.size": 9, "axes.labelsize": 9, "legend.fontsize": 8,
                     "xtick.labelsize": 8, "ytick.labelsize": 8, "pdf.fonttype": 42, "axes.linewidth": 0.7})
col = {"true": "#1f4e79", "opt": "#2e8b57", "print": "#c0392b", "inf": "#7f7f7f", "p4": "#e08e0b"}
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.0, 3.6))
fig.subplots_adjust(left=0.085, right=0.915, bottom=0.13, top=0.72, wspace=0.42)

h1, = ax1.plot(uu, 2 * (1 + Cu**2), color=col["true"], lw=1.6, label=r"proved at stated scale: $2(1+C(u)^2)$")
h2, = ax1.plot(uu, 4 + 0 * uu, color=col["opt"], lw=1.6, label=r"proved at optimised scale: $4$")
h3, = ax1.plot(uu, 4 * Cu, color=col["print"], lw=1.4, ls="--", label=r"printed: $4C(u)$")
h4, = ax1.plot(uu, 2 + 0 * uu, color=col["inf"], lw=1.0, ls=":", label=r"infimum over scales: $2$")
h5, = ax1.plot(uu, 1 + 0 * uu, color=col["p4"], lw=1.4, ls="-.", label=r"printed 4th bullet $2/\epsilon$ ($\kappa=1$)")
ax1.set_xlim(1, 3)
ax1.set_ylim(0, 4.6)
ax1.set_xlabel(r"$u$")
ax1.set_ylabel(r"$\kappa$ in $\kappa\,y\log q/(\log y)^2$")
sec = ax1.secondary_yaxis("right", functions=(lambda k: 2 * k, lambda k: k / 2))
sec.set_ylabel(r"$\epsilon\,K$ (4th bullet)")
ax1.text(0.04, 0.05, "(a)", transform=ax1.transAxes, ha="left", va="bottom", fontsize=10, fontweight="bold")

dd = np.logspace(-3, -1, 200)
for Lq, ls in ((1e2, "-"), (1e4, "--")):
    Wv = np.array([solve_lin(Lq, d) for d in dd])
    wv = np.array([solve_printed(Lq, d) for d in dd])
    if Lq == 1e2:
        g1, = ax2.plot(dd, Wv, color=col["true"], lw=1.6, ls=ls, label=r"$W=y/z^2$: $e^{-\gamma}\log y=4W\log(q\sqrt{W})$")
        g2, = ax2.plot(dd, wv**2, color=col["print"], lw=1.4, ls=ls,
                       label=r"$w^2$ with printed $e^{-\gamma}\log y=4w\log(qw)$, $y=(wz)^2$")
    else:
        ax2.plot(dd, Wv, color=col["true"], lw=1.2, ls=ls)
        ax2.plot(dd, wv**2, color=col["print"], lw=1.1, ls=ls)
g3, = ax2.plot(dd, EG / (2 * dd), color="k", lw=0.9, ls=":", label=r"$e^{-\gamma}/(2\delta)$")
ax2.set_xscale("log")
ax2.set_yscale("log")
ax2.set_xlim(1e-3, 1e-1)
ax2.set_xlabel(r"$\delta=\sqrt{(1-\beta_q)\log q}$")
ax2.set_ylabel(r"normalised gap $y/z^2$")
ax2.text(0.96, 0.95, "(b)", transform=ax2.transAxes, ha="right", va="top", fontsize=10, fontweight="bold")
ax2.text(0.05, 0.05, r"solid: $\log q=10^2$; dashed: $\log q=10^4$", transform=ax2.transAxes, fontsize=7,
         ha="left", va="bottom")

fig.legend(handles=[h1, h2, h3, h4, h5, g1, g2, g3], loc="upper center", ncol=2, frameon=False,
           bbox_to_anchor=(0.5, 1.0), columnspacing=1.2, handlelength=2.4, borderaxespad=0.2)
fig.savefig(OUT_PDF)
fig.savefig(OUT_PNG, dpi=200)

with open(OUT_TXT, "w", encoding="utf-8") as fh:
    fh.write("\n".join(_lines) + "\n")
sys.exit(1 if check.failures else 0)
