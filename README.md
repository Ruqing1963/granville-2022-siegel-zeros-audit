# A note on the secondary terms and the prime-gap heuristic in Granville's "Sieving intervals and Siegel zeros"

Ruqing Chen (2026). doi:[10.5281/zenodo.24296574](https://doi.org/10.5281/zenodo.24296574)

A. Granville, *Sieving intervals and Siegel zeros*, Acta Arith. 204 (2022), 297–315
(doi:[10.4064/aa201002-25-6](https://doi.org/10.4064/aa201002-25-6)), shows that under infinitely many Siegel zeros
the linear-sieve bounds are attained by intervals, and gives explicit secondary terms near the sifting limit.
This note checks those secondary terms, the prime-gap heuristic of Section 4, and the induction for Theorem 1.

* **Proposition 2.** The last line of the proof uses `sqrt((1-beta) log q) = log q/log y`, but at the scale chosen
  there it equals `log q/(C(u) log y)`. Correctly, the secondary term is `4 y log q/(log y)^2` at that scale, and
  `2(1+C(u)^2) y log q/(log y)^2` at the scale `log y_q = (log q/(1-beta))^(1/2)` of the statement; both exceed the
  printed `4C(u) y log q/(log y)^2` (by 47% at u = 3 in the second case). A corrected Proposition 2' is given for
  every scale `log y = t log y_q`, with coefficient `2(1 + C(u)^2 t^2)`.
* **Fourth bullet (1-beta <= q^-eps).** The argument gives `4(1+C^2)/eps` (t = 1), `8/eps` (t = 1/C), and at best
  `(4+o(1))/eps` (t -> 0), never the printed `2/eps`. The first three bullets stand.
* **Section 4.** `w` is defined by `y = (wz)^2` but used as `y/z^2`. With `W = y/z^2` throughout, the printed
  bound and all prime-gap predictions hold.
* **Theorem 1.** The induction uses a relative error near `x' = z^(k-1)`, where it fails; an absolute, uniform
  version is given. Theorem 1 and Corollary 5 stand. (No hypothesis `k < u` is needed: for `k >= u` both sides vanish.)

Page numbers refer to the Online First pagination (pp. 1–19 = pp. 297–315 of the print volume).

| path | content |
|---|---|
| `granville_sieving_siegel_zeros_audit.tex`, `.pdf` | the note (4 pages) |
| `code/verify_granville_2022_siegel.py` | symbolic checks (sympy), the w/W equations, prime-count checks for Theorem 1, and the figure |
| `data/verify_granville_2022_siegel.txt` | output of the script |
| `figures/fig_granville_2022_audit.pdf`, `.png` | Figure 1 |

Requirements: Python 3 with numpy, sympy, matplotlib. Run `python code/verify_granville_2022_siegel.py`.
