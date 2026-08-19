# Working notes (not for Sonia/Itza)

Things flagged during work sessions that don't belong in the colored
olive/red/blue scaffolding of the `.tex` files (that's for advisor review),
but that need to be resolved before the paper is done. Add to this file
instead of leaving loose comments in the LaTeX.

## Open

### `lambda'(W)` sign is backwards in the dynamic section (thesis_draft.tex)

Appendix B (`ProofProp2`) assumes $\mathrm{CE}_t'(\beta^*) < 0$ to sign
$d\beta^*/d\underline{w}$ and, downstream, $\lambda'(W_t)$ in Theorem 3.10
(`thm:central`) and Figure 3.7 (`fig:wealth_dynamics`). That premise is false
at the baseline calibration: $\mu = \theta^2/k = 0.5 > \kappa = \gamma\sigma^2
= 0.09$, so $\mathrm{CE}_t'(\beta^*) > 0$ unconditionally (same fact used in
the Pareto-frontier-concavity result in `bargainigpower_draft.tex`).

Confirmed numerically with `src/model.py` (see `scripts/check_lambda_sign.py`):
at the baseline calibration, $\lambda(W)$ **decreases** from $\approx0.4446$
at low $W$ to $\delta=0.40$ at $\overline W$ — i.e. $\lambda'(W) < 0$ in the
binding regime, not $>0$. This matches Section 5's own numerical results
(which solve the FOC directly, not through the closed-form Appendix B
algebra) but contradicts Theorem 3.10 / Prop 3.9 / Fig 3.7's stated direction.

**Story flips from:** "wealthier agents gain more bargaining power, converging
up to $\delta$" **to:** "wealth-constrained agents extract *more* than their
Nash share $\delta$ (because $\beta$ is pushed away from $\beta^{\text{eff}}$
to satisfy their tight LL floor), converging *down* to $\delta$ as wealth
grows."

**When we port the dynamic section into `bargainigpower_draft.tex`:**
rederive Appendix B in general (mirror the two-case $\mu \gtrless \kappa$
structure used for the frontier-concavity proposition), fix the sign in
eq. (dlambda_dw), condition (lambda_sign_condition), Theorem 3.10's
statement, and redraw Figure 3.7 (curve should start above $\delta$ and
descend to it, not rise from below).

Olive notes already sit at the three affected spots in `thesis_draft.tex`
(Theorem `thm:central`, Appendix `ProofProp2`, Figure `fig:wealth_dynamics`)
— those stay as the advisor-facing flags; this entry is just the durable
reminder for us.

## Resolved
(move items here once fixed, with the commit that fixed them)
