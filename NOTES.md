# Working notes (not for Sonia/Itza)

Things flagged during work sessions that don't belong in the colored
olive/red/blue scaffolding of the `.tex` files (that's for advisor review),
but that need to be resolved before the paper is done. Add to this file
instead of leaving loose comments in the LaTeX.

## Open

### Baseline-calibration numeric checks pulled out of `bargainigpower_draft.tex`, to reinstate once the calibration section is ported

`subsec:nash` and its two appendices had several sentences citing specific
baseline-calibration numbers ($\mu=0.5$, $\kappa=0.09$, $\underline{y}=-0.5$)
or pointing at `Table \ref{tab:params}` / `Section \ref{subsec:numericalresults}`,
neither of which exists yet in this file (they live only in
`thesis_draft.tex`, `subsec:calibration`). Removed them from the main text
so it doesn't reference sections that aren't there; the underlying checks
are all still true and should go back in once the calibration table and
numerical-results section are ported. Keeping the numbers here so nothing
is lost:

- **Concavity Proposition** (after the Pareto-frontier-curvature block in
  `subsec:nash`): at baseline calibration, $\mu=0.5>\kappa=0.09$, so the
  *unconditional* case of the Proposition applies — the constrained frontier
  is concave everywhere in the binding regime, not merely at the reported
  equilibrium range.
- **Assumption `ass:beta_bounds_reg`** (Appendix `app:beta_bounds`): both its
  inequality display and the existence/nonemptiness clause (what
  `feasible_beta_bracket` in `model.py` checks) hold at baseline calibration
  for representative $W_t$ in the binding domain.
- **Proposition `prop:lambda_dynamics`** (Appendix `app:lambda_dynamics`):
  at baseline calibration, $W_t=0.5$: $\beta_t^*=0.7464$,
  $\Pi_t'(\beta_t^*)=-0.7464$, $\text{CE}_t'(\beta_t^*)=+0.8060$,
  $d\beta_t^*/d\underline w_t=-1.2146$, $d\lambda_t/d\underline w_t=+0.2457$,
  so condition \eqref{eq:lambda_sign_condition} holds. Verified against a
  finite-difference derivative of `solve_beta_static` and `lambda_realized`
  in `model.py`, matching the closed form to 8 significant figures. Chain
  rule gives $d\beta_t^*/dW_t\approx+0.243$, $d\lambda_t/dW_t\approx-0.049$,
  consistent with $\lambda(W)$ falling from $\approx0.4446$ at low $W$ to
  $\delta=0.40$ at $\overline W$ (the numerical-results section this
  ultimately belongs next to hasn't been ported either).

### `bargainigpower_draft.tex` uses $\overline{W}$ and the wealth process without the dynamic environment that formally grounds them

`subsec:nash` now defines $\overline{W}$ at the static level, as the wealth
where the Case-1 (slack) fixed payment $\alpha_t^*$ exactly satisfies limited
liability: $\overline{W} = (\overline{w}-\alpha_t^*-\beta^{\text{eff}}\underline{y})/\gamma_w$.
That's enough to make every later use of $\overline{W}$ in this file legible
(Remark `rem:welfare`, Appendix `app:beta_bounds`, Appendix
`app:lambda_dynamics`).

What's still missing, and only exists in `thesis_draft.tex`: the actual
recursive/dynamic setup, the wealth law of motion $W_{t+1}=W_t+\dots+\beta_t^*\sigma\varepsilon_t$,
continuation value functions $V_P,V_A$, the state space $\mathcal S=[\underline W,W_{\max}]$,
and the ergodic-distribution results. Appendix `app:lambda_dynamics` already
leans on $W_{t+1}-W_t=\beta_t^*\sigma\varepsilon_t$ and treats $\lambda_t=\lambda(W_t)$
as an established object, and the boundedness appendix (line ~606) explicitly
flags that its sup/inf-over-$W$ claim needs "the continuity-in-$W$ and
compactness argument that will be available once the dynamic environment...
[is] added." None of that environment has been ported into
`bargainigpower_draft.tex` yet (see `thesis_draft.tex` around
`Definition \ref{def:recursive_nash}`, the budget-constraint law of motion,
and Proposition `prop:ergodic`). Until it's ported, the dynamic appendices
rest on borrowed intuition rather than a definition stated in this file.

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

Olive notes still sit at the three affected spots in `thesis_draft.tex`
(Theorem `thm:central`, Appendix `ProofProp2`, Figure `fig:wealth_dynamics`)
as the advisor-facing flags; this entry stays here as the durable reminder
that the thesis text itself has not been corrected yet, only the ported
version in `bargainigpower_draft.tex`.

## Resolved
(move items here once fixed, with the commit that fixed them)

### Appendix B / $\lambda'(W_t)$ sign, ported into `bargainigpower_draft.tex`

Rederived the $d\beta^*/d\underline w$ and $d\lambda/d\underline w$ argument
from the FOC alone (Proposition `prop:lambda_dynamics`, new Appendix
`app:lambda_dynamics`): $\Pi_t'(\beta^*)$ and $\text{CE}_t'(\beta^*)$ always
have opposite signs, and for $\mu\ge\kappa$, Lemma `beta_bounds` gives
$\text{CE}_t'(\beta^*)>0$ unconditionally, so $\Pi_t'(\beta^*)<0$. Redoing the
quotient-rule algebra for $d\lambda/d\underline w$ (the thesis's simplified
form of the numerator, $\text{CE}_t'(\beta^*)[\Pi_s+\frac{\delta}{1-\delta}\text{CE}_s]$,
was itself an algebra slip independent of the sign issue; the correct
simplification is $\Pi_s\,\text{CE}_t'(\beta^*)/(1-\delta)$) gives
$d\beta^*/d\underline w<0$ and, under a sufficient condition analogous to
the old one, $d\lambda/d\underline w>0$, hence $\lambda'(W_t)<0$: $\lambda_t>\delta$
throughout the binding regime, descending to $\delta$ at $\overline W$.

Verified numerically against `model.py` at $W=0.5$ (baseline calibration):
closed form and finite-difference derivatives of `solve_beta_static` and
`lambda_realized` agree to 8 significant figures ($d\beta^*/d\underline
w=-1.2146$, $d\lambda/d\underline w=+0.2457$), and imply $d\beta^*/dW\approx
+0.243$, $d\lambda/dW\approx-0.049$, matching the numbers already reported
in Section 5 and in the old note above. Fixed in commit that added Appendix
`app:lambda_dynamics` and redrew Figure `fig:wealth_dynamics` (curve now
starts above $\delta$ and descends).

`thesis_draft.tex` itself (Theorem `thm:central`, Appendix `ProofProp2`,
Figure `fig:wealth_dynamics`) still has the old, wrong-signed version; only
`bargainigpower_draft.tex` has been corrected so far.
