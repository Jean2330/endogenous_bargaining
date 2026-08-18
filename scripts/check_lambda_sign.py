"""
Verifies the sign of d(lambda)/dW in the binding regime against the closed-form
claim in Appendix B (thm:central / prop:dynamics) of the thesis, using only the
static Nash FOC machinery in src/model.py (no VFI needed, since the sign
question is a static comparative-static claim).

Finding (baseline calibration): CE'(beta*) > 0, Pi'(beta*) < 0, and
d(lambda)/dW < 0 -- i.e. lambda falls toward delta as W rises in the binding
regime. This is the OPPOSITE sign to Theorem 3.10 / Prop 3.9 / Fig 3.7 in the
thesis, but matches Section 5's own numerical results.
"""
import sys
sys.path.insert(0, "src")
from model import (beta_eff, disagreement_payoffs, w_bar_threshold,
                    solve_beta_static, lambda_realized, CE_prime, Pi_prime,
                    w_lower)
import params as P

theta, k, gamma, sigma = P.THETA, P.K, P.GAMMA, P.SIGMA
delta = P.DELTA
w_bar_ll, gamma_w, y_lower = P.W_BAR_LL, P.GAMMA_W, P.Y_LOWER

u_p_bar, ce_bar = disagreement_payoffs(theta, k, gamma, sigma, delta, P.FRAC_DISAGREEMENT)
W_bar = w_bar_threshold(delta, theta, k, gamma, sigma, w_bar_ll, gamma_w, y_lower, u_p_bar, ce_bar)

mu, kappa = theta**2 / k, gamma * sigma**2
print(f"mu={mu}, kappa={kappa}, beta_eff={beta_eff(theta,k,gamma,sigma):.4f}, W_bar={W_bar:.4f}")
print(f"mu >= kappa: {mu >= kappa}  (unconditional CE'(beta)>0 case, per the frontier-concavity result)")

print("\nW      beta*(W)   lambda(W)")
for W in [0.05, 0.25, 0.5, 0.75, 0.90, W_bar - 1e-3]:
    b = solve_beta_static(W, delta, w_bar_ll, gamma_w, y_lower, theta, k, gamma, sigma, u_p_bar, ce_bar)
    lam = lambda_realized(b, W, delta, w_bar_ll, gamma_w, y_lower, theta, k, gamma, sigma, u_p_bar, ce_bar)
    print(f"{W:6.3f}  {b:8.4f}   {lam:8.4f}")

h, W0 = 1e-4, 0.5
def lam_beta(W):
    b = solve_beta_static(W, delta, w_bar_ll, gamma_w, y_lower, theta, k, gamma, sigma, u_p_bar, ce_bar)
    return lambda_realized(b, W, delta, w_bar_ll, gamma_w, y_lower, theta, k, gamma, sigma, u_p_bar, ce_bar), b

lam_p, b_p = lam_beta(W0 + h)
lam_m, b_m = lam_beta(W0 - h)
b0 = solve_beta_static(W0, delta, w_bar_ll, gamma_w, y_lower, theta, k, gamma, sigma, u_p_bar, ce_bar)
print(f"\nAt W={W0}: beta*={b0:.4f}")
print(f"  CE'(beta*) = {CE_prime(b0, theta, k, gamma, sigma, y_lower):+.4f}")
print(f"  Pi'(beta*) = {Pi_prime(b0, theta, k, y_lower):+.4f}")
print(f"  d(lambda)/dW (finite diff) = {(lam_p - lam_m) / (2*h):+.6f}")
print(f"  d(beta*)/dW  (finite diff) = {(b_p - b_m) / (2*h):+.6f}")
