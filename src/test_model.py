# Checks:
#   Closed-form formulas match direct calculation
#   Model properties that must hold analytically
#   Calibration: feasible set is non-empty in the binding regime

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
import params as P
from model import (beta_eff, surplus, s_fb, s_eff, w_lower,
                   disagreement_payoffs, w_bar_threshold,
                   Pi_binding, CE_binding,
                   solve_beta_static, lambda_realized,
                   bargaining_distortion, nash_foc_static)


# ---------------------------------------------------------------------------
# Shared baseline fixture
# ---------------------------------------------------------------------------

def baseline():
    u_p, u_a = disagreement_payoffs(P.THETA, P.K, P.GAMMA, P.SIGMA,
                                     P.DELTA, P.FRAC_DISAGREEMENT)
    return dict(
        theta=P.THETA, k=P.K, gamma=P.GAMMA, sigma=P.SIGMA,
        delta=P.DELTA, w_bar_ll=P.W_BAR_LL, gamma_w=P.GAMMA_W,
        y_lower=P.Y_LOWER, rho=P.RHO,
        u_p_bar=u_p, u_a_bar=u_a,
    )


# ---------------------------------------------------------------------------
# (i) Closed-form formula checks
# ---------------------------------------------------------------------------

def test_beta_eff_formula():
    """beta_eff = (theta^2/k) / (gamma*sigma^2 + theta^2/k)."""
    p = baseline()
    num = p['theta']**2 / p['k']
    den = p['gamma'] * p['sigma']**2 + p['theta']**2 / p['k']
    assert abs(beta_eff(p['theta'], p['k'], p['gamma'], p['sigma']) - num/den) < 1e-12


def test_s_fb_formula():
    """S_FB = theta^2 / (2k)."""
    p = baseline()
    assert abs(s_fb(p['theta'], p['k']) - p['theta']**2 / (2*p['k'])) < 1e-12


def test_s_eff_less_than_fb():
    """Moral hazard strictly reduces surplus relative to first-best."""
    p = baseline()
    assert s_eff(p['theta'], p['k'], p['gamma'], p['sigma']) \
           < s_fb(p['theta'], p['k'])


def test_disagreement_payoffs_fractions():
    """u_p + u_a = frac * S_eff."""
    p = baseline()
    u_p, u_a = disagreement_payoffs(p['theta'], p['k'], p['gamma'], p['sigma'],
                                     p['delta'], P.FRAC_DISAGREEMENT)
    S = s_eff(p['theta'], p['k'], p['gamma'], p['sigma'])
    assert abs(u_p + u_a - P.FRAC_DISAGREEMENT * S) < 1e-12


def test_w_lower_decreases_with_W():
    """LL floor w_lower(W) is decreasing in W: wealthier agents face a lower floor."""
    p = baseline()
    assert w_lower(2.0, p['w_bar_ll'], p['gamma_w']) \
           < w_lower(1.0, p['w_bar_ll'], p['gamma_w'])


# ---------------------------------------------------------------------------
# (ii) Analytical model properties
# ---------------------------------------------------------------------------

def test_bargaining_distortion_zero_at_beta_eff():
    """Delta(W) = 0 when beta* = beta_eff."""
    p = baseline()
    b = beta_eff(p['theta'], p['k'], p['gamma'], p['sigma'])
    assert abs(bargaining_distortion(b, p['theta'], p['k'],
                                     p['gamma'], p['sigma'])) < 1e-12


def test_bargaining_distortion_positive_away():
    """Delta(W) > 0 when beta* != beta_eff."""
    p = baseline()
    b_off = beta_eff(p['theta'], p['k'], p['gamma'], p['sigma']) * 0.5
    assert bargaining_distortion(b_off, p['theta'], p['k'],
                                  p['gamma'], p['sigma']) > 0


def test_w_bar_threshold_positive():
    """Regime boundary W_bar must be strictly positive."""
    p = baseline()
    W_bar = w_bar_threshold(p['delta'], p['theta'], p['k'],
                             p['gamma'], p['sigma'],
                             p['w_bar_ll'], p['gamma_w'],
                             p['y_lower'], p['u_p_bar'], p['u_a_bar'])
    assert W_bar > 0


def test_foc_sign_change():
    """
    Nash FOC must change sign on the feasible interval at W_bar/2.
    By Lemma C.1, G > 0 at the lower end and G < 0 at the upper end
    of the feasible beta range, guaranteeing a unique root for brentq.
    """
    p = baseline()
    W_bar = w_bar_threshold(p['delta'], p['theta'], p['k'],
                             p['gamma'], p['sigma'],
                             p['w_bar_ll'], p['gamma_w'],
                             p['y_lower'], p['u_p_bar'], p['u_a_bar'])
    W_test = W_bar * 0.5

    # Find the feasible beta range at W_test
    betas = np.linspace(0.05, 0.95, 200)
    feasible = []
    for bv in betas:
        Pi = Pi_binding(bv, W_test, p['w_bar_ll'], p['gamma_w'],
                        p['y_lower'], p['theta'], p['k'], p['u_p_bar'])
        CE = CE_binding(bv, W_test, p['w_bar_ll'], p['gamma_w'],
                        p['y_lower'], p['theta'], p['k'], p['gamma'],
                        p['sigma'], p['u_a_bar'])
        if Pi > 0 and CE > 0:
            feasible.append(bv)

    assert len(feasible) >= 5, \
        f"Feasible set too small at W_bar/2 -- only {len(feasible)} feasible betas"

    beta_lo = feasible[0]  + 1e-4
    beta_hi = feasible[-1] - 1e-4

    foc_lo = nash_foc_static(beta_lo, W_test, p['delta'], p['w_bar_ll'],
                              p['gamma_w'], p['y_lower'], p['theta'], p['k'],
                              p['gamma'], p['sigma'], p['u_p_bar'], p['u_a_bar'])
    foc_hi = nash_foc_static(beta_hi, W_test, p['delta'], p['w_bar_ll'],
                              p['gamma_w'], p['y_lower'], p['theta'], p['k'],
                              p['gamma'], p['sigma'], p['u_p_bar'], p['u_a_bar'])

    assert foc_lo > 0, \
        f"FOC at lower feasible beta={beta_lo:.3f} should be positive, got {foc_lo:.4f}"
    assert foc_hi < 0, \
        f"FOC at upper feasible beta={beta_hi:.3f} should be negative, got {foc_hi:.4f}"


# ---------------------------------------------------------------------------
# (iii) Calibration: feasible set non-empty at W_bar/2
# ---------------------------------------------------------------------------

def test_feasible_set_nonempty_in_binding():
    """
    At W = W_bar/2, there must exist at least one beta where both
    Pi_s > 0 and CE_s > 0.  If this fails, the calibration is wrong.
    """
    p = baseline()
    W_bar = w_bar_threshold(p['delta'], p['theta'], p['k'],
                             p['gamma'], p['sigma'],
                             p['w_bar_ll'], p['gamma_w'],
                             p['y_lower'], p['u_p_bar'], p['u_a_bar'])
    W_test = W_bar * 0.5

    betas = np.linspace(0.05, 0.95, 50)
    feasible = []
    for b in betas:
        Pi = Pi_binding(b, W_test, p['w_bar_ll'], p['gamma_w'],
                        p['y_lower'], p['theta'], p['k'], p['u_p_bar'])
        CE = CE_binding(b, W_test, p['w_bar_ll'], p['gamma_w'],
                        p['y_lower'], p['theta'], p['k'], p['gamma'],
                        p['sigma'], p['u_a_bar'])
        feasible.append(Pi > 0 and CE > 0)

    assert any(feasible), \
        "No feasible beta found at W_bar/2 -- check W_BAR_LL and GAMMA_W in params.py"


def test_nash_solution_positive_surpluses():
    """The static Nash solution must yield positive surpluses for both parties."""
    p = baseline()
    W_bar = w_bar_threshold(p['delta'], p['theta'], p['k'],
                             p['gamma'], p['sigma'],
                             p['w_bar_ll'], p['gamma_w'],
                             p['y_lower'], p['u_p_bar'], p['u_a_bar'])
    W_test = W_bar * 0.5

    beta_opt = solve_beta_static(W_test, p['delta'], p['w_bar_ll'],
                                  p['gamma_w'], p['y_lower'],
                                  p['theta'], p['k'], p['gamma'], p['sigma'],
                                  p['u_p_bar'], p['u_a_bar'])
    Pi = Pi_binding(beta_opt, W_test, p['w_bar_ll'], p['gamma_w'],
                    p['y_lower'], p['theta'], p['k'], p['u_p_bar'])
    CE = CE_binding(beta_opt, W_test, p['w_bar_ll'], p['gamma_w'],
                    p['y_lower'], p['theta'], p['k'], p['gamma'],
                    p['sigma'], p['u_a_bar'])

    assert Pi > 0, f"Principal surplus Pi_s={Pi:.4f} at Nash solution"
    assert CE > 0, f"Agent surplus CE_s={CE:.4f} at Nash solution"
