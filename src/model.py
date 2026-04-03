# model.py
# Core model objects: payoff functions, regime identification, Nash FOC.
# All functions take parameters explicitly so they can be called with
# non-baseline values during comparative statics.

import numpy as np
from scipy.optimize import brentq


# ---------------------------------------------------------------------------
# Closed-form benchmarks
# ---------------------------------------------------------------------------

def beta_eff(theta, k, gamma, sigma):
    """Efficient incentive slope under moral hazard (unconstrained Nash)."""
    return (theta**2 / k) / (gamma * sigma**2 + theta**2 / k)


def a_star(beta, theta, k):
    """Incentive-compatible effort given slope beta."""
    return beta * theta / k


def s_fb(theta, k):
    """First-best total surplus."""
    return theta**2 / (2 * k)


def s_eff(theta, k, gamma, sigma):
    """Second-best (moral hazard only) total surplus at beta_eff."""
    b = beta_eff(theta, k, gamma, sigma)
    return surplus(b, theta, k, gamma, sigma)


# ---------------------------------------------------------------------------
# Per-period payoff components
# ---------------------------------------------------------------------------

def C(beta, theta, k, gamma, sigma):
    """Agent's net payoff from slope beta (excluding fixed payment alpha)."""
    return (beta**2 * theta**2) / (2 * k) - (gamma / 2) * beta**2 * sigma**2


def D(beta, theta, k):
    """Principal's net payoff from slope beta (excluding fixed payment alpha)."""
    return (beta * theta**2 / k) * (1 - beta)


def surplus(beta, theta, k, gamma, sigma):
    """Total surplus S(beta) = C(beta) + D(beta)."""
    return C(beta, theta, k, gamma, sigma) + D(beta, theta, k)


def w_lower(W, w_bar, gamma_w):
    """
    Wealth-dependent LL floor: w_bar - gamma_w * W.
    Decreasing in W: wealthier agents face a lower (less binding) floor,
    consistent with wealth serving as collateral or commitment capacity.
    """
    return w_bar - gamma_w * W


def disagreement_payoffs(theta, k, gamma, sigma, delta, frac):
    """
    Fixed outside options as fractions of each party's Nash share of S_eff.
    u_p_bar = frac * (1-delta) * S_eff
    u_a_bar = frac * delta     * S_eff
    These are constants (not wealth-dependent), avoiding perpetuity magnification.
    """
    b = beta_eff(theta, k, gamma, sigma)
    S = surplus(b, theta, k, gamma, sigma)
    u_p = frac * (1 - delta) * S
    u_a = frac * delta * S
    return u_p, u_a


# ---------------------------------------------------------------------------
# Regime identification
# ---------------------------------------------------------------------------

def w_bar_threshold(delta, theta, k, gamma, sigma, w_bar_ll, gamma_w,
                    y_lower, u_p_bar, u_a_bar):
    """
    Compute the wealth threshold W_bar above which LL is slack.
    W_bar solves: alpha_slack(W_bar) = w_lower(W_bar) - beta_eff * y_lower
    where alpha_slack is the unconstrained Nash fixed payment.
    u_a_bar: fixed agent disagreement payoff (scalar).
    """
    b = beta_eff(theta, k, gamma, sigma)
    s = surplus(b, theta, k, gamma, sigma)

    def residual(W):
        alpha_slack = u_a_bar - C(b, theta, k, gamma, sigma) + delta * s
        ll_floor = w_lower(W, w_bar_ll, gamma_w) - b * y_lower
        return alpha_slack - ll_floor

    W_low, W_high = 0.0, 50.0
    try:
        W_bar = brentq(residual, W_low, W_high, xtol=1e-8)
    except ValueError:
        W_bar = W_high
    return W_bar


def is_binding(W, W_bar):
    """True if limited liability binds at wealth W."""
    return W < W_bar


# ---------------------------------------------------------------------------
# Static Nash FOC (no continuation values)
# Used in Phase 1 and as a warm-start for VFI.
# ---------------------------------------------------------------------------

def Pi_binding(beta, W, w_bar_ll, gamma_w, y_lower, theta, k, u_p_bar):
    """Principal surplus above disagreement, binding regime."""
    wl = w_lower(W, w_bar_ll, gamma_w)
    return D(beta, theta, k) + beta * y_lower - wl - u_p_bar


def CE_binding(beta, W, w_bar_ll, gamma_w, y_lower, theta, k, gamma, sigma,
               ce_bar):
    """Agent surplus above disagreement, binding regime."""
    wl = w_lower(W, w_bar_ll, gamma_w)
    return C(beta, theta, k, gamma, sigma) - beta * y_lower + wl - ce_bar


def Pi_prime(beta, theta, k, y_lower):
    """Derivative of Pi_binding with respect to beta."""
    return (theta**2 / k) * (1 - 2 * beta) + y_lower


def CE_prime(beta, theta, k, gamma, sigma, y_lower):
    """Derivative of CE_binding with respect to beta."""
    return (beta * theta**2 / k) - gamma * beta * sigma**2 - y_lower


def nash_foc_static(beta, W, delta, w_bar_ll, gamma_w, y_lower, theta, k,
                    gamma, sigma, u_p_bar, ce_bar):
    """
    Nash FOC in the static binding regime. Returns zero at beta*.
    Sign: positive at beta=0, negative at beta=1 (by Lemma C.1).
    """
    Pi_s = Pi_binding(beta, W, w_bar_ll, gamma_w, y_lower, theta, k, u_p_bar)
    CE_s = CE_binding(beta, W, w_bar_ll, gamma_w, y_lower, theta, k, gamma,
                      sigma, ce_bar)

    if Pi_s <= 0 or CE_s <= 0:
        return np.nan

    dPi = Pi_prime(beta, theta, k, y_lower)
    dCE = CE_prime(beta, theta, k, gamma, sigma, y_lower)

    return (1 - delta) * dPi / Pi_s + delta * dCE / CE_s


def solve_beta_static(W, delta, w_bar_ll, gamma_w, y_lower, theta, k, gamma,
                      sigma, u_p_bar, ce_bar):
    """
    Solve the static Nash FOC for beta*(W) using Brent's method.
    Automatically detects the feasible interval where Pi_s > 0 and CE_s > 0,
    then finds the root of the FOC within that interval.
    Returns beta* in (0,1), or beta_eff if no feasible interior root exists.
    """
    b_eff = beta_eff(theta, k, gamma, sigma)

    # Scan for the feasible interval
    betas = np.linspace(0.02, 0.98, 500)
    feasible = []
    for bv in betas:
        Pi = Pi_binding(bv, W, w_bar_ll, gamma_w, y_lower, theta, k, u_p_bar)
        CE = CE_binding(bv, W, w_bar_ll, gamma_w, y_lower, theta, k, gamma,
                        sigma, ce_bar)
        if Pi > 0 and CE > 0:
            feasible.append(bv)

    if len(feasible) < 2:
        return b_eff

    lo = feasible[0]  + 1e-6
    hi = feasible[-1] - 1e-6

    f = lambda b: nash_foc_static(b, W, delta, w_bar_ll, gamma_w, y_lower,
                                  theta, k, gamma, sigma, u_p_bar, ce_bar)
    try:
        foc_lo = f(lo)
        foc_hi = f(hi)
        if np.isnan(foc_lo) or np.isnan(foc_hi):
            return b_eff
        if foc_lo * foc_hi > 0:
            # No sign change: return the endpoint with smaller |FOC|
            return lo if abs(foc_lo) < abs(foc_hi) else hi
        beta_opt = brentq(f, lo, hi, xtol=1e-10)
    except ValueError:
        beta_opt = b_eff
    return beta_opt


# ---------------------------------------------------------------------------
# Realized bargaining power
# ---------------------------------------------------------------------------

def lambda_realized(beta, W, delta, w_bar_ll, gamma_w, y_lower, theta, k,
                    gamma, sigma, u_p_bar, ce_bar):
    """
    Realized bargaining power lambda = CE_s / (Pi_s + CE_s).
    In slack regime this equals delta exactly.
    """
    Pi_s = Pi_binding(beta, W, w_bar_ll, gamma_w, y_lower, theta, k, u_p_bar)
    CE_s = CE_binding(beta, W, w_bar_ll, gamma_w, y_lower, theta, k, gamma,
                      sigma, ce_bar)
    S = Pi_s + CE_s
    if S <= 0:
        return np.nan
    return CE_s / S


def logit(lam):
    """Log-odds transformation of bargaining power."""
    lam = np.clip(lam, 1e-10, 1 - 1e-10)
    return np.log(lam / (1 - lam))


# ---------------------------------------------------------------------------
# Pareto frontier computation
# ---------------------------------------------------------------------------

def pareto_frontier_slack(theta, k, gamma, sigma, u_p_bar, ce_bar,
                          n_points=300):
    """
    Compute the slack-regime Pareto frontier.
    beta = beta_eff is fixed; alpha varies freely.
    As alpha increases by one unit, CE_s rises by 1 and Pi_s falls by 1.
    Returns arrays (Pi_s, CE_s) tracing the linear frontier.
    """
    b = beta_eff(theta, k, gamma, sigma)
    # Total surplus available
    S = surplus(b, theta, k, gamma, sigma)
    # Extreme points: all surplus to principal (CE_s = 0) and all to agent
    # CE_s ranges from 0 to S; Pi_s = S - CE_s
    CE_s = np.linspace(0.0, S, n_points)
    Pi_s = S - CE_s
    return Pi_s, CE_s


def pareto_frontier_binding(W, w_bar_ll, gamma_w, y_lower, theta, k, gamma,
                             sigma, u_p_bar, ce_bar, n_points=300):
    """
    Compute the binding-regime Pareto frontier.
    alpha is pinned: alpha = w_lower(W) - beta * y_lower.
    As beta varies over (0, 1), we trace out (Pi_s(beta), CE_s(beta)).
    Returns arrays (Pi_s, CE_s, beta_vals) for the feasible portion where
    both surpluses are non-negative.
    """
    beta_vals = np.linspace(1e-4, 1 - 1e-4, n_points)
    Pi_s = np.array([Pi_binding(b, W, w_bar_ll, gamma_w, y_lower, theta, k,
                                u_p_bar) for b in beta_vals])
    CE_s = np.array([CE_binding(b, W, w_bar_ll, gamma_w, y_lower, theta, k,
                                gamma, sigma, ce_bar) for b in beta_vals])

    # Keep only the portion where both surpluses are strictly positive
    feasible = (Pi_s > 0) & (CE_s > 0)
    return Pi_s[feasible], CE_s[feasible], beta_vals[feasible]


def nash_solution_slack(delta, theta, k, gamma, sigma, u_p_bar, ce_bar):
    """
    Nash solution in the slack regime.
    Returns (Pi_s*, CE_s*): agent gets share delta of total surplus.
    """
    b = beta_eff(theta, k, gamma, sigma)
    S = surplus(b, theta, k, gamma, sigma)
    CE_s = delta * S
    Pi_s = (1 - delta) * S
    return Pi_s, CE_s


def nash_solution_binding(W, delta, w_bar_ll, gamma_w, y_lower, theta, k,
                           gamma, sigma, u_p_bar, ce_bar):
    """
    Nash solution in the binding regime.
    Solves the static FOC and returns (Pi_s*, CE_s*, beta*).
    """
    beta_opt = solve_beta_static(W, delta, w_bar_ll, gamma_w, y_lower,
                                  theta, k, gamma, sigma, u_p_bar, ce_bar)
    Pi_s = Pi_binding(beta_opt, W, w_bar_ll, gamma_w, y_lower, theta, k, u_p_bar)
    CE_s = CE_binding(beta_opt, W, w_bar_ll, gamma_w, y_lower, theta, k,
                      gamma, sigma, ce_bar)
    return Pi_s, CE_s, beta_opt


def iso_nash_curve(Pi_s_star, CE_s_star, delta, n_points=200):
    """
    Trace the iso-Nash-product curve passing through (Pi_s*, CE_s*).
    The Nash product N = Pi_s^(1-delta) * CE_s^delta is constant on this curve.
    Returns (Pi_s, CE_s) arrays for the curve, parametrized by Pi_s.
    """
    N_star = (Pi_s_star ** (1 - delta)) * (CE_s_star ** delta)
    # CE_s = (N_star / Pi_s^(1-delta))^(1/delta)
    Pi_max = Pi_s_star * 4.0
    Pi_vals = np.linspace(1e-4, Pi_max, n_points)
    with np.errstate(invalid='ignore', divide='ignore'):
        CE_vals = (N_star / Pi_vals ** (1 - delta)) ** (1 / delta)
    valid = np.isfinite(CE_vals) & (CE_vals > 0)
    return Pi_vals[valid], CE_vals[valid]


# ---------------------------------------------------------------------------
# Welfare distortion
# ---------------------------------------------------------------------------

def bargaining_distortion(beta_opt, theta, k, gamma, sigma):
    """
    Delta(W) = S_eff - S(beta*).
    Uses the quadratic approximation from Remark 3.1.
    """
    b_eff = beta_eff(theta, k, gamma, sigma)
    curvature = theta**2 / k + gamma * sigma**2
    return 0.5 * (b_eff - beta_opt)**2 * curvature
