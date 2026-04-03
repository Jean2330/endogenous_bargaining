# vfi.py
# Value function iteration for the dynamic Nash bargaining model.
# Returns converged value functions and policy functions on the wealth grid.

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.optimize import brentq

from model import (beta_eff, C, D, surplus, w_lower,
                   Pi_prime, CE_prime, bargaining_distortion)


# ---------------------------------------------------------------------------
# Gauss-Hermite quadrature
# ---------------------------------------------------------------------------

def gauss_hermite_weights(n):
    """
    Nodes and weights for integrating f(x) against N(0,1).
    numpy's hermegauss gives nodes/weights for the probabilist's Hermite
    polynomials, i.e. for the weight exp(-x^2/2)/sqrt(2*pi), so the
    weights already sum to 1 and we use the nodes directly as standard
    normal draws.
    """
    nodes, weights = np.polynomial.hermite_e.hermegauss(n)
    return nodes, weights


# ---------------------------------------------------------------------------
# Expected continuation value via quadrature
# ---------------------------------------------------------------------------

def expected_continuation(W, beta, alpha, VP_spline, VA_spline,
                           nodes, weights, theta, k, sigma, W_min, W_max):
    """
    E[VP(W')] and E[VA(W')] by Gauss-Hermite quadrature.
    W' = W + alpha + beta*y - (k/2)*a*^2, y = theta*a* + sigma*eps
    with a* = beta*theta/k, eps ~ N(0,1).
    """
    a = beta * theta / k
    drift = W + alpha + beta * theta * a - (k / 2) * a**2
    W_next = np.clip(drift + beta * sigma * nodes, W_min, W_max)
    EV_P = float(np.dot(weights, VP_spline(W_next)))
    EV_A = float(np.dot(weights, VA_spline(W_next)))
    return EV_P, EV_A


# ---------------------------------------------------------------------------
# Dynamic payoffs in binding regime (alpha pinned by LL)
# ---------------------------------------------------------------------------

def binding_payoffs(beta, W, rho, VP_spline, VA_spline,
                    nodes, weights,
                    theta, k, gamma, sigma, w_bar_ll, gamma_w, y_lower,
                    u_p_bar, u_a_bar, W_min, W_max):
    """
    Continuation-value surpluses above disagreement in the binding regime.
    alpha = w_lower(W) - beta * y_lower.
    """
    alpha = w_lower(W, w_bar_ll, gamma_w) - beta * y_lower
    EV_P, EV_A = expected_continuation(W, beta, alpha, VP_spline, VA_spline,
                                        nodes, weights,
                                        theta, k, sigma, W_min, W_max)
    wl = w_lower(W, w_bar_ll, gamma_w)
    Pi_b = D(beta, theta, k) + beta * y_lower - wl - u_p_bar + rho * EV_P
    CE_b = C(beta, theta, k, gamma, sigma) - beta * y_lower + wl - u_a_bar + rho * EV_A
    return Pi_b, CE_b


# ---------------------------------------------------------------------------
# Dynamic Nash FOC
# ---------------------------------------------------------------------------

def nash_foc_dynamic(beta, W, delta, rho, VP_spline, VA_spline,
                     nodes, weights,
                     theta, k, gamma, sigma, w_bar_ll, gamma_w, y_lower,
                     u_p_bar, u_a_bar, W_min, W_max):
    """
    Nash FOC in the binding regime using dynamic payoffs.
    Returns zero at beta*(W). Positive at beta=0, negative at beta=1.
    """
    Pi_b, CE_b = binding_payoffs(
        beta, W, rho, VP_spline, VA_spline, nodes, weights,
        theta, k, gamma, sigma, w_bar_ll, gamma_w, y_lower,
        u_p_bar, u_a_bar, W_min, W_max)

    if Pi_b <= 0 or CE_b <= 0:
        return np.nan

    dPi = Pi_prime(beta, theta, k, y_lower)
    dCE = CE_prime(beta, theta, k, gamma, sigma, y_lower)
    return (1 - delta) * dPi / Pi_b + delta * dCE / CE_b


# ---------------------------------------------------------------------------
# Single VFI iteration
# ---------------------------------------------------------------------------

def vfi_iterate(W_grid, VP, VA, W_bar, params, nodes, weights):
    """
    One complete VFI sweep over the grid.
    Returns updated (VP_new, VA_new, beta_pol, alpha_pol, lam_pol).
    """
    theta    = params['theta']
    k        = params['k']
    gamma    = params['gamma']
    sigma    = params['sigma']
    rho      = params['rho']
    delta    = params['delta']
    w_bar_ll = params['w_bar_ll']
    gamma_w  = params['gamma_w']
    y_lower  = params['y_lower']
    u_p_bar  = params['u_p_bar']
    u_a_bar  = params['u_a_bar']

    N = len(W_grid)
    W_min, W_max = W_grid[0], W_grid[-1]

    VP_spline = CubicSpline(W_grid, VP, extrapolate=True)
    VA_spline = CubicSpline(W_grid, VA, extrapolate=True)

    VP_new   = np.zeros(N)
    VA_new   = np.zeros(N)
    beta_pol = np.zeros(N)
    alpha_pol = np.zeros(N)
    lam_pol  = np.zeros(N)

    b_eff = beta_eff(theta, k, gamma, sigma)

    for i, W in enumerate(W_grid):

        if W >= W_bar:
            # Slack regime: beta = beta_eff, alpha splits surplus by delta
            beta_i  = b_eff
            alpha_i = u_a_bar - C(b_eff, theta, k, gamma, sigma) \
                      + delta * surplus(b_eff, theta, k, gamma, sigma)
            lam_i   = delta

        else:
            # Binding regime: find feasible beta interval, then solve FOC
            betas_scan = np.linspace(0.02, 0.98, 200)
            feas = []
            for bv in betas_scan:
                Pi_b_test, CE_b_test = binding_payoffs(
                    bv, W, rho, VP_spline, VA_spline, nodes, weights,
                    theta, k, gamma, sigma, w_bar_ll, gamma_w, y_lower,
                    u_p_bar, u_a_bar, W_min, W_max)
                if Pi_b_test > 0 and CE_b_test > 0:
                    feas.append(bv)

            if len(feas) < 2:
                beta_i = b_eff
            else:
                lo_b = feas[0]  + 1e-6
                hi_b = feas[-1] - 1e-6
                foc = lambda b: nash_foc_dynamic(
                    b, W, delta, rho, VP_spline, VA_spline, nodes, weights,
                    theta, k, gamma, sigma, w_bar_ll, gamma_w, y_lower,
                    u_p_bar, u_a_bar, W_min, W_max)
                try:
                    f_lo = foc(lo_b)
                    f_hi = foc(hi_b)
                    if (not np.isnan(f_lo)) and (not np.isnan(f_hi)) and f_lo * f_hi < 0:
                        beta_i = brentq(foc, lo_b, hi_b, xtol=1e-10)
                    else:
                        beta_i = b_eff
                except ValueError:
                    beta_i = b_eff

            alpha_i = w_lower(W, w_bar_ll, gamma_w) - beta_i * y_lower
            Pi_b, CE_b = binding_payoffs(
                beta_i, W, rho, VP_spline, VA_spline, nodes, weights,
                theta, k, gamma, sigma, w_bar_ll, gamma_w, y_lower,
                u_p_bar, u_a_bar, W_min, W_max)
            S_b = Pi_b + CE_b
            lam_i = CE_b / S_b if S_b > 0 else delta

        # Update value functions
        EV_P, EV_A = expected_continuation(
            W, beta_i, alpha_i, VP_spline, VA_spline, nodes, weights,
            theta, k, sigma, W_min, W_max)

        if W >= W_bar:
            VP_new[i] = D(beta_i, theta, k) - alpha_i + rho * EV_P
            VA_new[i] = alpha_i + C(beta_i, theta, k, gamma, sigma) + rho * EV_A
        else:
            wl = w_lower(W, w_bar_ll, gamma_w)
            VP_new[i] = D(beta_i, theta, k) + beta_i * y_lower - wl + rho * EV_P
            VA_new[i] = C(beta_i, theta, k, gamma, sigma) - beta_i * y_lower + wl + rho * EV_A

        beta_pol[i]  = beta_i
        alpha_pol[i] = alpha_i
        lam_pol[i]   = lam_i

    return VP_new, VA_new, beta_pol, alpha_pol, lam_pol


# ---------------------------------------------------------------------------
# Full VFI
# ---------------------------------------------------------------------------

def run_vfi(W_grid, W_bar, params, verbose=True):
    """
    Run VFI to convergence. Returns a dict with value and policy functions.
    """
    tol      = params['vfi_tol']
    max_iter = params['vfi_max_iter']
    n_quad   = params['n_quad']
    theta    = params['theta']
    k        = params['k']
    gamma    = params['gamma']
    sigma    = params['sigma']

    nodes, weights = gauss_hermite_weights(n_quad)

    N  = len(W_grid)
    VP = np.zeros(N)
    VA = np.zeros(N)

    for iteration in range(max_iter):
        VP_new, VA_new, beta_pol, alpha_pol, lam_pol = vfi_iterate(
            W_grid, VP, VA, W_bar, params, nodes, weights)

        err = max(np.max(np.abs(VP_new - VP)), np.max(np.abs(VA_new - VA)))
        VP, VA = VP_new, VA_new

        if verbose and iteration % 50 == 0:
            print(f"  iter {iteration:4d}  error {err:.2e}")

        if err < tol:
            if verbose:
                print(f"  converged at iter {iteration}  error {err:.2e}")
            break
    else:
        print(f"  VFI did not converge after {max_iter} iterations")

    b_eff = beta_eff(theta, k, gamma, sigma)
    delta_W = np.array([
        bargaining_distortion(beta_pol[i], theta, k, gamma, sigma)
        for i in range(N)])

    return {
        'W_grid':    W_grid,
        'VP':        VP,
        'VA':        VA,
        'beta':      beta_pol,
        'alpha':     alpha_pol,
        'lambda':    lam_pol,
        'delta_W':   delta_W,
        'W_bar':     W_bar,
        'beta_eff':  b_eff,
    }
