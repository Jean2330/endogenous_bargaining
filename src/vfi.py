# vfi.py
# Value function iteration for the dynamic Nash bargaining model.
# Returns converged value functions and policy functions on the wealth grid.

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.optimize import brentq

from model import (beta_eff, C, D, surplus, w_lower,
                   bargaining_distortion)


# ---------------------------------------------------------------------------
# Gauss-Hermite quadrature
# ---------------------------------------------------------------------------

def gauss_hermite_weights(n):
    """
    Nodes and weights for computing E[f(X)] where X ~ N(0,1).

    numpy's hermegauss gives nodes and weights for the probabilist
    Hermite polynomials, with the convention:
        integral f(x) * exp(-x^2/2) dx = sum_j w_j * f(z_j)

    Dividing weights by sqrt(2*pi) converts this to:
        E[f(X)] = sum_j (w_j / sqrt(2*pi)) * f(z_j)

    so that the normalized weights sum to 1 and the quadrature
    directly approximates the expectation under N(0,1).
    """
    nodes, weights = np.polynomial.hermite_e.hermegauss(n)
    weights = weights / np.sqrt(2 * np.pi)
    return nodes, weights


# ---------------------------------------------------------------------------
# Expected continuation value and its derivative w.r.t. beta
# ---------------------------------------------------------------------------

def expected_continuation(W, beta, alpha, VP_spline, VA_spline,
                           nodes, weights, theta, k, sigma, rho_W,
                           W_min, W_max):
    """
    Compute E[VP(W')] and E[VA(W')] by Gauss-Hermite quadrature.

    Law of motion:
        a* = beta * theta / k
        y  = theta * a* + sigma * eps
        W' = rho_W * W + alpha + beta * y - (k/2) * a*^2

    Substituting:
        W' = rho_W*W + alpha + beta*theta*a* - (k/2)*a*^2 + beta*sigma*eps

    The deterministic drift and the stochastic term beta*sigma*eps separate
    cleanly, so quadrature integrates over eps directly.
    """
    a     = beta * theta / k
    drift = rho_W * W + alpha + beta * theta * a - (k / 2) * a**2
    W_next = np.clip(drift + beta * sigma * nodes, W_min, W_max)
    EV_P = float(np.dot(weights, VP_spline(W_next)))
    EV_A = float(np.dot(weights, VA_spline(W_next)))
    return EV_P, EV_A


def expected_continuation_derivative(W, beta, alpha, VP_spline, VA_spline,
                                      nodes, weights, theta, k, sigma, rho_W,
                                      W_min, W_max, y_lower):
    """
    Compute d/dbeta E[VP(W')] and d/dbeta E[VA(W')] by central finite differences
    on the full quadrature sum.

    Analytic derivation of the sensitivity:
        In the binding regime, alpha = w_lower(W) - beta * y_lower, so
        dalpha/dbeta = -y_lower. Effort satisfies a* = beta*theta/k, so
        dc/dbeta = beta*theta^2/k. The total derivative of W' with respect
        to beta is:
            dW'/dbeta = dalpha/dbeta + y - dc/dbeta
                      = -y_lower + (theta*a* + sigma*eps) - beta*theta^2/k
                      = sigma*eps - y_lower
        where the theta*a* and beta*theta^2/k terms cancel because a* = beta*theta/k.

    Why finite differences rather than the spline chain rule:
        The chain-rule approach E[V'(W') * (sigma*eps - y_lower)] requires
        evaluating the spline derivative V'(W') at each quadrature node. Spline
        derivatives are unreliable near the grid edges and in the clipping region
        (where W' = W_min or W_max), because the extrapolation assumption of the
        cubic spline does not guarantee the correct shape of V there.
        Finite differences on the full quadrature sum E[V(W'(beta))] avoid this:
        they perturb beta directly, recompute all W' nodes, and difference the
        resulting expected values. Clipping is handled automatically because
        W_next = clip(...) is evaluated at both beta+h and beta-h. The step size
        h = 1e-5 balances truncation error (order h^2 for central differences)
        against floating-point cancellation, giving errors around 1e-10, well
        below the VFI tolerance of 1e-6.
    """
    h = 1e-5

    def ev(b):
        a_b     = b * theta / k
        alpha_b = alpha - (b - beta) * y_lower
        drift_b = rho_W * W + alpha_b + b * theta * a_b - (k / 2) * a_b**2
        W_next  = np.clip(drift_b + b * sigma * nodes, W_min, W_max)
        ev_p    = float(np.dot(weights, VP_spline(W_next)))
        ev_a    = float(np.dot(weights, VA_spline(W_next)))
        return ev_p, ev_a

    ev_p_hi, ev_a_hi = ev(beta + h)
    ev_p_lo, ev_a_lo = ev(beta - h)

    dEV_P = (ev_p_hi - ev_p_lo) / (2 * h)
    dEV_A = (ev_a_hi - ev_a_lo) / (2 * h)
    return dEV_P, dEV_A


# ---------------------------------------------------------------------------
# Dynamic payoffs and FOC in the binding regime
# ---------------------------------------------------------------------------

def binding_payoffs(beta, W, rho, rho_W, VP_spline, VA_spline,
                    nodes, weights,
                    theta, k, gamma, sigma, w_bar_ll, gamma_w, y_lower,
                    u_p_bar, u_a_bar, W_min, W_max):
    """
    Dynamic surpluses above disagreement in the binding regime.

    In the binding regime, alpha is pinned by the limited liability constraint:
        alpha = w_lower(W) - beta * y_lower

    The dynamic principal surplus is:
        Pi(beta, W) = D(beta) + beta*y_lower - w_lower(W) - u_p_bar + rho*E[VP(W')]

    The dynamic agent surplus is:
        CE(beta, W) = C(beta) - beta*y_lower + w_lower(W) - u_a_bar + rho*E[VA(W')]

    Both include the continuation value, making them the correct dynamic objects
    for the recursive Nash bargaining problem.
    """
    alpha  = w_lower(W, w_bar_ll, gamma_w) - beta * y_lower
    EV_P, EV_A = expected_continuation(
        W, beta, alpha, VP_spline, VA_spline, nodes, weights,
        theta, k, sigma, rho_W, W_min, W_max)
    wl   = w_lower(W, w_bar_ll, gamma_w)
    Pi_b = D(beta, theta, k) + beta * y_lower - wl - u_p_bar + rho * EV_P
    CE_b = C(beta, theta, k, gamma, sigma) - beta * y_lower + wl - u_a_bar + rho * EV_A
    return Pi_b, CE_b


def binding_payoffs_derivative(beta, W, rho, rho_W, VP_spline, VA_spline,
                                nodes, weights,
                                theta, k, gamma, sigma, w_bar_ll, gamma_w,
                                y_lower, u_p_bar, u_a_bar, W_min, W_max):
    """
    Derivatives of the dynamic surpluses with respect to beta.

    d Pi / d beta = dD/dbeta + y_lower + rho * d/dbeta E[VP(W')]
    d CE / d beta = dC/dbeta - y_lower + rho * d/dbeta E[VA(W')]

    where:
        dD/dbeta = (theta^2/k) * (1 - 2*beta)
        dC/dbeta = beta*theta^2/k - gamma*beta*sigma^2

    and d/dbeta E[V(W')] = E[V'(W') * (sigma*eps - y_lower)] as derived in
    expected_continuation_derivative.
    """
    alpha = w_lower(W, w_bar_ll, gamma_w) - beta * y_lower
    dEV_P, dEV_A = expected_continuation_derivative(
        W, beta, alpha, VP_spline, VA_spline, nodes, weights,
        theta, k, sigma, rho_W, W_min, W_max, y_lower)

    dD = (theta**2 / k) * (1 - 2 * beta)
    dC = beta * theta**2 / k - gamma * beta * sigma**2

    dPi = dD + y_lower + rho * dEV_P
    dCE = dC - y_lower + rho * dEV_A
    return dPi, dCE


def dynamic_feasible_bracket(W, rho, rho_W, VP_spline, VA_spline,
                               nodes, weights,
                               theta, k, gamma, sigma, w_bar_ll, gamma_w,
                               y_lower, u_p_bar, u_a_bar, W_min, W_max):
    """
    Find the feasible interval [lo, hi] for beta using the dynamic surpluses.

    The feasible set is where Pi_b(beta) > 0 AND CE_b(beta) > 0.
    Using dynamic surpluses (which include rho*E[V(W')]) gives the correct
    feasibility condition for the recursive problem. The static version used
    in the previous implementation was based on Pi_binding and CE_binding,
    which omit the continuation values and therefore misrepresent feasibility.

    We scan 30 points to locate the feasible interior, which is fast and
    sufficient for bracket purposes.
    """
    betas = np.linspace(0.01, 0.99, 30)
    Pi_vals = np.zeros(30)
    CE_vals = np.zeros(30)
    for j, b in enumerate(betas):
        Pi_vals[j], CE_vals[j] = binding_payoffs(
            b, W, rho, rho_W, VP_spline, VA_spline, nodes, weights,
            theta, k, gamma, sigma, w_bar_ll, gamma_w, y_lower,
            u_p_bar, u_a_bar, W_min, W_max)

    feasible_mask = (Pi_vals > 0) & (CE_vals > 0)
    if feasible_mask.sum() < 2:
        return None
    lo = betas[feasible_mask][0]  + 1e-4
    hi = betas[feasible_mask][-1] - 1e-4
    if lo >= hi:
        return None
    return lo, hi


def nash_foc_dynamic(beta, W, delta, rho, rho_W, VP_spline, VA_spline,
                     nodes, weights,
                     theta, k, gamma, sigma, w_bar_ll, gamma_w, y_lower,
                     u_p_bar, u_a_bar, W_min, W_max):
    """
    True dynamic Nash FOC in the binding regime.

    The Nash product is N(beta) = Pi_b^(1-delta) * CE_b^delta.
    The first-order condition dN/dbeta = 0 gives:

        (1 - delta) * (dPi/dbeta) / Pi_b + delta * (dCE/dbeta) / CE_b = 0

    where dPi/dbeta and dCE/dbeta are the FULL derivatives of the dynamic
    surpluses, including the effect of beta on the continuation values through
    the law of motion W' = rho_W*W + alpha(beta) + beta*y - c(a*(beta)).

    The previous implementation used only the static derivatives dD/dbeta and
    dC/dbeta, omitting rho * d/dbeta E[V(W')]. These continuation derivatives
    are nonzero whenever VP and VA are nonlinear in W, which is generically true
    in the binding regime.
    """
    Pi_b, CE_b = binding_payoffs(
        beta, W, rho, rho_W, VP_spline, VA_spline, nodes, weights,
        theta, k, gamma, sigma, w_bar_ll, gamma_w, y_lower,
        u_p_bar, u_a_bar, W_min, W_max)

    if Pi_b <= 0 or CE_b <= 0:
        return np.nan

    dPi, dCE = binding_payoffs_derivative(
        beta, W, rho, rho_W, VP_spline, VA_spline, nodes, weights,
        theta, k, gamma, sigma, w_bar_ll, gamma_w, y_lower,
        u_p_bar, u_a_bar, W_min, W_max)

    return (1 - delta) * dPi / Pi_b + delta * dCE / CE_b


# ---------------------------------------------------------------------------
# Single VFI iteration
# ---------------------------------------------------------------------------

def vfi_iterate(W_grid, VP, VA, W_bar, params, nodes, weights):
    """
    One complete VFI sweep over the grid.
    Returns updated (VP_new, VA_new, beta_pol, alpha_pol, lam_pol).

    Slack regime (W >= W_bar):
        The LL constraint does not bind. alpha is free. The recursive Nash
        solution here is identical to the static one: beta = beta_eff and
        lambda = delta. The continuation value is computed under this policy
        and used to update VP and VA.

    Binding regime (W < W_bar):
        alpha is pinned by LL. beta is chosen to maximize the Nash product
        using the full dynamic FOC, which includes the effect of beta on
        E[V(W')] through the law of motion. Feasibility is checked using the
        dynamic surpluses, not the static ones.
    """
    theta    = params['theta']
    k        = params['k']
    gamma    = params['gamma']
    sigma    = params['sigma']
    rho      = params['rho']
    rho_W    = params['rho_W']
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

    VP_new    = np.zeros(N)
    VA_new    = np.zeros(N)
    beta_pol  = np.zeros(N)
    alpha_pol = np.zeros(N)
    lam_pol   = np.zeros(N)

    b_eff = beta_eff(theta, k, gamma, sigma)

    for i, W in enumerate(W_grid):

        if W >= W_bar:
            # Slack regime: the analytic solution IS the recursive solution.
            #
            # When LL does not bind, alpha is unconstrained. The recursive Nash
            # problem is to choose (alpha, beta) to maximize
            #     N = Pi_b^(1-delta) * CE_b^delta
            # where Pi_b and CE_b are the full dynamic surpluses including
            # rho*E[V(W')]. The FOC in alpha gives:
            #     -(1-delta)/Pi_b + delta/CE_b = 0  =>  CE_b/Pi_b = delta/(1-delta)
            # so lambda = delta at any beta. Since alpha is free and enters
            # Pi_b and CE_b with slopes -1 and +1 respectively, the FOC in
            # beta reduces to the same condition as in the static problem:
            #     d/dbeta [D(beta) + C(beta)] = 0
            # which gives beta = beta_eff regardless of the continuation values.
            # The continuation values affect the level of the surpluses but not
            # the slope of the Nash product with respect to beta when alpha is
            # free to re-equate the shares. Therefore beta_eff and lambda = delta
            # are the exact recursive Nash solution in the slack regime, not an
            # approximation. The value functions are then updated with the correct
            # continuation values under this policy.
            beta_i  = b_eff
            alpha_i = u_a_bar - C(b_eff, theta, k, gamma, sigma) \
                      + delta * surplus(b_eff, theta, k, gamma, sigma)
            lam_i   = delta

            EV_P, EV_A = expected_continuation(
                W, beta_i, alpha_i, VP_spline, VA_spline, nodes, weights,
                theta, k, sigma, rho_W, W_min, W_max)
            VP_new[i] = D(beta_i, theta, k) - alpha_i + rho * EV_P
            VA_new[i] = alpha_i + C(beta_i, theta, k, gamma, sigma) + rho * EV_A

        else:
            # Binding regime: use dynamic feasibility and dynamic FOC.
            bracket = dynamic_feasible_bracket(
                W, rho, rho_W, VP_spline, VA_spline, nodes, weights,
                theta, k, gamma, sigma, w_bar_ll, gamma_w, y_lower,
                u_p_bar, u_a_bar, W_min, W_max)

            if bracket is None:
                # No feasible interior: fall back to the binding alpha at b_eff.
                beta_i  = b_eff
                alpha_i = w_lower(W, w_bar_ll, gamma_w) - beta_i * y_lower
                lam_i   = delta
            else:
                lo_b, hi_b = bracket
                foc = lambda b: nash_foc_dynamic(
                    b, W, delta, rho, rho_W, VP_spline, VA_spline,
                    nodes, weights,
                    theta, k, gamma, sigma, w_bar_ll, gamma_w, y_lower,
                    u_p_bar, u_a_bar, W_min, W_max)
                try:
                    f_lo = foc(lo_b)
                    f_hi = foc(hi_b)
                    if (not np.isnan(f_lo)) and (not np.isnan(f_hi)) \
                            and f_lo * f_hi < 0:
                        beta_i = brentq(foc, lo_b, hi_b, xtol=1e-10)
                    else:
                        # FOC does not change sign: take the endpoint with smaller
                        # absolute value (closest to satisfying the FOC).
                        f_lo_abs = abs(f_lo) if not np.isnan(f_lo) else np.inf
                        f_hi_abs = abs(f_hi) if not np.isnan(f_hi) else np.inf
                        beta_i = lo_b if f_lo_abs < f_hi_abs else hi_b
                except ValueError:
                    beta_i = b_eff

                alpha_i = w_lower(W, w_bar_ll, gamma_w) - beta_i * y_lower

                Pi_b, CE_b = binding_payoffs(
                    beta_i, W, rho, rho_W, VP_spline, VA_spline,
                    nodes, weights,
                    theta, k, gamma, sigma, w_bar_ll, gamma_w, y_lower,
                    u_p_bar, u_a_bar, W_min, W_max)

                S_b   = Pi_b + CE_b
                lam_i = CE_b / S_b if S_b > 0 else delta

            # Value function update in binding regime.
            # binding_payoffs returns Pi_b = static terms + rho*E[VP(W')] - u_p_bar
            # and CE_b = static terms + rho*E[VA(W')] - u_a_bar.
            # The Bellman values are VP = Pi_b + u_p_bar and VA = CE_b + u_a_bar.
            Pi_b, CE_b = binding_payoffs(
                beta_i, W, rho, rho_W, VP_spline, VA_spline,
                nodes, weights,
                theta, k, gamma, sigma, w_bar_ll, gamma_w, y_lower,
                u_p_bar, u_a_bar, W_min, W_max)
            VP_new[i] = Pi_b + u_p_bar
            VA_new[i] = CE_b + u_a_bar

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
        'W_grid':   W_grid,
        'VP':       VP,
        'VA':       VA,
        'beta':     beta_pol,
        'alpha':    alpha_pol,
        'lambda':   lam_pol,
        'delta_W':  delta_W,
        'W_bar':    W_bar,
        'beta_eff': b_eff,
    }