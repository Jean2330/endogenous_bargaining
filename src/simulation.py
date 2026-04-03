# simulation.py
# Simulates wealth paths and the ergodic distribution given converged policy functions.

import numpy as np
from scipy.interpolate import CubicSpline

from model import w_lower, C, D


def simulate_paths(vfi_result, params, seed=42):
    """
    Simulate N_AGENTS agents for T_SIM periods each.
    Returns a dict with arrays of shape (N_agents, T_sim - T_burn).
    """
    rng = np.random.default_rng(seed)

    W_grid    = vfi_result['W_grid']
    beta_pol  = vfi_result['beta']
    alpha_pol = vfi_result['alpha']
    lam_pol   = vfi_result['lambda']
    W_bar     = vfi_result['W_bar']

    beta_spl  = CubicSpline(W_grid, beta_pol,  extrapolate=True)
    alpha_spl = CubicSpline(W_grid, alpha_pol, extrapolate=True)
    lam_spl   = CubicSpline(W_grid, lam_pol,   extrapolate=True)

    theta  = params['theta']
    k      = params['k']
    sigma  = params['sigma']
    T      = params['t_sim']
    M      = params['n_agents']
    T_burn = params['t_burn']
    W_min  = W_grid[0]
    W_max  = W_grid[-1]

    W_curr = np.full(M, W_bar * 0.5)

    W_out   = np.zeros((M, T))
    lam_out = np.zeros((M, T))
    ell_out = np.zeros((M, T))
    bind_out = np.zeros((M, T), dtype=bool)

    for t in range(T):
        eps = rng.standard_normal(M)

        beta_t  = np.clip(beta_spl(W_curr),  1e-6, 1 - 1e-6)
        alpha_t = alpha_spl(W_curr)
        lam_t   = np.clip(lam_spl(W_curr),   1e-6, 1 - 1e-6)

        a_t    = beta_t * theta / k
        y_t    = theta * a_t + sigma * eps
        W_next = W_curr + alpha_t + beta_t * y_t - (k / 2) * a_t**2
        W_next = np.clip(W_next, W_min, W_max)

        W_out[:, t]    = W_curr
        lam_out[:, t]  = lam_t
        ell_out[:, t]  = np.log(lam_t / (1 - lam_t))
        bind_out[:, t] = W_curr < W_bar

        W_curr = W_next

    return {
        'W':       W_out[:,   T_burn:],
        'lambda':  lam_out[:, T_burn:],
        'ell':     ell_out[:, T_burn:],
        'binding': bind_out[:, T_burn:],
        'W_bar':   W_bar,
    }


def single_path(vfi_result, params, W0=None, seed=0):
    """
    Simulate a single long agent path for illustrating dynamics.
    Returns arrays of length T_SIM - T_BURN.
    """
    rng = np.random.default_rng(seed)

    W_grid    = vfi_result['W_grid']
    beta_pol  = vfi_result['beta']
    alpha_pol = vfi_result['alpha']
    lam_pol   = vfi_result['lambda']
    W_bar     = vfi_result['W_bar']

    beta_spl  = CubicSpline(W_grid, beta_pol,  extrapolate=True)
    alpha_spl = CubicSpline(W_grid, alpha_pol, extrapolate=True)
    lam_spl   = CubicSpline(W_grid, lam_pol,   extrapolate=True)

    theta  = params['theta']
    k      = params['k']
    sigma  = params['sigma']
    T      = params['t_sim']
    T_burn = params['t_burn']
    W_min  = W_grid[0]
    W_max  = W_grid[-1]

    if W0 is None:
        W0 = W_bar * 0.4

    W_curr  = float(W0)
    W_arr   = np.zeros(T)
    lam_arr = np.zeros(T)
    ell_arr = np.zeros(T)

    for t in range(T):
        eps     = rng.standard_normal()
        beta_t  = float(np.clip(beta_spl(W_curr),  1e-6, 1 - 1e-6))
        alpha_t = float(alpha_spl(W_curr))
        lam_t   = float(np.clip(lam_spl(W_curr),   1e-6, 1 - 1e-6))

        a_t    = beta_t * theta / k
        y_t    = theta * a_t + sigma * eps
        W_next = W_curr + alpha_t + beta_t * y_t - (k / 2) * a_t**2
        W_next = float(np.clip(W_next, W_min, W_max))

        W_arr[t]   = W_curr
        lam_arr[t] = lam_t
        ell_arr[t] = np.log(lam_t / (1 - lam_t))

        W_curr = W_next

    return {
        'W':      W_arr[T_burn:],
        'lambda': lam_arr[T_burn:],
        'ell':    ell_arr[T_burn:],
        'W_bar':  W_bar,
    }
