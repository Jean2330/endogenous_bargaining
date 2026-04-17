# spear_srivastava.py
# Numerical implementation of Spear and Srivastava (1987).
#
# Binary output {y_L=0.4, y_H=0.8}, binary effort {a_L=0.1, a_H=0.2}.
# Agent CARA: u(w) = -exp(-gamma*w), psi(a) = k/2*a^2, gamma=0.5, k=2.
# Principal risk-neutral. State variable: promised utility v_hat.
#
# Key analytic reductions:
#   IC binding pins w_H from w_L: u(w_H) = u(w_L) + IC_RHS
#   PK pins v_H from (w_L, v_L, v_hat)
# => After eliminating w_H and v_H, the problem is 2D in (w_L, v_L).
#
# Speed: vectorized numpy grid over (w_L, v_L) at each v_hat point.
# At 40 x 40 grid the full iteration runs in a few seconds.

import numpy as np
from scipy.interpolate import CubicSpline

Y_L   = 0.4;  Y_H  = 0.8
A_L   = 0.1;  A_H  = 0.2
F_H_H = 2/3;  F_L_H = 1/3
F_H_L = 1/3;  F_L_L = 2/3
GAMMA = 0.5;  K = 2.0;  BETA = 0.96
N_VHAT = 150;  N_W = 40;  N_V = 40
VFI_TOL = 1e-5;  VFI_MAX_ITER = 600

IC_RHS = (K/2)*(A_H**2 - A_L**2) / (F_H_H - F_H_L)  # = 0.09


def u(w):
    return -np.exp(-GAMMA * np.maximum(w, 0.0))


def u_inv(val):
    val = np.minimum(val, -1e-10)
    return -np.log(-val) / GAMMA


def psi(a):
    return (K/2)*a**2


def build_vhat_grid():
    v_min = u(0.0)/(1-BETA) - psi(A_H)/(1-BETA)
    v_max = (F_H_H*u(Y_H) + F_L_H*u(Y_L))/(1-BETA) - psi(A_H)/(1-BETA)
    return np.linspace(v_min * 0.85, v_max * 1.05, N_VHAT)


def vfi_iterate_ss(v_grid, V):
    """
    Vectorized VFI sweep for SS model.
    For each v_hat: build a (N_W x N_V) grid over (w_L, v_L),
    compute payoff at all grid points in one numpy operation,
    then take the maximum over feasible points.
    """
    N = len(v_grid)
    V_spline = CubicSpline(v_grid, V, extrapolate=False)

    V_new   = np.array(V, copy=True)
    wH_pol  = np.zeros(N)
    wL_pol  = np.zeros(N)
    vH_pol  = np.zeros(N)
    vL_pol  = np.zeros(N)

    wL_vals = np.linspace(0.0, Y_L * 0.98, N_W)
    vL_vals = np.linspace(v_grid[0], v_grid[-1], N_V)

    # Precompute u(w_L) and w_H for all w_L values.
    u_wL = u(wL_vals)               # shape (N_W,)
    u_wH_target = u_wL + IC_RHS     # shape (N_W,)
    feasible_wL = u_wH_target < 0   # u must be negative for CARA
    wH_vals = np.where(feasible_wL, u_inv(np.minimum(u_wH_target, -1e-10)), np.inf)
    feasible_wL &= (wH_vals <= Y_H)

    # Build 2D grids: (N_W, N_V)
    wL_grid, vL_grid = np.meshgrid(wL_vals, vL_vals, indexing='ij')  # (N_W, N_V)
    u_wL_2d = u_wL[:, np.newaxis] * np.ones((1, N_V))
    u_wH_2d = u_wH_target[:, np.newaxis] * np.ones((1, N_V))
    wH_2d   = wH_vals[:, np.newaxis] * np.ones((1, N_V))
    feas_2d = feasible_wL[:, np.newaxis] * np.ones((1, N_V), dtype=bool)

    for i, v_hat in enumerate(v_grid):
        # v_H from PK: v_hat = F_H*(u_wH + beta*vH) + F_L*(u_wL + beta*vL) - psi
        # F_H*beta*vH = v_hat + psi - F_H*u_wH - F_L*(u_wL + beta*vL)
        vH_2d = ((v_hat + psi(A_H) - F_H_H * u_wH_2d -
                  F_L_H * (u_wL_2d + BETA * vL_grid)) / (F_H_H * BETA))

        # Feasibility: vH and vL must be on the grid; wH must be feasible.
        feas = (feas_2d &
                (vH_2d >= v_grid[0]) & (vH_2d <= v_grid[-1]) &
                (vL_grid >= v_grid[0]) & (vL_grid <= v_grid[-1]))

        if not feas.any():
            continue

        # Evaluate V spline at vH and vL (clip to grid bounds for spline safety).
        vH_clipped = np.clip(vH_2d, v_grid[0], v_grid[-1])
        vL_clipped = np.clip(vL_grid, v_grid[0], v_grid[-1])
        V_H = V_spline(vH_clipped)   # shape (N_W, N_V)
        V_L = V_spline(vL_clipped)   # shape (N_W, N_V)

        # Replace NaN from spline extrapolation with a large negative number.
        V_H = np.where(np.isnan(V_H), -1e6, V_H)
        V_L = np.where(np.isnan(V_L), -1e6, V_L)

        payoff = (F_H_H * (Y_H - wH_2d + BETA * V_H) +
                  F_L_H * (Y_L - wL_grid + BETA * V_L))

        payoff = np.where(feas, payoff, -1e10)

        idx_flat = np.argmax(payoff)
        iw, iv   = np.unravel_index(idx_flat, payoff.shape)

        if payoff[iw, iv] > -1e9:
            V_new[i]  = payoff[iw, iv]
            wL_pol[i] = wL_vals[iw]
            wH_pol[i] = float(wH_2d[iw, iv])
            vL_pol[i] = float(vL_grid[iw, iv])
            vH_pol[i] = float(vH_2d[iw, iv])

    return V_new, wH_pol, wL_pol, vH_pol, vL_pol


def run_vfi_ss(verbose=True):
    v_grid = build_vhat_grid()
    if verbose:
        print(f"  SS VFI: N={len(v_grid)} v_hat grid, N_W={N_W}, N_V={N_V}")
        print(f"  v_hat range: [{v_grid[0]:.3f}, {v_grid[-1]:.3f}]")
    V = np.zeros(len(v_grid))
    for iteration in range(VFI_MAX_ITER):
        V_new, wH_pol, wL_pol, vH_pol, vL_pol = vfi_iterate_ss(v_grid, V)
        err = np.max(np.abs(V_new - V))
        V = V_new
        if verbose and iteration % 20 == 0:
            print(f"  SS VFI iter {iteration:4d}  error {err:.2e}")
        if err < VFI_TOL:
            if verbose:
                print(f"  SS VFI converged at iter {iteration}  error {err:.2e}")
            break
    else:
        print("  SS VFI did not converge")
    return {'v_grid': v_grid, 'V': V,
            'w_H': wH_pol, 'w_L': wL_pol,
            'v_H': vH_pol, 'v_L': vL_pol}


def simulate_ss(result, T=100, v0=None, seed=0):
    rng = np.random.default_rng(seed)
    v_grid = result['v_grid']
    wH_pol = result['w_H']; wL_pol = result['w_L']
    vH_pol = result['v_H']; vL_pol = result['v_L']
    if v0 is None:
        v0 = float(np.median(v_grid))
    v_curr = float(np.clip(v0, v_grid[0], v_grid[-1]))
    v_path = np.zeros(T); w_path = np.zeros(T); y_path = np.zeros(T)
    for t in range(T):
        idx = int(np.clip(np.searchsorted(v_grid, v_curr), 0, len(v_grid)-1))
        y = Y_H if rng.random() < F_H_H else Y_L
        w = wH_pol[idx] if y == Y_H else wL_pol[idx]
        v_next = vH_pol[idx] if y == Y_H else vL_pol[idx]
        v_path[t] = v_curr; w_path[t] = w; y_path[t] = y
        v_curr = float(np.clip(v_next, v_grid[0], v_grid[-1]))
    return {'v': v_path, 'w': w_path, 'y': y_path}


if __name__ == '__main__':
    print("Running Spear-Srivastava VFI...")
    res = run_vfi_ss(verbose=True)
    print(f"V range: [{res['V'].min():.4f}, {res['V'].max():.4f}]")
    print(f"w_H: [{res['w_H'].min():.4f}, {res['w_H'].max():.4f}]")
    print(f"w_L: [{res['w_L'].min():.4f}, {res['w_L'].max():.4f}]")
