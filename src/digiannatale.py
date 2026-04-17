# digiannatale.py
# Numerical implementation of DiGiannatale, Curiel-Cabral, and Basulto (2025).
#
# Parameters and model from their Section 4.2.
#
# Speed-up relative to generic SLSQP:
#   With binary output and CRRA utility v(w) = 2*sqrt(w), the IC for
#   recommending high effort reduces to:
#       v(w_H) - v(w_L) >= 3*(a_H^2 - a_L^2) = 0.09
#   This is a constraint on wages only, independent of continuation values.
#   At the optimum IC binds (standard result in binary moral hazard).
#   Substituting IC as equality: v(w_H) = v(w_L) + IC_RHS
#   => w_H = v_inv(v(w_L) + IC_RHS)
#   reduces the per-period problem to a 1-variable line search over w_L,
#   which is about 100x faster than 2-variable SLSQP.

import numpy as np
from scipy.optimize import minimize_scalar

Y_L   = 0.4
Y_H   = 0.8
A_L   = 0.1
A_H   = 0.2
F_H_H = 2.0 / 3
F_L_H = 1.0 / 3
F_H_L = 1.0 / 3
F_L_L = 2.0 / 3
H     = 0.5
BETA  = 0.96
EPS   = 0.001
N_DELTA = 300
VFI_TOL = 1e-6
VFI_MAX_ITER = 1000

# Constant: IC slack with high effort binding
IC_SLACK_H = 3.0 * (A_H**2 - A_L**2)   # = 0.09, for high effort IC
IC_SLACK_L = 3.0 * (A_L**2 - A_H**2)   # = -0.09, never binds (low effort IC)


def v(w):
    """Agent CRRA utility."""
    w = np.maximum(w, 1e-10)
    return w**(1.0 - H) / (1.0 - H)


def v_inv(val):
    """Inverse of v: w such that v(w) = val."""
    return np.maximum(val * (1.0 - H), 0.0) ** (1.0 / (1.0 - H))


def build_delta_grid():
    return np.linspace(0.0, 1.0, N_DELTA)


def transition_indices(K, delta_grid, eps=EPS):
    """Nearest-grid index of delta' for each output realization."""
    N = len(delta_grid)
    delta = delta_grid[K]
    P = int(np.clip(np.searchsorted(delta_grid, min(1.0, delta + eps)), 0, N - 1))
    Q = int(np.clip(np.searchsorted(delta_grid, max(0.0, delta - eps * Y_L / Y_H)),
                    0, N - 1))
    return P, Q


def objective_high_effort(wL, delta, V_nH, V_nL, U_nH, U_nL):
    """
    Objective for high-effort contract, with IC holding as equality.
    wH is pinned by: v(wH) = v(wL) + IC_SLACK_H.
    Returns negative welfare (for minimization).
    """
    vL = v(wL)
    vH_val = vL + IC_SLACK_H
    wH = v_inv(vH_val)
    if wH > Y_H:
        return 1e10
    ev = (F_H_H * (vH_val - A_H**2 + BETA * V_nH) +
          F_L_H * (vL    - A_H**2 + BETA * V_nL))
    eu = (F_H_H * (Y_H - wH + BETA * U_nH) +
          F_L_H * (Y_L - wL + BETA * U_nL))
    return -(delta * ev + (1 - delta) * eu)


def optimize_one_point(delta, V_nH, V_nL, U_nH, U_nL):
    """
    Solve per-period optimization at one delta value.
    Only considers high-effort contracts (IC for a_H).
    Low effort is always dominated under the stated parameters.
    """
    result = minimize_scalar(
        objective_high_effort,
        args=(delta, V_nH, V_nL, U_nH, U_nL),
        bounds=(0.0, Y_L),
        method='bounded',
        options={'xatol': 1e-8, 'maxiter': 500},
    )

    wL = float(np.clip(result.x, 0.0, Y_L))
    vL = v(wL)
    vH = vL + IC_SLACK_H
    wH = float(np.clip(v_inv(vH), 0.0, Y_H))

    ev = (F_H_H * (vH - A_H**2 + BETA * V_nH) +
          F_L_H * (vL - A_H**2 + BETA * V_nL))
    eu = (F_H_H * (Y_H - wH + BETA * U_nH) +
          F_L_H * (Y_L - wL  + BETA * U_nL))
    S  = delta * ev + (1 - delta) * eu

    return S, wH, wL, A_H, ev, eu


def vfi_iterate_dg(delta_grid, V, U):
    N = len(delta_grid)
    V_new   = np.zeros(N)
    U_new   = np.zeros(N)
    S_new   = np.zeros(N)
    wH_pol  = np.zeros(N)
    wL_pol  = np.zeros(N)
    eff_pol = np.full(N, A_H)

    for K in range(N):
        P, Q = transition_indices(K, delta_grid)
        S, wH, wL, a, EV, EU = optimize_one_point(
            delta_grid[K], V[P], V[Q], U[P], U[Q])
        V_new[K]   = EV
        U_new[K]   = EU
        S_new[K]   = S
        wH_pol[K]  = wH
        wL_pol[K]  = wL
        eff_pol[K] = a

    return V_new, U_new, S_new, wH_pol, wL_pol, eff_pol


def run_vfi_dg(eps=EPS, verbose=True):
    """Run DiGiannatale VFI to convergence."""
    delta_grid = build_delta_grid()
    N = len(delta_grid)
    if verbose:
        print(f"  DG VFI: N={N} delta grid points, eps={eps}")

    V = np.zeros(N)
    U = np.zeros(N)

    for iteration in range(VFI_MAX_ITER):
        V_new, U_new, S_new, wH_pol, wL_pol, eff_pol = vfi_iterate_dg(
            delta_grid, V, U)
        err = max(np.max(np.abs(V_new - V)), np.max(np.abs(U_new - U)))
        V, U = V_new, U_new
        if verbose and iteration % 20 == 0:
            print(f"  DG VFI iter {iteration:4d}  error {err:.2e}")
        if err < VFI_TOL:
            if verbose:
                print(f"  DG VFI converged at iter {iteration}  error {err:.2e}")
            break
    else:
        print("  DG VFI did not converge")

    S_final = delta_grid * V + (1 - delta_grid) * U
    return {
        'delta_grid': delta_grid,
        'V': V, 'U': U, 'S': S_final,
        'w_H': wH_pol, 'w_L': wL_pol,
        'effort': eff_pol, 'eps': eps,
    }


def simulate_dg(result, T=100, delta0=0.1, seed=0):
    """Simulate the DiGiannatale model for T periods."""
    rng = np.random.default_rng(seed)
    delta_grid = result['delta_grid']
    wH_pol     = result['w_H']
    wL_pol     = result['w_L']
    eff_pol    = result['effort']
    eps        = result['eps']

    delta = float(np.clip(delta0, 0.0, 1.0))
    delta_path = np.zeros(T)
    w_path     = np.zeros(T)
    y_path     = np.zeros(T)

    for t in range(T):
        idx = int(np.clip(np.searchsorted(delta_grid, delta), 0, len(delta_grid) - 1))
        a   = eff_pol[idx]
        f_H = F_H_H if a == A_H else F_H_L
        y   = Y_H if rng.random() < f_H else Y_L
        w   = wH_pol[idx] if y == Y_H else wL_pol[idx]

        delta_path[t] = delta
        w_path[t] = w
        y_path[t] = y

        if y == Y_H:
            delta = min(1.0, delta + eps)
        else:
            delta = max(0.0, delta - eps * Y_L / Y_H)
        delta = float(np.clip(delta, 0.0, 1.0))

    return {'delta': delta_path, 'w': w_path, 'y': y_path}


if __name__ == '__main__':
    print("Running DiGiannatale VFI...")
    result = run_vfi_dg(verbose=True)
    print(f"S  range: [{result['S'].min():.4f}, {result['S'].max():.4f}]")
    print(f"w_H range: [{result['w_H'].min():.4f}, {result['w_H'].max():.4f}]")
    print(f"w_L range: [{result['w_L'].min():.4f}, {result['w_L'].max():.4f}]")
    print()
    sim = simulate_dg(result, T=100, delta0=0.1)
    print(f"Simulation delta range: [{sim['delta'].min():.4f}, {sim['delta'].max():.4f}]")
    print(f"Simulation w range: [{sim['w'].min():.4f}, {sim['w'].max():.4f}]")
