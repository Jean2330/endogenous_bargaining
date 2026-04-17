# fig_comparison_models.py
# Generates two comparison figures against SS (1987) and DG (2025).
# All comparisons are honest: only objects in the same units are overlaid.

import os, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.interpolate import CubicSpline

sys.path.insert(0, os.path.dirname(__file__))

from spear_srivastava import run_vfi_ss
from digiannatale import (run_vfi_dg, simulate_dg,
                           F_H_H as DG_FHH, Y_L as DG_YL, Y_H as DG_YH)

import params as P
from model import w_bar_threshold, disagreement_payoffs
from vfi import run_vfi, gauss_hermite_weights
from simulation import single_path

plt.rcParams.update({
    'font.family': 'serif', 'font.size': 11,
    'axes.labelsize': 11, 'axes.titlesize': 10,
    'legend.fontsize': 9, 'figure.dpi': 150,
    'lines.linewidth': 1.6,
})

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          '..', 'paper', 'figures')
BLUE = '#1f77b4'
RED  = '#d62728'


def savefig(name):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    path = os.path.join(OUTPUT_DIR, name)
    plt.savefig(path, bbox_inches='tight', dpi=200)
    plt.close()
    print(f"  saved {path}")


def make_params():
    p = dict(
        theta=P.THETA, k=P.K, gamma=P.GAMMA, sigma=P.SIGMA,
        rho=P.RHO, rho_W=P.RHO_W, delta=P.DELTA,
        w_bar_ll=P.W_BAR_LL, gamma_w=P.GAMMA_W, y_lower=P.Y_LOWER,
        frac_disagreement=P.FRAC_DISAGREEMENT,
        n_grid=P.N_GRID, n_quad=P.N_QUAD,
        vfi_tol=P.VFI_TOL, vfi_max_iter=P.VFI_MAX_ITER,
        t_sim=P.T_SIM, n_agents=P.N_AGENTS, t_burn=P.T_BURN,
    )
    u_p, u_a = disagreement_payoffs(
        p['theta'], p['k'], p['gamma'], p['sigma'],
        p['delta'], p['frac_disagreement'])
    p['u_p_bar'] = u_p
    p['u_a_bar'] = u_a
    return p


# ---------------------------------------------------------------------------
# Figure 1: comparison with Spear-Srivastava (1987)
# ---------------------------------------------------------------------------

def fig_compare_ss(ss_result, this_result, this_params):
    """
    Four panels, each making one clean comparison with SS.

    Panels 1-2: Value functions side by side in their own spaces.
    Panel 3: Incentive slope beta (same units, directly comparable).
    Panel 4: Wage sensitivity to output (same units, directly comparable).
    """
    v_grid = ss_result['v_grid']
    V_ss   = ss_result['V']
    wH_ss  = ss_result['w_H']
    wL_ss  = ss_result['w_L']

    W_grid = this_result['W_grid']
    VP     = this_result['VP']
    beta_W = this_result['beta']
    W_bar  = this_result['W_bar']
    b_eff  = this_result['beta_eff']
    sigma  = this_params['sigma']

    fig, axes = plt.subplots(1, 4, figsize=(17, 4.2))

    # Panel 1: SS value function
    ax = axes[0]
    ax.plot(V_ss, v_grid, color=BLUE, linewidth=1.8)
    ax.set_xlabel(r'$V(v_t)$', color=BLUE)
    ax.set_ylabel(r'Promised utility $v_t$')
    ax.tick_params(axis='x', labelcolor=BLUE)
    ax.set_title('Spear-Srivastava\nprincipal value $V(v_t)$')

    # Panel 2: this model value function
    ax = axes[1]
    ax.plot(VP, W_grid, color=RED, linewidth=1.8, label=r'$V_P(W)$')
    ax.axhline(W_bar, color='gray', linewidth=0.8, linestyle=':',
               label=r'$\bar{W}$')
    ax.set_xlabel(r'$V_P(W)$', color=RED)
    ax.set_ylabel(r'Wealth $W$')
    ax.tick_params(axis='x', labelcolor=RED)
    ax.set_title('This model\nprincipal value $V_P(W)$')
    ax.text(VP.min() + 0.02 * (VP.max() - VP.min()),
            W_bar * 0.45, 'Binding\nregime',
            ha='left', va='center', fontsize=8, color='gray')
    ax.legend(fontsize=9, loc='upper left')

    # Panel 3: incentive slope
    ax = axes[2]
    ax.axhline(b_eff, color=BLUE, linewidth=1.8, linestyle='--',
               label=r'$\beta^{\mathrm{eff}}$ [SS: constant]')
    ax.plot(W_grid, beta_W, color=RED, linewidth=1.8,
            label=r'$\beta^*(W)$ [this model]')
    ax.axvline(W_bar, color='gray', linewidth=0.8, linestyle=':')
    ax.axvspan(W_grid[0], W_bar, color='lightyellow', alpha=0.5)
    ax.text((W_grid[0] + W_bar) / 2,
            beta_W.min() + 0.05 * (b_eff - beta_W.min()),
            'Binding\nregime', ha='center', va='bottom', fontsize=8, color='gray')
    ax.set_xlabel(r'Wealth $W$')
    ax.set_ylabel(r'Incentive slope $\beta$')
    ax.set_title('Incentive slope:\nconstant (SS) vs state-dependent')
    ax.legend(loc='lower right')

    # Panel 4: wage sensitivity to output
    ax = axes[3]
    v_norm = (v_grid - v_grid[0]) / (v_grid[-1] - v_grid[0])
    ax.plot(v_norm, wH_ss - wL_ss, color=BLUE, linewidth=1.8,
            label=r'$w_H(v_t) - w_L(v_t)$ [SS]')
    W_norm = (W_grid - W_grid[0]) / (W_grid[-1] - W_grid[0])
    ax.plot(W_norm, beta_W * sigma, color=RED, linewidth=1.8,
            label=r'$\beta^*(W)\cdot\sigma$ [this model]')
    ax.axvline((W_bar - W_grid[0]) / (W_grid[-1] - W_grid[0]),
               color='gray', linewidth=0.8, linestyle=':', alpha=0.7)
    ax.set_xlabel('State variable (normalized to [0,1])')
    ax.set_ylabel('Wage sensitivity to output')
    ax.set_title('Wage responsiveness\nto output shocks')
    ax.legend(fontsize=9, loc='lower right')

    plt.tight_layout()
    savefig('fig_comparison_ss.png')


# ---------------------------------------------------------------------------
# Figure 2: comparison with DiGiannatale et al. (2025)
# ---------------------------------------------------------------------------

def fig_compare_dg(dg_result, this_result, this_params):
    """
    Three panels comparing DG (2025) and this model by mechanism, not level.

    Panel 1: Mean drift of bargaining power.
    DG: constant by construction. This model: wealth-dependent, crosses zero.

    Panel 2: Performance pay sensitivity.
    DG: w_H - w_L as function of delta.
    This model: beta*(W) * wage_range as function of W (normalized).

    Panel 3: Simulated bargaining power paths.
    DG: slow upward drift. This model: fast endogenous fluctuation.
    Legend placed below the plot to keep data area unobstructed.
    """
    delta_grid = dg_result['delta_grid']
    wH_dg = dg_result['w_H']
    wL_dg = dg_result['w_L']
    eps   = dg_result['eps']

    W_grid  = this_result['W_grid']
    beta_W  = this_result['beta']
    alpha_W = this_result['alpha']
    lam_W   = this_result['lambda']
    W_bar   = this_result['W_bar']

    theta   = this_params['theta']
    k       = this_params['k']
    sigma   = this_params['sigma']
    rho_W   = this_params['rho_W']
    delta   = this_params['delta']
    y_lower = this_params['y_lower']

    W_min = W_grid[0]
    W_max = W_grid[-1]

    lam_spl   = CubicSpline(W_grid, lam_W,   extrapolate=True)
    beta_spl  = CubicSpline(W_grid, beta_W,  extrapolate=True)
    alpha_spl = CubicSpline(W_grid, alpha_W, extrapolate=True)
    nodes, weights = gauss_hermite_weights(10)

    E_lam_next = np.zeros(len(W_grid))
    for i, W in enumerate(W_grid):
        b = float(beta_spl(W))
        a = b * theta / k
        mu = rho_W * W + float(alpha_spl(W)) + b * theta * a - (k/2) * a**2
        Wn = np.clip(mu + b * sigma * nodes, W_min, W_max)
        E_lam_next[i] = float(np.dot(weights, lam_spl(Wn)))

    drift_this = E_lam_next - lam_W
    drift_dg   = DG_FHH * eps + (1 - DG_FHH) * (-eps * DG_YL / DG_YH)

    x_this = (W_grid - W_min) / (W_max - W_min)
    w_bar_norm = (W_bar - W_min) / (W_max - W_min)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

    # Panel 1: mean drift
    # Find W* (zero crossing of this model's drift) for annotation.
    # The drift crosses zero near the ergodic mean, which is just above W_bar.
    zero_crossings = np.where(np.diff(np.sign(drift_this)))[0]
    if len(zero_crossings) > 0:
        # Linear interpolation for the precise zero crossing
        i0 = zero_crossings[-1]
        x_star = x_this[i0] - drift_this[i0] * (x_this[i0+1] - x_this[i0]) / (drift_this[i0+1] - drift_this[i0])
    else:
        x_star = w_bar_norm

    ax = axes[0]
    ax.axhline(drift_dg, color=BLUE, linewidth=1.8,
               label=fr'DG: constant drift = {drift_dg:.4f}')
    ax.plot(x_this, drift_this, color=RED, linewidth=1.8,
            label=r'This model: $E[\lambda(W_{t+1})|W_t] - \lambda(W_t)$')
    ax.axhline(0, color='black', linewidth=0.5, linestyle=':')

    # W_bar: regime boundary
    ax.axvline(w_bar_norm, color='gray', linewidth=1.0, linestyle='--')
    ax.text(w_bar_norm - 0.02, ax.get_ylim()[1] if ax.get_ylim()[1] != 0 else drift_this.max() * 0.5,
            r'$\bar{W}$', fontsize=9, color='gray', ha='right', va='top')

    # W*: ergodic mean (zero crossing of drift)
    ax.axvline(x_star, color=RED, linewidth=0.8, linestyle=':')
    ax.text(x_star + 0.02, drift_this.min() * 0.6,
            r'$W^*$', fontsize=9, color=RED, ha='left')

    # Region labels
    y_mid = (drift_this.max() + drift_dg) / 2
    ax.text(w_bar_norm * 0.45, drift_this.min() * 0.4,
            'LL binding', ha='center', fontsize=8, color='gray',
            style='italic')
    ax.text(w_bar_norm + (1 - w_bar_norm) * 0.45, drift_this.min() * 0.4,
            'LL slack', ha='center', fontsize=8, color='gray',
            style='italic')

    ax.set_xlabel('State variable (normalized to [0,1])')
    ax.set_ylabel('Expected one-period change\nin bargaining power')
    ax.set_title('Mean drift of bargaining power:\nconstant (DG) vs state-dependent')
    ax.legend(fontsize=9, loc='upper right')

    # Panel 2: performance pay -- normalized by output range so comparison is
    # about incentive intensity, not levels. Both curves show the fraction of
    # the output differential that the agent bears through the wage.
    ax = axes[1]
    dg_y_range = DG_YH - DG_YL
    dg_normalized = (wH_dg - wL_dg) / dg_y_range
    ax.plot(delta_grid, dg_normalized, color=BLUE, linewidth=1.8,
            label=r'$(w_H - w_L)/(y_H - y_L)$ [DG]')

    a_star  = beta_W * theta / k
    E_y_hi  = theta * a_star + sigma
    pay_raw = beta_W * (E_y_hi - y_lower)
    # Normalize by the same output range concept: sigma is the s.d. of the
    # shock, and y_lower is the lower bound; the effective output range in
    # the linear contract is approximately 2*sigma (one s.d. above and below).
    # We normalize by the total output range for comparability.
    this_y_range = E_y_hi.mean() - y_lower
    pay_normalized = pay_raw / this_y_range
    ax.plot(x_this, pay_normalized, color=RED, linewidth=1.8,
            label=r'$\beta^*(W)\cdot(\bar{y}-y_\ell)/\overline{(y_H-y_L)}$ [this model]')

    ax.axvline(w_bar_norm, color='gray', linewidth=0.8, linestyle='--')
    ax.text(w_bar_norm - 0.02, ax.get_ylim()[1] if ax.get_ylim()[1] != 0 else pay_normalized.max() * 0.95,
            r'$\bar{W}$', fontsize=9, color='gray', ha='right', va='top')
    ax.text(w_bar_norm * 0.45, pay_normalized.min() + 0.02 * (pay_normalized.max() - pay_normalized.min()),
            'LL binding', ha='center', fontsize=8, color='gray', style='italic')
    ax.text(w_bar_norm + (1 - w_bar_norm) * 0.45, pay_normalized.min() + 0.02 * (pay_normalized.max() - pay_normalized.min()),
            'LL slack', ha='center', fontsize=8, color='gray', style='italic')
    ax.set_xlabel('State variable (normalized to [0,1])')
    ax.set_ylabel('Incentive intensity\n(output-normalized wage range)')
    ax.set_title('Performance pay intensity:\nnormalized by output range')
    ax.legend(fontsize=9, loc='center right')

    # Panel 3: simulated paths with legend BELOW the plot
    ax = axes[2]
    T_show = 120
    delta0_vals = [0.10, 0.20, 0.35]
    W0_vals     = [W_bar * 0.15, W_bar * 0.5, W_bar * 1.1]
    linestyles  = ['solid', 'dashed', 'dotted']

    for d0, ls in zip(delta0_vals, linestyles):
        sim = simulate_dg(dg_result, T=T_show, delta0=d0, seed=5)
        ax.plot(np.arange(T_show), sim['delta'],
                color=BLUE, linewidth=1.0, linestyle=ls, alpha=0.9)

    for W0, ls in zip(W0_vals, linestyles):
        path = single_path(this_result, this_params, W0=W0, seed=5)
        ax.plot(np.arange(T_show), path['lambda'][:T_show],
                color=RED, linewidth=1.0, linestyle=ls, alpha=0.9)

    ax.axhline(delta, color='gray', linewidth=0.8, linestyle='--')
    ax.text(T_show - 2, delta + 0.003,
            r'$\delta = {:.1f}$'.format(delta),
            ha='right', fontsize=8, color='gray')

    ax.set_xlabel('Period $t$')
    ax.set_ylabel('Bargaining power')
    ax.set_title('Bargaining power paths:\nexogenous slow drift (DG) vs endogenous')

    # Legend placed below the panel to avoid covering any lines
    legend_elements = [
        Line2D([0], [0], color=BLUE, linewidth=1.4,
               label=r'DG: $\delta_t$ (three $\delta_0$ values)'),
        Line2D([0], [0], color=RED, linewidth=1.4,
               label=r'This model: $\lambda_t$ (three $W_0$ values)'),
    ]
    ax.legend(handles=legend_elements, fontsize=8,
              loc='lower center', bbox_to_anchor=(0.5, -0.38),
              ncol=1, frameon=True)

    plt.tight_layout()
    # Extra bottom margin for the external legend
    plt.subplots_adjust(bottom=0.28)
    savefig('fig_comparison_dg.png')


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    print("Running Spear-Srivastava VFI...")
    ss_result = run_vfi_ss(verbose=True)

    print("\nRunning DiGiannatale VFI...")
    dg_result = run_vfi_dg(eps=0.001, verbose=True)

    print("\nBuilding this model...")
    params = make_params()
    W_bar_val = w_bar_threshold(
        params['delta'], params['theta'], params['k'],
        params['gamma'], params['sigma'], params['w_bar_ll'],
        params['gamma_w'], params['y_lower'],
        params['u_p_bar'], params['u_a_bar'])
    W_min = 0.05
    W_max = W_bar_val + 6.0 * params['sigma'] / max(1 - params['rho']**2, 1e-6)**0.5
    W_grid_vals = np.linspace(W_min, W_max, params['n_grid'])
    this_result = run_vfi(W_grid_vals, W_bar_val, params, verbose=True)

    print("\nGenerating comparison figures...")
    fig_compare_ss(ss_result, this_result, params)
    fig_compare_dg(dg_result, this_result, params)
    print("Done.")