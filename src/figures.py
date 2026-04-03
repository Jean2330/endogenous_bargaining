# figures.py
# Single entry point. Run this file to regenerate all figures in the paper.
# Output: paper/figures/*.pdf
#
# Figure map:
#   fig_pareto_frontiers.pdf  -- slack (linear) vs binding (curved) frontiers [Section 3.4]
#   fig_policy_functions.pdf  -- beta*(W), lambda(W), Delta(W)                [Section 4.2]
#   fig_value_functions.pdf   -- VP(W), VA(W)                                 [Section 4.2]
#   fig_ergodic.pdf           -- ergodic distribution Phi*                    [Section 4.3]
#   fig_single_path.pdf       -- simulated W_t, lambda_t path                 [Section 4.3]
#   fig_cs_gamma.pdf          -- comparative statics in gamma                 [Section 4.4]
#   fig_cs_sigma.pdf          -- comparative statics in sigma                 [Section 4.4]
#   fig_cs_gamma_w.pdf        -- comparative statics in gamma_w               [Section 4.4]

import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))

import params as P
from model import (beta_eff, w_bar_threshold, disagreement_payoffs,
                   bargaining_distortion,
                   pareto_frontier_slack, pareto_frontier_binding,
                   nash_solution_slack, nash_solution_binding,
                   iso_nash_curve)
from vfi import run_vfi
from simulation import simulate_paths, single_path

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'paper', 'figures')
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# Parameter dict assembly
# ---------------------------------------------------------------------------

def make_params(**overrides):
    """
    Build a complete parameter dict from params.py with optional overrides.
    u_p_bar and u_a_bar are derived from primitives so they stay consistent
    whenever a primitive (theta, k, gamma, sigma, delta, frac) changes.
    """
    p = dict(
        theta=P.THETA, k=P.K, gamma=P.GAMMA, sigma=P.SIGMA,
        rho=P.RHO, delta=P.DELTA,
        w_bar_ll=P.W_BAR_LL, gamma_w=P.GAMMA_W, y_lower=P.Y_LOWER,
        frac_disagreement=P.FRAC_DISAGREEMENT,
        n_grid=P.N_GRID, n_quad=P.N_QUAD,
        vfi_tol=P.VFI_TOL, vfi_max_iter=P.VFI_MAX_ITER,
        t_sim=P.T_SIM, n_agents=P.N_AGENTS, t_burn=P.T_BURN,
    )
    p.update(overrides)
    u_p, u_a = disagreement_payoffs(
        p['theta'], p['k'], p['gamma'], p['sigma'],
        p['delta'], p['frac_disagreement'])
    p['u_p_bar'] = u_p
    p['u_a_bar'] = u_a
    return p


# ---------------------------------------------------------------------------
# Grid and regime-boundary construction
# ---------------------------------------------------------------------------

def build_grid_and_wbar(params):
    """
    Construct the wealth grid and identify W_bar (regime boundary).
    Grid runs from W_min to W_max = W_bar + buffer.
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
    n_grid   = params['n_grid']

    W_bar = w_bar_threshold(delta, theta, k, gamma, sigma,
                            w_bar_ll, gamma_w, y_lower, u_p_bar, u_a_bar)

    W_min = 0.05
    W_max = W_bar + 6.0 * sigma / np.sqrt(max(1 - rho**2, 1e-6))
    W_grid = np.linspace(W_min, W_max, n_grid)

    return W_grid, W_bar


# ---------------------------------------------------------------------------
# Matplotlib style
# ---------------------------------------------------------------------------

plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 12,
    'legend.fontsize': 10,
    'figure.dpi': 150,
    'lines.linewidth': 1.8,
})


def savefig(name):
    path = os.path.join(OUTPUT_DIR, name)
    plt.savefig(path, bbox_inches='tight')
    plt.close()
    print(f"  saved {path}")


# ---------------------------------------------------------------------------
# Figure 0: Pareto frontiers (slack vs binding)
# ---------------------------------------------------------------------------

def fig_pareto_frontiers(params, W_bar):
    """
    Two-panel figure.
    Left:  slack regime -- linear frontier, Nash point, iso-Nash curve.
    Right: binding regime at W = W_bar/2 -- curved frontier, Nash point,
           iso-Nash curve, with the slack frontier overlaid for reference.
    """
    theta    = params['theta']
    k        = params['k']
    gamma    = params['gamma']
    sigma    = params['sigma']
    delta    = params['delta']
    w_bar_ll = params['w_bar_ll']
    gamma_w  = params['gamma_w']
    y_lower  = params['y_lower']
    u_p_bar  = params['u_p_bar']
    u_a_bar  = params['u_a_bar']

    W_bind = W_bar * 0.5

    # Pareto frontiers
    Pi_slack, CE_slack = pareto_frontier_slack(
        theta, k, gamma, sigma, u_p_bar, u_a_bar)

    Pi_bind, CE_bind, _ = pareto_frontier_binding(
        W_bind, w_bar_ll, gamma_w, y_lower, theta, k, gamma, sigma,
        u_p_bar, u_a_bar)

    # Nash solutions
    Pi_N_sl, CE_N_sl = nash_solution_slack(
        delta, theta, k, gamma, sigma, u_p_bar, u_a_bar)

    Pi_N_bd, CE_N_bd, beta_N_bd = nash_solution_binding(
        W_bind, delta, w_bar_ll, gamma_w, y_lower, theta, k, gamma, sigma,
        u_p_bar, u_a_bar)

    # Iso-Nash curves
    Pi_iso_sl, CE_iso_sl = iso_nash_curve(Pi_N_sl, CE_N_sl, delta)
    Pi_iso_bd, CE_iso_bd = iso_nash_curve(Pi_N_bd, CE_N_bd, delta)

    S_bind = Pi_N_bd + CE_N_bd
    lam_bind = CE_N_bd / S_bind if S_bind > 0 else float('nan')

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))

    # Left panel: slack
    ax = axes[0]
    ax.plot(Pi_slack, CE_slack, color='steelblue', linewidth=2.0,
            label='Pareto frontier (slack)')
    ax.plot(Pi_iso_sl, CE_iso_sl, color='steelblue', linewidth=1.0,
            linestyle='--', label='Iso-Nash curve')
    ax.scatter([Pi_N_sl], [CE_N_sl], color='steelblue', zorder=5, s=60,
               label=r'$N^{\mathrm{slack}}$')
    ax.plot([0, Pi_N_sl], [0, CE_N_sl], color='gray',
            linewidth=0.8, linestyle=':', label='Ray from $d$')
    ax.scatter([0], [0], color='black', zorder=5, s=40)
    ax.annotate('$d$', xy=(0, 0), xytext=(0.003, 0.003), fontsize=10)
    ax.set_xlabel(r'Principal surplus $\Pi_s$')
    ax.set_ylabel(r'Agent surplus $\mathrm{CE}_s$')
    ax.set_title('Slack regime: linear frontier')
    ax.set_xlim(left=-0.005)
    ax.set_ylim(bottom=-0.005)
    ax.text(0.97, 0.05,
            r'$\lambda = \delta = {:.2f}$'.format(delta),
            transform=ax.transAxes, ha='right', fontsize=10, color='steelblue')
    ax.legend(fontsize=9)

    # Right panel: binding
    ax = axes[1]
    ax.plot(Pi_slack, CE_slack, color='steelblue', linewidth=1.2,
            linestyle='-', alpha=0.3, label='Slack frontier (reference)')
    ax.plot(Pi_bind, CE_bind, color='firebrick', linewidth=2.0,
            label='Pareto frontier (binding)')
    ax.plot(Pi_iso_bd, CE_iso_bd, color='firebrick', linewidth=1.0,
            linestyle='--', label='Iso-Nash curve')
    ax.scatter([Pi_N_bd], [CE_N_bd], color='firebrick', zorder=5, s=60,
               label=r'$N^{\mathrm{bind}}$')
    ax.plot([0, Pi_N_bd], [0, CE_N_bd], color='gray',
            linewidth=0.8, linestyle=':', label='Ray from $d$')
    ax.scatter([0], [0], color='black', zorder=5, s=40)
    ax.annotate('$d$', xy=(0, 0), xytext=(0.003, 0.003), fontsize=10)
    ax.set_xlabel(r'Principal surplus $\Pi_s$')
    ax.set_ylabel(r'Agent surplus $\mathrm{CE}_s$')
    ax.set_title(r'Binding regime: curved frontier ($W = \bar{W}/2$)')
    ax.set_xlim(left=-0.005)
    ax.set_ylim(bottom=-0.005)
    if not np.isnan(lam_bind):
        ax.text(0.97, 0.05,
                r'$\lambda = {:.3f} \neq \delta = {:.2f}$'.format(lam_bind, delta),
                transform=ax.transAxes, ha='right', fontsize=10, color='firebrick')
    ax.legend(fontsize=9)

    plt.tight_layout()
    savefig('fig_pareto_frontiers.pdf')
    print(f"    slack:   Pi_s={Pi_N_sl:.4f}  CE_s={CE_N_sl:.4f}"
          f"  lambda=delta={delta:.3f}")
    if not np.isnan(lam_bind):
        print(f"    binding: Pi_s={Pi_N_bd:.4f}  CE_s={CE_N_bd:.4f}"
              f"  lambda={lam_bind:.4f}  beta*={beta_N_bd:.4f}")
    else:
        print("    binding: feasible set empty at W_bar/2 -- check calibration")


# ---------------------------------------------------------------------------
# Figure 1: Policy functions
# ---------------------------------------------------------------------------

def fig_policy_functions(vfi_result, params):
    W_grid  = vfi_result['W_grid']
    beta_p  = vfi_result['beta']
    lam_p   = vfi_result['lambda']
    delta_p = vfi_result['delta_W']
    W_bar   = vfi_result['W_bar']
    b_eff   = vfi_result['beta_eff']
    delta   = params['delta']

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))

    ax = axes[0]
    ax.plot(W_grid, beta_p, color='black', label=r'$\beta^*(W)$')
    ax.axhline(b_eff, color='gray', linestyle='--',
               label=r'$\beta^{\mathrm{eff}}$')
    ax.axvline(W_bar, color='gray', linestyle=':', alpha=0.7,
               label=r'$\bar{W}$')
    ax.set_xlabel(r'Wealth $W$')
    ax.set_ylabel(r'Incentive slope $\beta^*(W)$')
    ax.set_title('Equilibrium slope')
    ax.legend()

    ax = axes[1]
    ax.plot(W_grid, lam_p, color='black', label=r'$\lambda(W)$')
    ax.axhline(delta, color='gray', linestyle='--', label=r'$\delta$')
    ax.axvline(W_bar, color='gray', linestyle=':', alpha=0.7,
               label=r'$\bar{W}$')
    ax.set_xlabel(r'Wealth $W$')
    ax.set_ylabel(r'Realized bargaining power $\lambda(W)$')
    ax.set_title('Realized bargaining power')
    ax.legend()

    ax = axes[2]
    ax.plot(W_grid, delta_p, color='black', label=r'$\Delta(W)$')
    ax.axvline(W_bar, color='gray', linestyle=':', alpha=0.7,
               label=r'$\bar{W}$')
    ax.set_xlabel(r'Wealth $W$')
    ax.set_ylabel(r'Bargaining distortion $\Delta(W)$')
    ax.set_title('Welfare distortion')
    ax.legend()

    plt.tight_layout()
    savefig('fig_policy_functions.pdf')


# ---------------------------------------------------------------------------
# Figure 2: Value functions
# ---------------------------------------------------------------------------

def fig_value_functions(vfi_result, params):
    W_grid = vfi_result['W_grid']
    VP     = vfi_result['VP']
    VA     = vfi_result['VA']
    W_bar  = vfi_result['W_bar']

    fig, axes = plt.subplots(1, 2, figsize=(9, 4))

    ax = axes[0]
    ax.plot(W_grid, VP, color='black')
    ax.axvline(W_bar, color='gray', linestyle=':', alpha=0.7,
               label=r'$\bar{W}$')
    ax.set_xlabel(r'Wealth $W$')
    ax.set_ylabel(r'$V_P(W)$')
    ax.set_title("Principal's value function")
    ax.legend()

    ax = axes[1]
    ax.plot(W_grid, VA, color='black')
    ax.axvline(W_bar, color='gray', linestyle=':', alpha=0.7,
               label=r'$\bar{W}$')
    ax.set_xlabel(r'Wealth $W$')
    ax.set_ylabel(r'$V_A(W)$')
    ax.set_title("Agent's value function")
    ax.legend()

    plt.tight_layout()
    savefig('fig_value_functions.pdf')


# ---------------------------------------------------------------------------
# Figure 3: Ergodic distribution
# ---------------------------------------------------------------------------

def fig_ergodic(sim_result, params):
    W     = sim_result['W'].flatten()
    W_bar = sim_result['W_bar']

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(W, bins=80, density=True, color='gray', alpha=0.7,
            label=r'Simulated $\Phi^*$')
    ax.axvline(W_bar, color='black', linestyle='--', label=r'$\bar{W}$')
    ax.set_xlabel(r'Wealth $W$')
    ax.set_ylabel('Density')
    ax.set_title('Ergodic distribution of wealth')
    ax.legend()

    plt.tight_layout()
    savefig('fig_ergodic.pdf')


# ---------------------------------------------------------------------------
# Figure 4: Single simulated path
# ---------------------------------------------------------------------------

def fig_single_path(path_result, params):
    W     = path_result['W']
    lam   = path_result['lambda']
    W_bar = path_result['W_bar']
    t     = np.arange(len(W))

    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

    ax = axes[0]
    ax.plot(t, W, color='black', linewidth=0.8)
    ax.axhline(W_bar, color='gray', linestyle='--', label=r'$\bar{W}$')
    ax.set_ylabel(r'Wealth $W_t$')
    ax.set_title('Simulated wealth and bargaining power path')
    ax.legend()

    ax = axes[1]
    ax.plot(t, lam, color='black', linewidth=0.8)
    ax.axhline(params['delta'], color='gray', linestyle='--',
               label=r'$\delta$')
    ax.set_xlabel('Period $t$')
    ax.set_ylabel(r'$\lambda_t$')
    ax.legend()

    plt.tight_layout()
    savefig('fig_single_path.pdf')


# ---------------------------------------------------------------------------
# Figures 5-7: Comparative statics
# ---------------------------------------------------------------------------

def fig_comparative_statics(param_name, param_values, labels,
                             output_name, params_baseline):
    fig, ax = plt.subplots(figsize=(7, 4))
    grays = ['black', 'dimgray', 'darkgray', 'silver']

    for val, lab, col in zip(param_values, labels, grays):
        p = make_params(**{param_name: val})
        W_grid, W_bar = build_grid_and_wbar(p)
        print(f"    {param_name}={val}  W_bar={W_bar:.3f}")
        res = run_vfi(W_grid, W_bar, p, verbose=False)
        sim = simulate_paths(res, p)
        W_flat = sim['W'].flatten()
        ax.hist(W_flat, bins=60, density=True, alpha=0.55,
                color=col, label=lab)
        ax.axvline(W_bar, color=col, linestyle=':', alpha=0.7)

    ax.set_xlabel(r'Wealth $W$')
    ax.set_ylabel('Density')
    ax.legend()
    plt.tight_layout()
    savefig(output_name)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    print("Building baseline model...")
    params = make_params()
    W_grid, W_bar = build_grid_and_wbar(params)
    print(f"  W_bar={W_bar:.4f}  u_p_bar={params['u_p_bar']:.5f}"
          f"  u_a_bar={params['u_a_bar']:.5f}")
    print(f"  grid [{W_grid[0]:.3f}, {W_grid[-1]:.3f}]")

    print("Generating Figure 0: Pareto frontiers")
    fig_pareto_frontiers(params, W_bar)

    print("Running VFI...")
    vfi_result = run_vfi(W_grid, W_bar, params, verbose=True)

    print("Generating Figure 1: policy functions")
    fig_policy_functions(vfi_result, params)

    print("Generating Figure 2: value functions")
    fig_value_functions(vfi_result, params)

    print("Simulating ergodic distribution...")
    sim_result = simulate_paths(vfi_result, params)

    print("Generating Figure 3: ergodic distribution")
    fig_ergodic(sim_result, params)

    print("Simulating single path...")
    path_result = single_path(vfi_result, params)

    print("Generating Figure 4: single path")
    fig_single_path(path_result, params)

    print("Generating Figure 5: comparative statics in gamma")
    fig_comparative_statics(
        'gamma', [0.5, 1.0, 2.0, 3.0],
        [r'$\gamma=0.5$', r'$\gamma=1.0$', r'$\gamma=2.0$', r'$\gamma=3.0$'],
        'fig_cs_gamma.pdf', params)

    print("Generating Figure 6: comparative statics in sigma")
    fig_comparative_statics(
        'sigma', [0.15, 0.3, 0.45, 0.6],
        [r'$\sigma=0.15$', r'$\sigma=0.30$',
         r'$\sigma=0.45$', r'$\sigma=0.60$'],
        'fig_cs_sigma.pdf', params)

    print("Generating Figure 7: comparative statics in gamma_w")
    fig_comparative_statics(
        'gamma_w', [0.05, 0.10, 0.20, 0.35],
        [r'$\gamma_w=0.05$', r'$\gamma_w=0.10$',
         r'$\gamma_w=0.20$', r'$\gamma_w=0.35$'],
        'fig_cs_gamma_w.pdf', params)

    print("Done. All figures written to paper/figures/")
