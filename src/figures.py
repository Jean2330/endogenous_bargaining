# figures.py
# Single entry point. Run this file to regenerate all figures in the paper.
# Output: paper/figures/*.png
#
# Figure map:
#   fig_pareto_frontiers.png   -- slack (linear) vs binding (curved) frontiers [Section 3.4]
#   fig_policy_functions.png   -- beta*(W), lambda(W), Delta(W) at baseline    [Section 4.2]
#   fig_policy_by_delta.png    -- beta(W), alpha(W), lambda(W) by delta        [Section 4.2]
#   fig_value_functions.png    -- VP(W), VA(W)                                 [Section 4.2]
#   fig_drift.png              -- E[W'|W] - W by delta (mean reversion)        [Section 4.3]
#   fig_ergodic.png            -- ergodic distribution of W                    [Section 4.3]
#   fig_ergodic_lambda.png     -- ergodic distribution of lambda_t             [Section 4.3]
#   fig_single_path.png        -- W_t differences and lambda_t by delta        [Section 4.3]
#   fig_binding_prob.png       -- Pr(binding) by parameter                     [Section 4.4]
#   fig_cs_gamma.png           -- comparative statics in gamma                 [Section 4.4]
#   fig_cs_sigma.png           -- comparative statics in sigma                 [Section 4.4]
#   fig_cs_gamma_w.png         -- comparative statics in gamma_w               [Section 4.4]
#   fig_cs_delta.png           -- comparative statics in delta                 [Section 4.4]

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
from simulation import simulate_paths, single_path, single_path_from_shocks

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
        rho=P.RHO, rho_W=P.RHO_W, delta=P.DELTA,
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

# Shared palette used consistently across all multi-delta and multi-parameter figures.
# Four colors that are distinguishable in print and on screen.
PALETTE     = ['#1f77b4', '#d62728', '#2ca02c', '#ff7f0e']
LINESTYLES  = ['solid', 'dashed', 'dotted', 'dashdot']
DELTA_VALUES = [0.3, 0.4, 0.5, 0.6]


def savefig(name):
    # Accept either .pdf or .png name; always save as PNG at 200 dpi.
    name_png = os.path.splitext(name)[0] + '.png'
    path = os.path.join(OUTPUT_DIR, name_png)
    plt.savefig(path, bbox_inches='tight', dpi=200)
    plt.close()
    print(f"  saved {path}")


# ---------------------------------------------------------------------------
# Figure 0: Pareto frontiers (slack vs binding)
# ---------------------------------------------------------------------------

def fig_pareto_frontiers(params, W_bar):
    """
    Single-panel figure showing both Pareto frontiers overlaid.

    The slack frontier is linear (alpha is free, beta = beta_eff fixed).
    The binding frontier is the model-computed curve from pareto_frontier_binding,
    which traces (Pi_s(beta), CE_s(beta)) as beta varies over the feasible set.
    This is a genuine NTU frontier: as beta increases, more risk is transferred
    to the agent, so CE_s falls while Pi_s first rises then falls. The curve is
    generically concave toward the origin, a consequence of the moral hazard
    technology. No Bezier approximation is used.

    N_bind is computed at W = W_bar * 0.15 (deep binding) so the two Nash
    points are visibly separated. Two dotted rays from d show the different
    realized shares.
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

    W_bind = W_bar * 0.15

    Pi_slack, CE_slack = pareto_frontier_slack(
        theta, k, gamma, sigma, u_p_bar, u_a_bar)

    # Actual model-computed binding frontier, not a Bezier approximation.
    Pi_bind, CE_bind, _ = pareto_frontier_binding(
        W_bind, w_bar_ll, gamma_w, y_lower, theta, k, gamma, sigma,
        u_p_bar, u_a_bar)

    Pi_N_sl, CE_N_sl = nash_solution_slack(
        delta, theta, k, gamma, sigma, u_p_bar, u_a_bar)

    Pi_N_bd, CE_N_bd, beta_N_bd = nash_solution_binding(
        W_bind, delta, w_bar_ll, gamma_w, y_lower, theta, k, gamma, sigma,
        u_p_bar, u_a_bar)

    S_eff    = Pi_slack.max()
    S_bind   = Pi_N_bd + CE_N_bd
    lam_bind = CE_N_bd / S_bind if S_bind > 0 else float('nan')

    Pi_iso_sl, CE_iso_sl = iso_nash_curve(Pi_N_sl, CE_N_sl, delta)
    Pi_iso_bd, CE_iso_bd = iso_nash_curve(Pi_N_bd, CE_N_bd, delta)

    lim    = S_eff * 1.25
    margin = lim * 0.04

    mask_sl = ((Pi_iso_sl >= 0) & (Pi_iso_sl <= lim) &
               (CE_iso_sl >= 0) & (CE_iso_sl <= lim))
    mask_bd = ((Pi_iso_bd >= 0) & (Pi_iso_bd <= lim) &
               (CE_iso_bd >= 0) & (CE_iso_bd <= lim))

    fig, ax = plt.subplots(figsize=(7, 6))

    # Shade the binding feasible set using the actual model frontier.
    ax.fill(np.concatenate([Pi_bind, [0]]),
            np.concatenate([CE_bind, [0]]),
            color='lightgray', alpha=0.5, zorder=0,
            label='Feasible set (binding)')

    # Slack frontier and its iso-Nash curve.
    ax.plot(Pi_slack, CE_slack, color='steelblue', linewidth=2.2,
            label='Slack (linear)')
    ax.plot(Pi_iso_sl[mask_sl], CE_iso_sl[mask_sl],
            color='steelblue', linewidth=1.0, linestyle='--')

    # Binding frontier from the actual model.
    ax.plot(Pi_bind, CE_bind, color='firebrick', linewidth=2.2,
            label='Binding (curved)')
    ax.plot(Pi_iso_bd[mask_bd], CE_iso_bd[mask_bd],
            color='firebrick', linewidth=1.0, linestyle='--',
            label='Iso-Nash curves')

    # Nash points.
    ax.scatter([Pi_N_sl], [CE_N_sl], color='steelblue', zorder=6, s=70,
               label=r'$N^{\mathrm{slack}}$')
    ax.scatter([Pi_N_bd], [CE_N_bd], color='firebrick', zorder=6, s=70,
               label=r'$N^{\mathrm{bind}}$')

    # Rays from d through each Nash point.
    slope_sl = CE_N_sl / Pi_N_sl
    slope_bd = CE_N_bd / Pi_N_bd
    ax.plot([0, lim], [0, slope_sl * lim],
            color='steelblue', linewidth=0.8, linestyle=':')
    ax.plot([0, lim], [0, slope_bd * lim],
            color='firebrick', linewidth=0.8, linestyle=':')

    # Disagreement point d at the origin.
    ax.scatter([0], [0], color='black', zorder=6, s=50)
    ax.annotate('$d$', xy=(0, 0),
                xytext=(lim * 0.025, lim * 0.025), fontsize=11)

    ax.text(Pi_N_sl + lim * 0.03, CE_N_sl,
            r'$\lambda^{{\mathrm{{slack}}}} = \delta = {:.2f}$'.format(delta),
            fontsize=9, color='steelblue', va='center')

    if not np.isnan(lam_bind):
        ax.text(Pi_N_bd - lim * 0.02, CE_N_bd + lim * 0.04,
                r'$\lambda^{{\mathrm{{bind}}}} = {:.3f} \neq \delta$'.format(lam_bind),
                fontsize=9, color='firebrick', va='bottom', ha='right')

    ax.set_xlabel(r'Principal surplus $\Pi_s$')
    ax.set_ylabel(r'Agent surplus $\mathrm{CE}_s$')
    ax.set_title('Pareto frontiers: slack vs. binding regime')
    ax.set_xlim(-margin, lim)
    ax.set_ylim(-margin, lim)
    ax.legend(fontsize=9, loc='upper right')

    plt.tight_layout()
    savefig('fig_pareto_frontiers.png')
    print(f"    slack:   Pi_s={Pi_N_sl:.4f}  CE_s={CE_N_sl:.4f}"
          f"  lambda=delta={delta:.3f}")
    if not np.isnan(lam_bind):
        print(f"    binding: Pi_s={Pi_N_bd:.4f}  CE_s={CE_N_bd:.4f}"
              f"  lambda={lam_bind:.4f}  beta*={beta_N_bd:.4f}")


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
    savefig('fig_policy_functions.png')


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
    savefig('fig_value_functions.png')


# ---------------------------------------------------------------------------
# Figure 3: Ergodic distribution
# ---------------------------------------------------------------------------

def ergodic_hist_range(W_flat):
    """
    Return (lo, hi) histogram range that excludes the boundary-clip spike.

    When W_next is clipped to W_min, agents accumulate at exactly W_min = 0.05
    after bad shock sequences. This creates an isolated first bin disconnected
    from the main distribution. Setting the histogram range to start at the
    5th percentile removes that bin entirely without discarding data from the
    density estimate (density=True renormalises over the chosen range).
    """
    lo = np.percentile(W_flat, 5)
    hi = W_flat.max()
    return lo, hi


def fig_ergodic(sim_result, params):
    W     = sim_result['W'].flatten()
    W_bar = sim_result['W_bar']

    lo, hi = ergodic_hist_range(W)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(W, bins=80, density=True, range=(lo, hi),
            color='gray', alpha=0.7, label=r'Simulated $\Phi^*$')
    ax.axvline(W_bar, color='black', linestyle='--', label=r'$\bar{W}$')
    ax.set_xlabel(r'Wealth $W$')
    ax.set_ylabel('Density')
    ax.set_title('Ergodic distribution of wealth')
    ax.legend()

    plt.tight_layout()
    savefig('fig_ergodic.png')


# ---------------------------------------------------------------------------
# Figure 4: Single simulated path -- differences and lambda
# ---------------------------------------------------------------------------

def fig_single_path(vfi_result, params):
    """
    Two-panel figure using the shared shock sequence across all delta values.

    Top panel: W_t(delta) - W_t(delta=0.4), the wealth path relative to the
    baseline delta. This removes the common shock movement and shows the pure
    effect of the Nash weight on wealth dynamics. The baseline (delta=0.4)
    is a flat zero line by construction.

    Bottom panel: lambda_t for each delta. The four series are vertically
    separated because lambda = delta in the slack regime, so they fluctuate
    around different levels.

    Both panels use color plus linestyle for legibility in print.
    """
    T_show       = 200
    delta_base   = 0.4

    rng    = np.random.default_rng(seed=7)
    shocks = rng.standard_normal(T_show)
    t      = np.arange(T_show)

    # Simulate all four paths, storing results for differencing.
    paths = {}
    for delta_val in DELTA_VALUES:
        p_d = make_params(delta=delta_val)
        W_grid_d, W_bar_d = build_grid_and_wbar(p_d)
        print(f"    running VFI for delta={delta_val}...")
        vfi_d = run_vfi(W_grid_d, W_bar_d, p_d, verbose=False)
        path_d = single_path_from_shocks(vfi_d, p_d, shocks,
                                          W0=W_bar_d * 0.4)
        paths[delta_val] = {
            'W':     path_d['W'],
            'lam':   path_d['lambda'],
            'W_bar': W_bar_d,
        }

    W_base = paths[delta_base]['W']

    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    fig.suptitle('Simulated paths by Nash weight')

    for delta_val, col, ls in zip(DELTA_VALUES, PALETTE, LINESTYLES):
        lab = r'$\delta = {:.1f}$'.format(delta_val)
        W_diff = paths[delta_val]['W'] - W_base

        axes[0].plot(t, W_diff, color=col, linewidth=1.0,
                     linestyle=ls, label=lab)
        axes[1].plot(t, paths[delta_val]['lam'], color=col, linewidth=1.0,
                     linestyle=ls, label=lab)

    axes[0].axhline(0, color='black', linewidth=0.6, linestyle=':')
    axes[0].set_ylabel(r'$W_t(\delta) - W_t(\delta=0.4)$')
    axes[0].set_title('Wealth difference relative to baseline')
    axes[0].legend(fontsize=9, loc='upper right')

    axes[1].set_xlabel('Period $t$')
    axes[1].set_ylabel(r'$\lambda_t$')
    axes[1].set_title('Realized bargaining power')
    axes[1].legend(fontsize=9, loc='upper right')

    plt.tight_layout()
    savefig('fig_single_path.png')


# ---------------------------------------------------------------------------
# Figure 5: Policy functions by delta
# ---------------------------------------------------------------------------

def fig_policy_by_delta(params):
    """
    Three-panel figure showing beta(W), alpha(W), and lambda(W) for four
    values of delta. This is the most direct illustration of how the Nash
    weight changes the equilibrium contract and realized bargaining power
    across the wealth distribution. All panels share the same x-axis so the
    regime boundary W_bar is visible in each.
    """
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))

    for delta_val, col, ls in zip(DELTA_VALUES, PALETTE, LINESTYLES):
        p = make_params(delta=delta_val)
        W_grid, W_bar = build_grid_and_wbar(p)
        print(f"    running VFI for delta={delta_val}...")
        res = run_vfi(W_grid, W_bar, p, verbose=False)
        lab = r'$\delta={:.1f}$'.format(delta_val)

        axes[0].plot(W_grid, res['beta'],  color=col, linestyle=ls,
                     linewidth=1.4, label=lab)
        axes[0].axvline(W_bar, color=col, linewidth=0.5, linestyle=ls,
                        alpha=0.4)

        axes[1].plot(W_grid, res['alpha'], color=col, linestyle=ls,
                     linewidth=1.4, label=lab)
        axes[1].axvline(W_bar, color=col, linewidth=0.5, linestyle=ls,
                        alpha=0.4)

        axes[2].plot(W_grid, res['lambda'], color=col, linestyle=ls,
                     linewidth=1.4, label=lab)
        axes[2].axvline(W_bar, color=col, linewidth=0.5, linestyle=ls,
                        alpha=0.4)

    b_eff_val = beta_eff(params['theta'], params['k'],
                         params['gamma'], params['sigma'])
    axes[0].axhline(b_eff_val, color='gray', linewidth=0.8, linestyle=':',
                    label=r'$\beta^{\mathrm{eff}}$')

    axes[0].set_xlabel(r'Wealth $W$')
    axes[0].set_ylabel(r'$\beta^*(W)$')
    axes[0].set_title('Incentive slope')
    axes[0].legend(fontsize=8)

    axes[1].set_xlabel(r'Wealth $W$')
    axes[1].set_ylabel(r'$\alpha^*(W)$')
    axes[1].set_title('Fixed payment')
    axes[1].legend(fontsize=8)

    axes[2].set_xlabel(r'Wealth $W$')
    axes[2].set_ylabel(r'$\lambda(W)$')
    axes[2].set_title('Realized bargaining power')
    axes[2].legend(fontsize=8)

    plt.tight_layout()
    savefig('fig_policy_by_delta.png')


# ---------------------------------------------------------------------------
# Figure 6: Drift function
# ---------------------------------------------------------------------------

def fig_drift(vfi_result, params):
    """
    Plot the conditional mean drift E[W_{t+1} | W_t = W] - W as a function
    of W for four values of delta. This makes mean reversion visible: the
    drift is positive below the ergodic mean and negative above it. The zero
    crossing identifies the ergodic mean W*. The regime boundary W_bar for
    each delta is marked, showing how the boundary interacts with mean
    reversion.

    The drift is computed analytically from the law of motion:
        E[W' | W] = rho_W*W + alpha*(W) + beta*(W)*E[y] - c(a*(W))
    with E[y] = theta * a*(W) = theta * beta*(W) * theta / k.
    """
    from scipy.interpolate import CubicSpline

    fig, ax = plt.subplots(figsize=(8, 4))

    for delta_val, col, ls in zip(DELTA_VALUES, PALETTE, LINESTYLES):
        p = make_params(delta=delta_val)
        W_grid, W_bar = build_grid_and_wbar(p)
        print(f"    computing drift for delta={delta_val}...")
        res = run_vfi(W_grid, W_bar, p, verbose=False)

        theta = p['theta']
        k     = p['k']
        rho_W = p['rho_W']

        beta_arr  = res['beta']
        alpha_arr = res['alpha']
        a_arr     = beta_arr * theta / k
        Ey        = theta * a_arr
        c_arr     = (k / 2) * a_arr**2

        E_W_next  = rho_W * W_grid + alpha_arr + beta_arr * Ey - c_arr
        drift     = E_W_next - W_grid

        lab = r'$\delta={:.1f}$, $\bar{{W}}={:.2f}$'.format(delta_val, W_bar)
        ax.plot(W_grid, drift, color=col, linestyle=ls, linewidth=1.4,
                label=lab)
        ax.axvline(W_bar, color=col, linewidth=0.5, linestyle=ls, alpha=0.4)

    ax.axhline(0, color='black', linewidth=0.7, linestyle=':')
    ax.set_xlabel(r'Wealth $W$')
    ax.set_ylabel(r'$E[W_{t+1}\mid W_t=W] - W$')
    ax.set_title('Conditional mean drift')
    ax.legend(fontsize=8)
    plt.tight_layout()
    savefig('fig_drift.png')


# ---------------------------------------------------------------------------
# Figure 7: Ergodic distribution of lambda
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Figure 7: Ergodic distribution of lambda (split by regime) and log-odds
# ---------------------------------------------------------------------------

def fig_ergodic_lambda(sim_result, params):
    """
    Two-panel figure for the ergodic distribution of bargaining power.

    Left panel: histogram of lambda_t split by regime. The binding and slack
    observations are plotted as separate overlaid histograms so the reader
    can see immediately that the spike at lambda = delta is not a numerical
    artifact but the entire slack-regime mass. The binding distribution is
    concentrated above delta and spreads toward higher lambda values as W
    falls deeper into the binding region.

    Right panel: histogram of the log-odds ell_t = log(lambda / (1 - lambda)).
    This is the natural domain for the logit-transformed bargaining power that
    appears in the drift equation of the thesis. In the slack regime ell_t is
    exactly log(delta / (1 - delta)) (a point mass). In the binding regime it
    spreads above that value.
    """
    lam     = sim_result['lambda'].flatten()
    ell     = sim_result['ell'].flatten()
    binding = sim_result['binding'].flatten()
    delta   = params['delta']
    ell_delta = float(np.log(delta / (1 - delta)))

    lam_bind  = lam[binding]
    lam_slack = lam[~binding]
    ell_bind  = ell[binding]
    ell_slack = ell[~binding]

    frac_bind = binding.mean()

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    # Left panel: lambda histogram split by regime.
    ax = axes[0]
    lo = np.percentile(lam_bind, 1) if len(lam_bind) > 0 else delta
    hi = lam.max()

    if len(lam_bind) > 0:
        ax.hist(lam_bind, bins=60, density=False, weights=np.ones(len(lam_bind)) / len(lam),
                range=(lo, hi), color='firebrick', alpha=0.6,
                label='Binding ({:.0f}%)'.format(100 * frac_bind))
    if len(lam_slack) > 0:
        ax.hist(lam_slack, bins=60, density=False, weights=np.ones(len(lam_slack)) / len(lam),
                range=(lo, hi), color='steelblue', alpha=0.6,
                label='Slack ({:.0f}%)'.format(100 * (1 - frac_bind)))

    ax.axvline(delta, color='black', linewidth=1.0, linestyle='--',
               label=r'$\delta = {:.1f}$'.format(delta))
    ax.set_xlabel(r'Realized bargaining power $\lambda_t$')
    ax.set_ylabel('Fraction of observations')
    ax.set_title('Ergodic distribution of bargaining power')
    ax.legend(fontsize=9)

    # Right panel: log-odds histogram split by regime.
    ax = axes[1]
    lo_e = np.percentile(ell_bind, 1) if len(ell_bind) > 0 else ell_delta
    hi_e = ell.max()

    if len(ell_bind) > 0:
        ax.hist(ell_bind, bins=60, density=False, weights=np.ones(len(ell_bind)) / len(ell),
                range=(lo_e, hi_e), color='firebrick', alpha=0.6,
                label='Binding')
    if len(ell_slack) > 0:
        ax.hist(ell_slack, bins=60, density=False, weights=np.ones(len(ell_slack)) / len(ell),
                range=(lo_e, hi_e), color='steelblue', alpha=0.6,
                label='Slack')

    ax.axvline(ell_delta, color='black', linewidth=1.0, linestyle='--',
               label=r'$\ell(\delta) = {:.2f}$'.format(ell_delta))
    ax.set_xlabel(r'Log-odds $\ell_t = \log(\lambda_t / (1-\lambda_t))$')
    ax.set_ylabel('Fraction of observations')
    ax.set_title('Ergodic distribution of log-odds bargaining power')
    ax.legend(fontsize=9)

    plt.tight_layout()
    savefig('fig_ergodic_lambda.png')


# ---------------------------------------------------------------------------
# Figure 8: Binding probability comparative statics
# ---------------------------------------------------------------------------

def fig_binding_prob(params):
    """
    Plot Pr(W_t < W_bar) -- the stationary probability of being in the binding
    regime -- as a function of each of four parameters: delta, gamma, sigma,
    gamma_w. Each sub-figure is a line plot computed from the simulated ergodic
    distribution. This is one of the most interpretable comparative statics
    figures: it translates the distribution shift into a single scalar summary.
    """
    param_configs = [
        ('delta',   [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
         r'Nash weight $\delta$'),
        ('gamma',   [0.5, 1.0, 1.5, 2.0, 2.5, 3.0],
         r'Risk aversion $\gamma$'),
        ('sigma',   [0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45],
         r'Output volatility $\sigma$'),
        ('gamma_w', [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35],
         r'LL elasticity $\gamma_w$'),
    ]

    fig, axes = plt.subplots(1, 4, figsize=(14, 4))

    for ax, (pname, pvals, xlabel) in zip(axes, param_configs):
        probs = []
        for val in pvals:
            p = make_params(**{pname: val})
            W_grid, W_bar = build_grid_and_wbar(p)
            print(f"    {pname}={val:.2f}  W_bar={W_bar:.3f}")
            res = run_vfi(W_grid, W_bar, p, verbose=False)
            sim = simulate_paths(res, p)
            W_flat = sim['W'].flatten()
            prob_bind = float((W_flat < W_bar).mean())
            probs.append(prob_bind)
        ax.plot(pvals, probs, color='black', linewidth=1.4, marker='o',
                markersize=4)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(r'$\Pr(W_t < \bar{W})$')
        ax.set_ylim(0, 1)

    plt.tight_layout()
    savefig('fig_binding_prob.png')


# ---------------------------------------------------------------------------
# Figure 9: Joint distribution of (W_t, lambda_t)
# ---------------------------------------------------------------------------

def fig_joint_distribution(sim_result, params):
    """
    Density scatter of (W_t, lambda_t) from the ergodic simulation.

    Each point is one agent-period observation. The color encodes local
    density (kernel density estimate on a grid) so the reader sees where
    the joint distribution concentrates. The vertical line marks W_bar.
    The horizontal dashed line marks delta.

    This figure proves the mechanism visually: wealth maps into bargaining
    power through the policy function lambda(W). In the slack regime all
    mass collapses onto the horizontal line lambda = delta. In the binding
    regime the mass spreads upward and to the left, tracing the policy curve.
    """
    from scipy.stats import gaussian_kde

    W   = sim_result['W'].flatten()
    lam = sim_result['lambda'].flatten()
    W_bar = sim_result['W_bar']
    delta = params['delta']

    # Subsample for kde speed; 20000 points is enough for a smooth estimate.
    rng = np.random.default_rng(42)
    idx = rng.choice(len(W), size=min(20000, len(W)), replace=False)
    W_s   = W[idx]
    lam_s = lam[idx]

    # Kernel density estimate on a grid for the color layer.
    kde    = gaussian_kde(np.vstack([W_s, lam_s]))
    Wg     = np.linspace(W_s.min(), W_s.max(), 120)
    lg     = np.linspace(lam_s.min(), lam_s.max(), 120)
    Wgg, lgg = np.meshgrid(Wg, lg)
    Z      = kde(np.vstack([Wgg.ravel(), lgg.ravel()])).reshape(Wgg.shape)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.contourf(Wgg, lgg, Z, levels=30, cmap='Blues')
    ax.contour( Wgg, lgg, Z, levels=8,  colors='white',
                linewidths=0.4, alpha=0.5)

    ax.axvline(W_bar, color='firebrick', linewidth=1.2, linestyle='--',
               label=r'$\bar{W}$')
    ax.axhline(delta, color='black', linewidth=0.8, linestyle=':',
               label=r'$\delta = {:.1f}$'.format(delta))

    ax.set_xlabel(r'Wealth $W_t$')
    ax.set_ylabel(r'Realized bargaining power $\lambda_t$')
    ax.set_title('Joint ergodic distribution of wealth and bargaining power')
    ax.legend(fontsize=9)

    plt.tight_layout()
    savefig('fig_joint_distribution.png')


# ---------------------------------------------------------------------------
# Figure 10: Heatmap of lambda(W, delta)
# ---------------------------------------------------------------------------

def fig_heatmap_lambda(params):
    """
    Heatmap of the policy function lambda(W; delta) across a grid of W and
    delta values. Each row corresponds to one delta, each column to one W.
    The color encodes realized bargaining power.

    This is the most compact representation of the main result: for any
    (W, delta) pair in the binding region, lambda(W; delta) > delta, and
    the deviation grows as W falls. The white contour marks lambda = delta
    (the boundary between amplification and no distortion). The diagonal
    pattern shows how the binding region shifts as delta changes.
    """
    delta_grid = np.linspace(0.2, 0.8, 25)
    n_W_show   = 120

    # Collect lambda(W; delta) for each delta. We trim each W grid to a
    # common range for the heatmap.
    W_common = np.linspace(0.05, 3.0, n_W_show)
    lam_mat  = np.full((len(delta_grid), n_W_show), np.nan)
    wbar_vec = np.zeros(len(delta_grid))

    from scipy.interpolate import CubicSpline

    for i, delta_val in enumerate(delta_grid):
        p = make_params(delta=delta_val)
        W_grid, W_bar = build_grid_and_wbar(p)
        wbar_vec[i] = W_bar
        print(f"    heatmap VFI delta={delta_val:.2f}  W_bar={W_bar:.3f}")
        res   = run_vfi(W_grid, W_bar, p, verbose=False)
        spl   = CubicSpline(W_grid, res['lambda'], extrapolate=True)
        lam_mat[i, :] = np.clip(spl(W_common), 0.0, 1.0)

    fig, ax = plt.subplots(figsize=(8, 5))

    im = ax.imshow(
        lam_mat,
        origin='lower',
        aspect='auto',
        extent=[W_common[0], W_common[-1],
                delta_grid[0], delta_grid[-1]],
        cmap='RdBu_r',
        vmin=0.2, vmax=0.8,
    )
    cbar = fig.colorbar(im, ax=ax, label=r'$\lambda(W;\delta)$')

    # Overlay the W_bar curve (boundary between binding and slack).
    ax.plot(wbar_vec, delta_grid, color='black', linewidth=1.4,
            linestyle='--', label=r'$\bar{W}(\delta)$')

    # Contour where lambda = delta (no distortion line).
    # Build a matrix of (lambda - delta) and contour at zero.
    delta_mat = np.tile(delta_grid[:, np.newaxis], (1, n_W_show))
    ax.contour(W_common, delta_grid, lam_mat - delta_mat,
               levels=[0.0], colors='white', linewidths=1.2,
               linestyles='solid')

    ax.set_xlabel(r'Wealth $W$')
    ax.set_ylabel(r'Nash weight $\delta$')
    ax.set_title(r'Realized bargaining power $\lambda(W;\delta)$')
    ax.legend(fontsize=9, loc='upper right')

    plt.tight_layout()
    savefig('fig_heatmap_lambda.png')


# ---------------------------------------------------------------------------
# Figures 11-14: Comparative statics (ergodic wealth distributions)
# ---------------------------------------------------------------------------

def fig_comparative_statics(param_name, param_values, labels,
                             output_name, params_baseline):
    """
    Overlay ergodic distributions for several values of param_name.
    The histogram range is set to (5th percentile, max) to suppress the
    boundary-clip spike at W_min without discarding data from the density.
    """
    fig, ax = plt.subplots(figsize=(7, 4))
    grays = ['black', 'dimgray', 'darkgray', 'silver']

    for val, lab, col in zip(param_values, labels, grays):
        p = make_params(**{param_name: val})
        W_grid, W_bar = build_grid_and_wbar(p)
        print(f"    {param_name}={val}  W_bar={W_bar:.3f}")
        res     = run_vfi(W_grid, W_bar, p, verbose=False)
        sim     = simulate_paths(res, p)
        W_flat  = sim['W'].flatten()
        lo, hi  = ergodic_hist_range(W_flat)
        ax.hist(W_flat, bins=60, density=True, range=(lo, hi),
                alpha=0.55, color=col, label=lab)
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

    print("Generating Figure 1: policy functions (baseline)")
    fig_policy_functions(vfi_result, params)

    print("Generating Figure 2: policy functions by delta")
    fig_policy_by_delta(params)

    print("Generating Figure 3: value functions")
    fig_value_functions(vfi_result, params)

    print("Generating Figure 4: drift function by delta")
    fig_drift(vfi_result, params)

    print("Simulating ergodic distribution...")
    sim_result = simulate_paths(vfi_result, params)

    print("Generating Figure 5: ergodic distribution of wealth")
    fig_ergodic(sim_result, params)

    print("Generating Figure 6: ergodic distribution of lambda")
    fig_ergodic_lambda(sim_result, params)

    print("Generating Figure 7: single path comparison by delta")
    fig_single_path(vfi_result, params)

    print("Generating Figure 8: binding probability comparative statics")
    fig_binding_prob(params)

    print("Generating Figure 9: joint distribution of (W_t, lambda_t)")
    fig_joint_distribution(sim_result, params)

    print("Generating Figure 10: heatmap of lambda(W, delta)")
    fig_heatmap_lambda(params)

    print("Generating Figure 11: comparative statics in gamma")
    fig_comparative_statics(
        'gamma', [0.5, 1.0, 2.0, 3.0],
        [r'$\gamma=0.5$', r'$\gamma=1.0$', r'$\gamma=2.0$', r'$\gamma=3.0$'],
        'fig_cs_gamma.png', params)

    print("Generating Figure 12: comparative statics in sigma")
    fig_comparative_statics(
        'sigma', [0.15, 0.25, 0.35, 0.45],
        [r'$\sigma=0.15$', r'$\sigma=0.25$',
         r'$\sigma=0.35$', r'$\sigma=0.45$'],
        'fig_cs_sigma.png', params)

    print("Generating Figure 13: comparative statics in gamma_w")
    fig_comparative_statics(
        'gamma_w', [0.05, 0.10, 0.20, 0.35],
        [r'$\gamma_w=0.05$', r'$\gamma_w=0.10$',
         r'$\gamma_w=0.20$', r'$\gamma_w=0.35$'],
        'fig_cs_gamma_w.png', params)

    print("Generating Figure 14: comparative statics in delta")
    fig_comparative_statics(
        'delta', [0.2, 0.4, 0.6, 0.8],
        [r'$\delta=0.2$', r'$\delta=0.4$',
         r'$\delta=0.6$', r'$\delta=0.8$'],
        'fig_cs_delta.png', params)

    print("Done. All figures written to paper/figures/")