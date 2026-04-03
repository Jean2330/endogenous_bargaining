# params.py
# Baseline parameter values for the numerical implementation.
# Every number used in the paper lives here. No other file defines parameters.
# To run a comparative static, change a value here and rerun figures.py.

# ---------------------------------------------------------------------------
# Technology
# ---------------------------------------------------------------------------
THETA = 1.0       # productivity (normalization)
K = 2.0           # effort cost curvature: c(a) = (k/2)*a^2

# ---------------------------------------------------------------------------
# Preferences
# ---------------------------------------------------------------------------
GAMMA = 1.0       # agent CARA risk aversion coefficient
RHO = 0.95        # common discount factor (annual rate ~ 5%)

# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------
SIGMA = 0.3       # output volatility

# Effective lower bound on output for the LL constraint.
# With a_eff = beta_eff*theta/k ~ 0.424 and sigma = 0.3,
# mean output mu_y ~ 0.424. Setting Y_LOWER = -0.5 places the
# bound roughly 3.1 standard deviations below the mean, which
# ensures the LL constraint is operative while keeping the feasible
# set non-empty across the entire binding domain [0, W_bar].
Y_LOWER = -0.5

# ---------------------------------------------------------------------------
# Contract and bargaining
# ---------------------------------------------------------------------------
DELTA = 0.4       # exogenous Nash bargaining weight on agent

# ---------------------------------------------------------------------------
# Limited liability
# ---------------------------------------------------------------------------
# w_lower(W) = W_BAR_LL - GAMMA_W * W  (decreasing in W, wealthier agents
# face a lower floor, consistent with wealth as collateral).
#
# Calibration target: W_bar ~ 0.80 (interior, visible in all figures).
# With S_eff ~ 0.212 and FRAC_DISAGREEMENT = 0.3:
#   alpha_slack = u_a_bar - C(b_eff) + delta*S_eff ~ -0.037
#   W_bar solves: alpha_slack = w_lower(W_bar) - b_eff*Y_LOWER
#   => W_bar = (W_BAR_LL - alpha_slack - b_eff*Y_LOWER) / GAMMA_W
#   With W_BAR_LL=-0.30, GAMMA_W=0.20: W_bar ~ 0.80
FRAC_DISAGREEMENT = 0.3   # outside options as fraction of each party's Nash share
W_BAR_LL = -0.30          # intercept of LL floor
GAMMA_W = 0.20            # wealth elasticity of LL floor, must satisfy 0 < GAMMA_W < 1

# Outside options (u_p_bar, u_a_bar) are computed in model.py via
# disagreement_payoffs() as fixed fractions of S_eff, not from the
# autarky perpetuity. This avoids magnification by 1/(1-rho).

# ---------------------------------------------------------------------------
# Numerical settings
# ---------------------------------------------------------------------------
N_GRID = 300          # wealth grid points
N_QUAD = 10           # Gauss-Hermite quadrature nodes
VFI_TOL = 1e-6        # VFI convergence tolerance (sup-norm)
VFI_MAX_ITER = 2000   # maximum VFI iterations

# ---------------------------------------------------------------------------
# Simulation settings
# ---------------------------------------------------------------------------
T_SIM = 10000         # periods per agent
N_AGENTS = 1000       # number of agents for ergodic distribution
T_BURN = 500          # burn-in periods discarded
