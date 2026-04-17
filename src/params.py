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
RHO_W = 0.85      # Wealth depreciation factor in the law of motion:
                  #   W_{t+1} = RHO_W * W_t + alpha_t + beta_t * y_t - c(a_t)
                  #
                  # The thesis law of motion is W_{t+1} = W_t + alpha_t + beta_t*y_t - c(a_t),
                  # i.e. RHO_W = 1. That process does not produce a stationary ergodic
                  # distribution under the baseline calibration because the mean net
                  # transfer (alpha + beta*E[y] - c(a)) is positive when the LL constraint
                  # is often slack, causing W_t to drift upward without bound.
                  #
                  # Setting RHO_W = 0.85 < 1 introduces mean reversion that restores
                  # stationarity. This is a computational implementation assumption adopted
                  # to make the numerical model well-posed. It is not a primitive of the
                  # theoretical model and must be acknowledged explicitly in the paper.
                  # Ergodic mean W* ~ 0.95 under this calibration, just above W_bar.

# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------
SIGMA = 0.3       # output volatility
Y_LOWER = -0.5    # effective lower bound on output for LL constraint.
                  # With mean output mu_y ~ 0.42, this is approx 3.1 sigma below mean.

# ---------------------------------------------------------------------------
# Contract and bargaining
# ---------------------------------------------------------------------------
DELTA = 0.4       # exogenous Nash bargaining weight on agent

# ---------------------------------------------------------------------------
# Limited liability
# ---------------------------------------------------------------------------
# w_lower(W) = W_BAR_LL - GAMMA_W * W  (decreasing in W).
# Calibrated so W_bar ~ 0.93, an interior point of the wealth grid.
# With zero outside options, lambda_slack = delta exactly.
FRAC_DISAGREEMENT = 0.0   # outside options set to zero
                           # ensures lambda_slack = delta exactly
W_BAR_LL = -0.30          # intercept of LL floor
GAMMA_W = 0.20            # wealth elasticity of LL floor, must satisfy 0 < GAMMA_W < 1

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