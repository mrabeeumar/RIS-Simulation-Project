# RIS-Assisted NOMA Communication Simulator — Implementation Plan

*Refined through 3 rounds of independent external review (each formula and
algorithm below was checked for internal consistency and mathematical
correctness against standard downlink-NOMA/RIS literature conventions). This
is the authoritative spec — implementation should not deviate from it without
updating this document first.*

## 0. Goal
Build a research-grade Python simulator + Streamlit dashboard for a RIS-assisted
downlink NOMA multi-user wireless network, with configurable RIS optimization
algorithms, NOMA power allocation, imperfect-SIC modeling, user pairing, and a
full 8-experiment suite. Every formula and algorithm below is fully specified —
no implementer judgment calls should be needed.

## 1. Tech stack
Python 3.11+, NumPy, SciPy, CVXPY (ECOS/SCS backend), Matplotlib/Plotly, Streamlit,
pandas, pytest. Optional later: PyTorch/RL for controller learning, numba for speed.

## 2. Repository layout
```
ris_noma_sim/
  core/
    config.py            # SimConfig dataclass (single source of truth, shared by dashboard + experiments)
    channel.py            # Rayleigh/Rician generation, path loss, effective channel
    ris.py                 # RIS class: phase vectors, quantize_phase()
    pairing.py             # NOMA user clustering/pairing
    noma.py                 # power allocation application, superposition signal
    sic.py                   # SIC decoding order, K-user SINR, perfect/imperfect residual
    metrics.py             # sum-rate, EE, fairness (Jain), outage, BER
  optimization/
    random_phase.py
    fixed_phase.py
    max_snr.py
    max_sumrate.py           # alternating optimization / coordinate ascent
    fairness_aware.py       # alpha*Rsum + beta*Jain
    power_allocation.py      # NOMA power split optimizer (closed-form + CVXPY fallback)
    objective.py              # shared evaluate(channels, theta, config) -> sum-rate/user-rates,
                              # used by max_sumrate.py/fairness_aware.py's search loop AND later
                              # reused as-is by network/controller.py's final metrics pass, so the
                              # two never compute rates via separate code paths
  network/
    topology.py              # BS/RIS/user placement, distances
    mobility.py                # random-waypoint mobility model
    controller.py              # NetworkController: stateful, drives reconfiguration (Exp. 8)
  experiments/
    exp1_ris_gain.py .. exp8_mobility.py, run_all.py
  dashboard/
    app.py
    components/
  tests/
    test_channel.py, test_pairing.py, test_sic.py, test_metrics.py,
    test_ris_opt.py, test_controller.py, test_config.py
  results/                     # generated CSV/plots (gitignored)
  configs/default.yaml
  README.md
  requirements.txt
```

## 3. Shared config schema (core/config.py)
```python
@dataclass(frozen=True)
class SimConfig:
    n_ris: int = 64                     # RIS elements: 0 (no-RIS baseline, Exp.1 only) or 16-256
    n_users: int = 4                    # 2-10
    snr_db: float = 15.0                # -10 to 30
    ris_bits: int | str = 2             # 1, 2, 3, 4, or "continuous"
    ris_algo: str = "max_sumrate"       # random|fixed|max_snr|max_sumrate|fairness_aware
    power_algo: str = "fair_constrained" # fixed|inverse_gain|fair_constrained|max_sumrate_qos
    channel_type: str = "rician"        # rayleigh|rician
    rician_k_factor: float = 5.0        # dB, used only if channel_type == "rician"
    path_loss_exponent: float = 2.0     # tuned so d_max_m stays non-degenerate across the SNR range, see Sec 4
    cluster_size: int = 2               # NOMA pairing group size; == n_users means full-K NOMA
    sic_epsilon: float = 0.0            # 0 = perfect SIC, 0.05/0.10/0.20 = imperfect
    modulation: str = "bpsk"            # bpsk|qpsk
    outage_rate_threshold_bps_hz: float = 0.5
    fairness_alpha: float = 0.5         # for fairness_aware RIS algo
    fairness_beta: float = 0.5
    p_element_w: float = 0.005          # RIS element static power (5 mW, literature-typical)
    p_bs_static_w: float = 1.0          # BS circuit static power
    user_distances_m: tuple[float, ...] | None = None  # None => drawn from topology defaults
    tx_power_w: float = 1.0             # P, fixed transmit power in Watts (single source of truth, see Sec 10)
    direct_link_blocked: bool = False   # True => h_BU,k forced to 0 (used by Exp.1's RIS-only baselines)
    ris_offset_m: float = 10.0          # BS-to-RIS distance
    d_min_m: float = 5.0                # min RIS-to-user distance for random placement
    d_max_m: float = 15.0               # max RIS-to-user distance (kept non-degenerate, see Sec 4 sanity check)
    power_alloc_fixed_weak: float = 0.7 # a_weak for power_algo="fixed" (2-user case); a_strong = 1 - this
    reconfig_mode: str = "sinr_drop"    # "sinr_drop"|"periodic", used by NetworkController (Sec 8)
    reconfig_sinr_drop_db: float = 3.0
    reconfig_period_s: float = 5.0
    user_speed_mps: float = 1.0         # mobility model speed (Sec 8, Experiment 8 only)
    mobility_dt_s: float = 0.1
    mobility_sim_duration_s: float = 60.0
    n_trials: int = 500
    seed: int = 42

    def __post_init__(self):
        if self.n_ris != 0 and not (16 <= self.n_ris <= 256):
            raise ValueError("n_ris must be 0 (no-RIS baseline) or in [16, 256]")
        if not (2 <= self.n_users <= 10):
            raise ValueError("n_users must be in [2, 10]")
        if self.ris_bits not in (1, 2, 3, 4, "continuous"):
            raise ValueError("ris_bits must be 1, 2, 3, 4, or 'continuous'")
        # additional field-range checks follow the same pattern (snr_db, sic_epsilon in [0,1], etc.)
```
Both `network.controller.simulate(config)` (single Monte-Carlo batch) and the
dashboard bind directly to this dataclass — no parallel parameter definitions
anywhere else. Every module takes an explicit `rng: np.random.Generator`
argument; **no global `np.random.seed` calls anywhere in the codebase**.
Experiment scripts derive per-trial seeds via
`np.random.Generator(np.random.PCG64(seed=config.seed))` and record `config.seed`
in the output CSV metadata so any run is exactly reproducible. Dashboard "Live"
mode exposes an editable seed (default fixed) for reproducible screenshots.

## 4. Layer 1 — Channel model
- BS: M antennas, default M=1 (SISO), extendable to MISO without API change (H_BR becomes N x M).
- H_BR (N x M): BS -> RIS, Rician(K-factor) or Rayleigh per `channel_type`.
- h_RU,k (N x 1): RIS -> user k, same fading family.
- h_BU,k (1 x M): direct BS -> user k; settable to exactly 0 via config flag `direct_link_blocked: bool` to force RIS-dependence for Experiment 1's "No RIS" baseline meaning "no direct link and no RIS help" vs "direct link only, no RIS".
- Path loss: L(d) = (d / d0)^-path_loss_exponent, d0 = 1 m reference; applied as sqrt(L(d)) scaling on the corresponding small-scale fading coefficient.
- Rician fading composition: `h = sqrt(K/(K+1))*los + sqrt(1/(K+1))*nlos`, `nlos ~ CN(0,1)`, `K` = linear Rician factor from `rician_k_factor` (dB). **The LOS component's phase must be drawn independently per element/link, fresh each trial** (`los = exp(j*U(0,2*pi))`, not a shared constant such as phase 0 for every element) -- since this simulator does not model a physical antenna-array steering vector, a shared/constant LOS phase across RIS elements is a modeling artifact that makes `theta=0` ("no RIS intelligence") accidentally near-optimal, defeating the entire premise of every RIS-optimization experiment. This was caught empirically during implementation (Milestone 2): with a constant LOS phase, the Fixed baseline outperformed Random RIS by ~5x on average in `test_ris_opt.py`'s statistical-ordering test, which should never happen.
- Effective channel: h_k = h_BU,k + h_RU,k^H @ Theta @ H_BR, Theta = diag(exp(j*theta)).
- Regenerated fresh every Monte-Carlo trial using the trial's `rng`; user positions/distances fixed per experiment sweep (only fading regenerates per trial) unless the experiment explicitly sweeps distance (Experiment 7) or mobility (Experiment 8).
- **`n_ris == 0` (no-RIS baseline, used only by Experiment 1) is an explicit,
  tested special case, not an incidental NumPy degeneracy**: `H_BR` and
  `h_RU,k` are empty (shape `(0, M)` / `(0,)`) arrays, `Theta` is a 0x0 matrix,
  and `channel.py` must special-case this to skip the RIS term entirely and
  return `h_k = h_BU,k` directly (with `direct_link_blocked=False` required
  for this baseline to be non-trivial — `n_ris=0` with the direct link also
  blocked is an invalid/degenerate config and rejected by validation).
  `tests/test_channel.py` includes an explicit `n_ris=0` case confirming this.

### Default topology (network/topology.py)
BS fixed at origin (0,0). RIS fixed at (`ris_offset_m`, 0), default 10 m from
BS (config field `ris_offset_m: float = 10.0`). Users placed uniformly at
random within an annulus around the RIS: distance from RIS drawn uniformly in
`[d_min_m, d_max_m]` = `[5.0, 15.0]` (config fields, see the numeric sanity
check below for why these values), angle drawn uniformly in
`[0, 2*pi)`, independently per user per trial — unless `user_distances_m` is
explicitly set (Experiment 7), in which case those exact RIS-to-user distances
are used (angle still randomized, or fixed at 0 for deterministic 1-D
placement — Experiment 7 uses angle=0 for all users, i.e. colinear placement,
to isolate the distance effect cleanly). BS-to-user distance for the direct
link is computed geometrically from BS, RIS, and user positions (law of
cosines), not independently specified.

**Numeric sanity check (verified numerically, not just asserted — a first
attempt at these defaults, `ris_offset_m=30`, `d_max_m=25`,
`path_loss_exponent=2.2`, was actually computed during implementation and
measured at -13 dB, failing this very check, which is exactly the degeneracy
this section warns about)**: at `n_ris=64`, `snr_db=15`, with per-element
phase coherently aligned to a single far user (best-case Max-SNR bound, no
optimization-algorithm imperfection), the resulting SINR must exceed -5 dB
for a user at `d_max_m` — i.e., the *far* end of the annulus must still show
a measurable, non-floor-clipped rate at a mid-range SNR setting. With the
defaults above (`ris_offset_m=10`, `d_min_m=5`, `d_max_m=15`,
`path_loss_exponent=2.0`, `rician_k_factor=5.0` dB), empirically measured
mean SINR at `d_max_m` over 50 trials is ~6.7 dB with a worst-observed-trial
value of ~5.2 dB — a solid margin above the -5 dB floor. `test_channel.py`
(`test_geometry_sanity_far_user_snr_not_degenerate`) encodes this exact check
as a regression guard: it is not a hypothetical, it caught a real defect in
this plan's first-draft defaults.

## 5. Layer 2 — RIS controller & quantization
- `quantize_phase(theta: np.ndarray, bits: int | Literal["continuous"]) -> np.ndarray`:
  if "continuous", return theta unchanged; else
  `theta_q = round(theta / (2*pi/2**bits)) * (2*pi/2**bits)`.
- **Rule for all 5 optimization algorithms**: each algorithm first computes/searches
  a continuous-phase solution (closed form or coordinate ascent), then applies
  `quantize_phase` as the final step. Exhaustive search over the discrete phase
  set is used only as a validation baseline for small N (<=8) in unit tests, not
  in the main algorithms (avoids combinatorial blow-up).
- `ris_bits` accepts 1/2/3/4/"continuous" — the dashboard slider exposes 1/2/3-bit
  and "continuous" (per original spec) while Experiment 2's sweep uses the full
  1/2/3/4-bit/continuous set directly via config, bypassing the dashboard widget
  range. This reconciles the dashboard-vs-experiment resolution range mismatch.

## 6. Layer 3 — NOMA clustering, superposition, power allocation

### 6.1 User pairing / clustering (core/pairing.py)
```python
def pair_users(channel_gains: np.ndarray, cluster_size: int = 2) -> list[list[int]]:
    """Sort users by descending |h_k|^2, then group into clusters of
    `cluster_size` by strong-weak interleaving: cluster i gets the i-th
    strongest and i-th weakest remaining user (standard NOMA pairing).
    If cluster_size == n_users, returns a single cluster (full-K NOMA).

    If n_users is not evenly divisible by cluster_size (e.g. n_users=3,
    cluster_size=2, required by Experiment 4's sweep), the single leftover
    user (the one left unmatched after strong-weak interleaving exhausts
    pairs) forms its own singleton cluster. A singleton cluster carries no
    superposition/SIC (K'=1: SINR_1 = P*|h_1|^2/N0, no interference term) and
    still occupies one full share of the orthogonal resource, i.e. it counts
    toward n_clusters in the Sec 6.1 `1/n_clusters` normalization exactly
    like any other cluster."""
```
Default `cluster_size = 2`: users are paired strong-weak, each pair transmits on
its own orthogonal RIS-serving time/frequency slot (i.e., NOMA-within-pair,
OMA-across-pairs — this is the standard literature configuration and matches the
dashboard's "Strong user / Weak user" 2-user diagram). `cluster_size = n_users`
enables full-K superposition SIC for experiments/algorithms that want it.

**System-level sum-rate normalization (must match the OMA baseline's
convention for a fair Experiment 5 comparison)**: because clusters occupy
orthogonal slots, each cluster's per-user rates are scaled by its share of
the total resource, `1/n_clusters`, before summing:
`R_sum = sum_over_clusters( (1/n_clusters) * sum_{m in cluster} log2(1+SINR_m) )`.
This is the system-level sum-rate used everywhere in this plan (Sec 6.3, Sec
10, all experiments) — the un-normalized per-cluster figure from Sec 6.3 is an
intermediate quantity only. With `n_clusters=1` (i.e. `cluster_size == n_users`,
full-K NOMA reusing the whole resource) the factor is 1 and no scaling
applies, which is the correct comparator against OMA's `1/n_users` per-user
split in that mode too (`n_clusters=1` there still means all K users share one
resource via superposition, same as OMA sharing one resource via time-division
— both normalized to the same total resource budget).

### 6.2 Superposition signal (per cluster of size K')
x = sum_{k=1}^{K'} sqrt(a_k * P) * s_k, sum_k a_k = 1, with power ordered
inversely to channel gain: if |h_1|^2 >= |h_2|^2 >= ... >= |h_K'|^2 then
a_1 <= a_2 <= ... <= a_K' (weakest channel gets most power).

### 6.3 SIC decoding order and K-user SINR (core/sic.py)
Decoding order within a cluster = descending |h_k|^2 (position m=1 = strongest
channel). Power is allocated inversely to channel gain (Sec 6.2: a_1<=a_2<=...),
so a user with a *weaker* channel (larger position index, j>m) has been given a
*higher*-power signal. A receiver at position m is capable of decoding and
cancelling the higher-power signals of weaker users (j>m) because, at its own
(better) channel quality, those higher-power symbols are trivially decodable
first — this is what "SIC" means operationally. It cannot decode/cancel the
lower-power signals of users with better channels than itself (j<m); those
remain full, unremovable interference. The weakest user (m=K') performs no
cancellation at all and treats every other superposed signal as interference.

Per-stage SINR for the user at position m:

SINR_m = (a_m * P * |h_m|^2) /
         ( P*|h_m|^2 * sum_{j<m} a_j              -- stronger users' lower-power signals, NOT cancellable, full interference
           + epsilon * P*|h_m|^2 * sum_{j>m} a_j   -- weaker users' higher-power signals, cancelled by m with residual epsilon
           + N0 )

For m=1 (strongest channel): sum_{j<1}=0, so interference is purely the
residual `epsilon * sum_{j>1} a_j` from (near-)cancelling every weaker user —
this user benefits most from SIC. For m=K' (weakest channel): sum_{j>K'}=0, so
interference is the full, uncancelled `sum_{j<K'} a_j` from every stronger
user — this user gets no SIC benefit, consistent with it never performing
cancellation.

N0 = noise power derived from config `snr_db` as N0 = P / 10^(snr_db/10), with
P fixed at `config.tx_power_w` (default 1.0 W; see Sec 10 for the single
P/N0 convention used everywhere in this plan). `epsilon=0` reproduces perfect
SIC (residual fully removed); `epsilon=1` reproduces no-SIC (every other
signal acts as full interference regardless of position). `sic_epsilon` is
applied once, uniformly, to the aggregate residual power of all
already-cancelled interferers combined (not per-interferer separately) — this
matches the spec's `I_res = epsilon * I` with I read as the *cumulative*
already-cancelled power.

Rate: R_m = log2(1 + SINR_m). Per-cluster (intra-cluster) sum-rate = sum_m R_m;
this is an intermediate quantity — see Sec 6.1 for the `1/n_clusters`
system-level normalization applied across clusters.

`tests/test_sic.py` must assert this direction explicitly: for a 2-user
cluster, the position-1 (strong) user's SINR must *increase* as `epsilon`
decreases toward 0 (more effective cancellation of the weak user's high-power
signal), while the position-2 (weak) user's SINR must be *independent* of
`epsilon` (it never cancels anything).

### 6.4 Power allocation strategies (optimization/power_allocation.py)
Interface: `allocate(channel_gains: np.ndarray, config: SimConfig) -> np.ndarray`
(returns a_k per user within a cluster, summing to 1, ordered per 6.2 constraint).
- `fixed`: static split by rank; for a pair, `a_weak = config.power_alloc_fixed_weak`
  (default 0.7), `a_strong = 1 - a_weak`. For K'>2 clusters, the same fraction
  is applied to the single weakest position and the remainder split equally
  among the rest, preserving the ordering constraint.
- `inverse_gain`: a_k proportional to 1/|h_k|^2, normalized to sum 1.
- `fair_constrained`: maximize sum-rate s.t. each user's rate >= min-rate QoS
  threshold (`outage_rate_threshold_bps_hz`) and the a_k ordering constraint.
  **2-user closed form**: for position-0 (strong) and position-1 (weak) users
  (0-indexed, matching Sec 6.3's SINR formula), the weak user's SINR is
  `SINR_1 = a_1*P*g_1 / (P*g_1*a_0 + N0)` -- note this genuinely depends on
  `a_0` (the stronger user's power), since the weak user suffers *full*
  uncancelled interference from the stronger user's lower-power signal per
  Sec 6.3; an earlier draft of this closed form incorrectly dropped that
  dependency. Substituting `a_0 = 1 - a_1` and solving the QoS floor
  `SINR_1 >= target` where `target = 2^R_th - 1` for `a_1` gives:
  `a_1_min = target*(P*g_1 + N0) / (P*g_1*(1+target))`. Set
  `a_1 = clip(a_1_min, 0.5, 1.0)` to preserve the `a_0<=a_1` ordering
  (giving the weak user exactly its QoS-floor power, maximizing the strong
  user's remaining share and hence sum-rate), `a_0 = 1 - a_1`.
  **K>2 case, solved via CVXPY through Successive Convex Approximation (SCA)**:
  outer loop (max 10 iterations or until a_k changes by <1e-4) over convex
  subproblems. In each outer iteration, treat every OTHER user's power
  fractions from the previous iterate (`a_j^(t-1)`) as fixed constants inside
  user m's own interference term — this makes `SINR_m(a_m)` affine in the
  single free variable `a_m` (all denominators are now constants), so
  `log2(1+SINR_m) >= R_th` becomes a simple linear lower bound on `a_m`.
  CVXPY problem per outer iteration:
  ```python
  a = cp.Variable(K)  # K = cluster_size
  constraints = [a >= 0, cp.sum(a) == 1]
  constraints += [a[m] >= a[m-1] for m in range(1, K)]           # ordering
  for m in range(K):
      denom = N0 + P*gains[m]*(sum(a_prev[j] for j in range(m)) +
                                epsilon*sum(a_prev[j] for j in range(m+1, K)))
      constraints += [P*gains[m]*a[m] >= (2**R_th - 1) * denom]   # QoS floor, affine in a[m]
  objective = cp.Maximize(cp.sum([cp.log(1 + P*gains[m]*a[m]/denom_m) for m in range(K)]))  # concave, denom_m constant this iter
  prob = cp.Problem(objective, constraints); prob.solve(solver=cp.ECOS)
  ```
  Fallback to `inverse_gain` if any outer iteration is infeasible (QoS
  unsatisfiable at current channel realization) or ECOS fails to converge.
- `max_sumrate_qos`: maximize sum-rate only (ignore fairness), still respecting
  the power-ordering constraint (weaker gets more power) and sum-to-1.

## 7. Layer 5 — RIS optimization algorithms (optimization/*.py)
Common interface: `optimize(channels: ChannelSet, config: SimConfig, rng: np.random.Generator) -> np.ndarray`
returning the (already-quantized per Sec.5) theta vector of length N. `rng` is
part of every algorithm's signature for interface consistency; Fixed and
Max-SNR are deterministic and ignore it, Random uses it directly, and
Max-Sum-Rate/Fairness-aware use it for their random-restart seeds (below).

1. **Random**: `theta = rng.uniform(0, 2*pi, size=N)`, then quantize.
2. **Fixed**: `theta = zeros(N)` (baseline, no intelligence).
3. **Max-SNR**: objective = maximize the **sum of effective-channel gains
   across all users** (well-defined default for multi-user case, resolving the
   "single reference user" ambiguity): for each element n,
   `theta_n* = -angle( sum_k conj(h_RU,k,n) * H_BR,n )` — note the conjugate
   on `h_RU,k,n`, required because Sec 4's effective-channel formula uses
   `h_RU,k^H` (conjugate transpose), so the per-element contribution to `h_k`
   is `conj(h_RU,k,n) * H_BR,n * exp(j*theta_n)`; phase-aligning without the
   conjugate would align to the wrong term and not actually maximize
   coherent combining. This phase-aligns element n to the composite
   (summed) user channel rather than one arbitrarily chosen user. Document
   this explicitly as the default; do not offer a second variant.
4. **Max-Sum-Rate**: coordinate ascent — for each element n, evaluate sum-rate
   (using current power allocation, recomputed via the configured
   `power_algo` each full sweep) at a grid of candidate phases and keep the
   best; repeat for up to `max_iters=5` full sweeps or until sum-rate
   improvement < 1e-3. **Multi-start**: single-coordinate greedy ascent can
   get stuck in a local optimum that only a simultaneous multi-element flip
   would escape (measured empirically during implementation: a single run
   from the Max-SNR seed alone landed 23% below the true optimum on a small
   N=5, 1-bit test case, violating this section's own "near-optimal" claim
   below). To make that claim actually hold, run coordinate ascent from
   **4 starting points** -- the Max-SNR solution, plus 3 random-phase
   starts drawn the same way as the Random algorithm (using the same `rng`
   passed to `optimize`) -- and return the single best-scoring result across
   all 4 runs. This raises the cost to `4x` a single run's evaluate() calls,
   which is acceptable given N<=256 and this is an offline/experiment-time
   cost, not a per-dashboard-interaction one for large sweeps (see Sec 12).
   **Candidate grid depends on `ris_bits`**: if `ris_bits` is a finite bit
   depth, the candidates are exactly the `2**ris_bits` quantized phase levels
   from `quantize_phase`'s alphabet (so search and final output already agree
   — no separate quantization pass needed for finite-bit modes, since the
   search is directly over the discrete set this behaves as coordinate descent
   over a discrete alphabet, which for N<=8 is cross-checked against
   exhaustive search in `test_ris_opt.py`); if `ris_bits == "continuous"`, the
   candidates are 8 evenly spaced phases in [0,2pi) refined by one
   `scipy.optimize.minimize_scalar` polish per element per sweep. This
   reconciles the coordinate-ascent search with the final quantization step
   for every bit depth, including `ris_bits=1`. `test_ris_opt.py` includes a
   case at `ris_bits=1` asserting Max-Sum-Rate sum-rate >= Random sum-rate
   (averaged over >=200 seeds).
5. **Fairness-aware**: same multi-start coordinate-ascent scheme as (4)
   (Max-SNR seed + 3 random starts, best-of-4) but objective is
   `alpha * Rsum + beta * JainIndex(rates)` per candidate phase evaluation.
   `alpha`, `beta` from config (default 0.5/0.5).

Exhaustive search remains available as `optimization/exhaustive.py`, used only
in unit tests for N<=8 to validate that coordinate ascent finds the same or a
near-optimal (within 1%) sum-rate as brute force.

## 8. Layer 6 — Network Controller (network/controller.py)
```python
class NetworkController:
    def __init__(self, config: SimConfig, rng: np.random.Generator):
        """Generates the topology once (Sec 4) -- fixed for the controller's
        lifetime, matching "user positions/distances fixed per experiment
        sweep, only fading regenerates per trial" (Sec 4)."""
    def simulate_batch(self) -> BatchResult:
        """Stateless Monte-Carlo batch used by experiments 1-7 and dashboard
        'Live' tab: for n_trials, regenerate channel fading at the fixed
        topology -> run ris_algo.optimize -> evaluate() (pairing -> power
        allocation -> SIC -> rates, Sec 2's objective.py) -> average metrics
        (sum-rate, avg user rate, Jain fairness, outage, EE, BER) over trials."""
    def init_dynamic(self) -> ControllerState:
        """Experiment 8 only: initializes the mobility model and an initial
        RIS configuration at t=0 (always "reconfigures" at t=0, there being
        no prior configuration to compare against)."""
    def step(self) -> ControllerState:
        """Experiment 8 only, called after init_dynamic(): advances user
        positions one config.mobility_dt_s step via the mobility model,
        regenerates channel at the new positions, evaluates the *current*
        theta there first, calls maybe_reconfigure() on that observation, and
        only re-runs RIS optimization if triggered."""
    def maybe_reconfigure(self, current_sinr: np.ndarray, current_time_s: float) -> tuple[bool, str | None]:
        """Returns (should_reconfigure, cause) where cause is "periodic",
        "sinr_drop", or None."""
```
`ReconfigTrigger` (config fields `reconfig_mode`, `reconfig_sinr_drop_db`,
`reconfig_period_s`): supported modes:
- `"periodic"`: reconfigure every `reconfig_period_s` seconds regardless of
  state. The elapsed-time comparison must use a small epsilon tolerance
  (`elapsed >= reconfig_period_s - 1e-9`), not a bare `>=`: repeated
  `time_s += mobility_dt_s` float accumulation (e.g. 0.1+0.1+0.1 != 0.3
  exactly in binary floating point) can otherwise land just under the
  threshold and make the trigger fire one step late intermittently --
  observed empirically during implementation (Milestone 3).
- `"sinr_drop"`: reconfigure when any user's instantaneous SINR has dropped more
  than `reconfig_sinr_drop_db` dB relative to the SINR measured at the last
  reconfiguration.
Default mode = `"sinr_drop"` with `reconfig_sinr_drop_db = 3.0`.

Mobility model (network/mobility.py): random waypoint. **The "cell" is the
same disk used by the default topology (Sec 4): centered at the RIS position,
radius `d_max_m`.** Each user picks a uniform-random destination within this
disk (distance in `[0, d_max_m]` from the RIS, angle in `[0, 2*pi)`), moves in
a straight line toward it at constant speed `user_speed_mps` (config, default
1.0 m/s pedestrian) with step size `user_speed_mps * mobility_dt_s` per
`step()` call (`mobility_dt_s` default 0.1 s). Because every destination is
inside the disk and the user always moves directly toward its current target,
the straight-line path never exits the disk, so no boundary-reflection case
can actually arise -- "reflects at cell boundary" in an earlier draft of this
section was based on a generic RWP description and is not needed for this
specific cell shape; it is dropped. On arrival at (or within one step's
distance of) the target, the position snaps exactly to the target and a new
uniform-random destination is drawn for the *next* step (any leftover
distance in the arriving step is discarded rather than applied toward the new
target, a negligible simplification at `dt_s=0.1s`/`v=1 m/s` step sizes of
~0.1 m against a >=5 m cell). `step()` is called in a loop by
`experiments/exp8_mobility.py` for a configured `mobility_sim_duration_s`,
logging (time, per-user SINR, reconfiguration events with cause) for the
required "SINR-over-time with reconfiguration markers" plot.

## 9. Layer 4 — BER (core/metrics.py)
Default modulation = BPSK: `BER_k = 0.5 * erfc(sqrt(SINR_k))` (scipy.special.erfc).
QPSK optional: `BER_k = 0.5 * erfc(sqrt(SINR_k / 2))`. Selected via
`config.modulation`. Computed analytically from per-user SINR (no bit-level
Monte-Carlo simulation in MVP; noted as a documented simplification, with a
`tests/test_metrics.py` case cross-checking against a bit-level Monte-Carlo
simulation for one configuration to validate the analytic formula).

## 10. Metrics module (core/metrics.py) — full formulas
- Sum-rate: R_sum per Sec 6.1's `1/n_clusters`-normalized system-level formula
  (not a plain unnormalized sum across clusters).
- Outage probability: fraction of trials (or fraction of users across trials,
  documented explicitly as "per-user-trial outage rate") where R_k < config.outage_rate_threshold_bps_hz.
- Energy efficiency: EE = R_sum / (P_total_tx + p_bs_static_w + n_ris * p_element_w),
  where P_total_tx = `config.tx_power_w` directly (P is the single fixed
  quantity in Watts; N0 is always *derived* from P and `snr_db` per the
  formula in Sec 6.3 — this is the one P/N0 convention used everywhere in this
  plan, superseding any other phrasing).
- Jain fairness: J = (sum_k R_k)^2 / (n_users * sum_k R_k^2).
- BER: per Sec. 9, also averaged across users for summary display.

## 11. Experiments (experiments/*.py)
Each script builds a list of `SimConfig` variants (varying exactly one field),
calls `NetworkController(config, rng).simulate_batch()` for each, collects
results into a pandas DataFrame, saves to `results/expN_<name>.csv`, and plots
via a shared `experiments/plotting.py` helper (consistent style/colors).
`run_all.py` runs all 8 sequentially and writes a summary `results/summary.md`.
Explicit mapping to spec:
1. exp1: sweep snr_db, 3 series (no-RIS via `direct_link_blocked=False,n_ris=0`
   effectively h_k=h_BU,k only; random RIS; optimized/max_sumrate RIS).
2. exp2: sweep ris_bits in [1,2,3,4,"continuous"].
3. exp3: sweep n_ris in [16,32,64,128,256].
4. exp4: sweep n_users in [2,3,4,6,8,10], report sum-rate, avg per-user rate,
   Jain fairness, outage, EE (5 sub-plots or one grid figure).
5. exp5: uses `NetworkController.simulate_noma_vs_oma_batch()` (computes both
   under identical per-trial channel/RIS realizations for a fair paired
   comparison, rather than two separate `simulate_batch()` calls). NOMA
   (cluster_size=2 or n_users) vs OMA baseline. OMA: each user gets
   an equal orthogonal time-slot fraction 1/n_users, full power in its own
   slot (no superposition, no SIC), rate
   `R_k^OMA = (1/n_users) * log2(1 + P*|h_k|^2/N0)`,
   `R_sum^OMA = sum_k R_k^OMA`; compared against clustered-NOMA `R_sum` from
   Sec 6.3/10 under identical channel realizations (same RIS config, same
   trial's fading draw) for a fair per-trial comparison.
6. exp6: sweep sic_epsilon in [0, 0.05, 0.10, 0.20], report BER, SINR,
   throughput (=sum-rate), outage.
7. exp7: sweep user_distances_m tuples (e.g., near/mid/far configurations),
   report resulting power allocation a_k and throughput per user.
8. exp8: `NetworkController.step()` loop per Sec.8, plot SINR-over-time with
   reconfiguration markers, compare against a "never reconfigure" baseline run.

## 12. Dashboard (dashboard/app.py)
- Sidebar bound directly to `SimConfig` fields (widgets mirror dataclass names).
- Two tabs: **"Live Simulation"** (recomputes `simulate_batch()` on parameter
  change, `n_trials` capped at a smaller default e.g. 100 for responsiveness,
  cached via `st.cache_data` keyed on the full frozen `SimConfig` — since
  `SimConfig` is a frozen dataclass of primitives it is natively hashable) and
  **"Experiments"** tab (loads precomputed CSVs from `results/` produced by
  `run_all.py` offline — not recomputed live, since full 500-1000-trial sweeps
  across e.g. N=16..256 are too slow for interactive use). The Live tab must
  display its active `n_trials` alongside every metric value (e.g., "Sum-rate:
  4.2 bps/Hz (n_trials=100, live)") so that any numeric difference from the
  Experiments tab's precomputed 500-1000-trial figures is self-explanatory
  rather than appearing as an inconsistency.
- Main panel: topology diagram (matplotlib schematic, BS/RIS/user positions
  from `topology.py`), metrics readout (sum-rate, avg SINR, Jain fairness, EE,
  outage %), per-user rate bar chart.

## 13. Testing strategy
- `test_channel.py`: shape/statistics checks (Rayleigh vs Rician moments), path-loss scaling.
- `test_pairing.py`: strong-weak pairing correctness for various n_users/cluster_size.
- `test_sic.py`: 2-user closed-form SINR check against hand-derived values; epsilon=0 vs epsilon=1 boundary behavior.
- `test_metrics.py`: Jain fairness known cases (equal rates -> J=1), BER analytic vs. bit-level Monte-Carlo cross-check.
- `test_ris_opt.py`: statistical ordering `max(Fixed, Random) <= Max-SNR <= Max-Sum-Rate` (averaged over many seeds, with tolerance) -- **not** a strict `Fixed <= Random` ordering: with the per-element-random-LOS-phase channel model (Sec 4), Fixed (theta=0) and Random are both "non-intelligent" baselines with statistically equal expected performance (confirmed empirically during implementation: paired difference over 200 trials had mean/std ratio near zero, i.e. indistinguishable from no systematic difference), so asserting one beats the other is not a physically grounded claim and the test must not require it; exhaustive-search cross-check for N<=8.
- `test_controller.py`: reconfiguration trigger fires under a forced SINR-drop scenario and not otherwise; periodic mode fires at expected cadence.
- `test_config.py`: SimConfig validation (a_k sums, valid ranges) — validation performed at controller construction time via `__post_init__` checks, raising `ValueError` with a clear message on invalid config (e.g., n_users<2, ris_bits not in allowed set).

## 14. Milestones
1. `core/config.py` + `core/channel.py` + `core/ris.py` + `core/pairing.py` + `core/sic.py` + `core/metrics.py`, with unit tests (Sections 3-6,9,10,13).
2. `optimization/*` (5 RIS algorithms + power allocation strategies), with `test_ris_opt.py`.
3. `network/topology.py`, `network/mobility.py`, `network/controller.py` (both `simulate_batch` and `step`/reconfiguration), with `test_controller.py`.
4. `experiments/exp1..exp8.py` + `run_all.py` + `experiments/plotting.py`.
5. `dashboard/app.py` + components; wire to both live simulation and precomputed experiment CSVs.
6. Polish: README, `configs/default.yaml`, requirements.txt pinning, docstrings referencing the formulas in this plan.

## 15. Project bootstrap (copy-pasteable, so implementation can start immediately)
```
python -m venv .venv
.venv\Scripts\activate          # Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
pytest -q                        # run unit tests
streamlit run ris_noma_sim/dashboard/app.py   # launch dashboard
python -m ris_noma_sim.experiments.run_all    # regenerate all experiment CSVs/plots (add --quick for a fast smoke run)
```
`requirements.txt` (pinned, minimum versions — exact pins finalized at repo
init via `pip freeze` after first successful install):
```
numpy>=1.26
scipy>=1.11
cvxpy>=1.4
matplotlib>=3.8
plotly>=5.18
pandas>=2.1
streamlit>=1.32
pytest>=8.0
```
README.md structure: (1) project description + architecture diagram, (2) setup
commands above, (3) how to run tests, (4) how to launch dashboard, (5) how to
regenerate experiment results, (6) directory layout reference (Section 2 of
this plan), (7) key formulas summary with links to source modules.
