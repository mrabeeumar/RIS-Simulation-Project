# RIS-Assisted NOMA Communication Simulator

A research-grade Python simulator for a RIS-assisted downlink NOMA multi-user
wireless network: BS -> RIS -> user channel modeling (Rayleigh/Rician), RIS
phase optimization (5 algorithms), NOMA power allocation, SIC decoding
(perfect and imperfect), user pairing, mobility-triggered RIS
reconfiguration, and a full 8-experiment evaluation suite, with an
interactive Streamlit dashboard.

The full implementation spec -- every formula, algorithm, config field, and
module boundary -- lives in [`PLAN.md`](PLAN.md). Read it before modifying
the simulation logic; it is the authoritative source of truth, kept in sync
with the code as defects were found during implementation.

```
                 RIS (16-256 elements)
              /        |         \
         BS  --------------------------  Users (2-10)
             \  direct link (optional) /
```

## Architecture

```
ris_noma_sim/
  core/            channel model, RIS quantization, pairing, SIC, metrics, SimConfig
  optimization/    5 RIS phase algorithms, NOMA power allocation, shared evaluate() pipeline
  network/         topology, mobility, NetworkController (the shared simulation pipeline)
  experiments/     8 experiment scripts + run_all.py + plotting
  dashboard/       Streamlit app
  tests/           pytest unit tests for every module above
```

`NetworkController` (in `network/controller.py`) is the single simulation
pipeline shared by the experiment scripts and the dashboard's Live tab, so
they can never disagree about how a metric is computed.

## Setup

```
python -m venv .venv
.venv\Scripts\activate          # Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Running the tests

```
pytest -q
```

All core formulas (channel model, SIC direction, power allocation, RIS
optimization, metrics, mobility, reconfiguration triggers) have dedicated
regression tests -- see `ris_noma_sim/tests/`.

## Launching the dashboard

```
streamlit run ris_noma_sim/dashboard/app.py
```

The sidebar exposes every parameter a professor would want to vary: RIS
elements (16-256), users (2-10), SNR (-10 to 30 dB), RIS phase resolution
(1/2/3-bit or continuous), RIS/power allocation algorithm, channel
conditions (Rayleigh/Rician), user distance, and SIC imperfection.

- **Live Simulation** tab recomputes on every parameter change (capped at a
  smaller trial count for responsiveness; the trial count is always shown
  next to each metric).
- **Experiments** tab loads precomputed results from `results/` -- it never
  recomputes live, since a full 500-1000-trial sweep across N=16..256 is too
  slow for interactive use.

## Regenerating experiment results

```
python -m ris_noma_sim.experiments.run_all            # full run (500 trials/point, slow -- run offline)
python -m ris_noma_sim.experiments.run_all --quick     # fast smoke run (20 trials/point)
```

Writes CSVs and PNG plots to `results/`, plus a `results/summary.md` index.
The 8 experiments (see `PLAN.md` Section 11 for the full spec of each):

1. Does RIS help? (No-RIS vs Random RIS vs Optimized RIS, sum-rate vs SNR)
2. RIS phase resolution vs sum-rate (1/2/3/4-bit/continuous)
3. Number of RIS elements vs sum-rate
4. Number of users vs sum-rate / avg rate / fairness / outage / EE
5. NOMA vs OMA (paired comparison under identical channel realizations)
6. Perfect vs imperfect SIC vs BER/SINR/throughput/outage
7. User distance vs power allocation vs throughput
8. User mobility -> RIS reconfiguration

## Key formulas

- Effective channel: `h_k = h_BU,k + h_RU,k^H @ Theta @ H_BR`, `Theta = diag(exp(j*theta))` -- `core/channel.py`
- SIC SINR (position `m`, descending gain order): stronger users' signals are
  full, uncancellable interference; weaker users' signals are cancelled with
  residual `epsilon` -- `core/sic.py`
- System sum-rate: `1/n_clusters`-normalized sum of per-cluster NOMA
  sum-rates -- `core/metrics.py`
- 5 RIS algorithms (random, fixed, max-SNR, max-sum-rate, fairness-aware) --
  `optimization/`
- NOMA power allocation (fixed, inverse-gain, fair-constrained via
  closed-form/SCA-CVXPY, max-sum-rate-QoS) -- `optimization/power_allocation.py`

Full derivations and the exact formula for every quantity are in `PLAN.md`.

## Configuration

`ris_noma_sim/core/config.py`'s `SimConfig` dataclass is the single source of
truth for every simulation parameter -- both the dashboard and every
experiment script bind to it directly. `configs/default.yaml` is a
human-readable reference copy of its defaults (not a loader input).
