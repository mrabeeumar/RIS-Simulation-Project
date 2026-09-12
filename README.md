# RIS-Assisted NOMA Communication Simulator

[![Python](https://img.shields.io/badge/python-3.11%2B-blue)](https://www.python.org/)
[![Tests](https://img.shields.io/badge/tests-pytest-0A9EDC)](ris_noma_sim/tests)
[![Dashboard](https://img.shields.io/badge/dashboard-streamlit-FF4B4B)](ris_noma_sim/dashboard/app.py)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)

A research-grade Python simulator — with an interactive Streamlit dashboard —
for a **RIS-assisted downlink NOMA multi-user wireless network**. It models
the full signal chain (BS → RIS → users), 5 RIS phase-optimization
algorithms, NOMA power allocation with SIC decoding (perfect and
imperfect), user pairing, mobility-triggered RIS reconfiguration, and an
8-experiment evaluation suite used to benchmark all of the above.

```
                 RIS (16-256 elements)
              /        |         \
         BS  --------------------------  Users (2-10)
             \  direct link (optional) /
```

> **Spec of record:** every formula, algorithm, config field, and module
> boundary is written out in [`PLAN.md`](PLAN.md). It's the authoritative
> source of truth — read it before modifying simulation logic.

---

## Table of Contents

- [Features](#features)
- [Architecture](#architecture)
- [Setup](#setup)
- [Running the Tests](#running-the-tests)
- [Launching the Dashboard](#launching-the-dashboard)
- [Regenerating Experiment Results](#regenerating-experiment-results)
- [Experiment Suite](#experiment-suite)
- [Key Formulas](#key-formulas)
- [Configuration](#configuration)
- [License](#license)

---

## Features

| Capability | Details |
|---|---|
| **Channel modeling** | Rayleigh or Rician (configurable K-factor) small-scale fading, BS→RIS→user cascaded channel + optional direct link |
| **RIS phase control** | 5 algorithms — Fixed, Random, Max-SNR, Max-Sum-Rate, Fairness-Aware — with 1/2/3-bit or continuous phase resolution |
| **NOMA** | Superposition coding with 4 power-allocation strategies (Fixed, Inverse-Gain, QoS-Fair via SCA/CVXPY, Max-Sum-Rate-QoS) |
| **SIC decoding** | Perfect and imperfect (configurable residual `epsilon`), with correct cancellation-direction semantics |
| **User pairing** | Configurable NOMA cluster size (2-user pairs or full cluster) |
| **Mobility** | User movement with mobility-triggered RIS reconfiguration |
| **Metrics** | Sum-rate, per-user rate, SINR, BER (BPSK/QPSK), outage probability, Jain fairness, energy efficiency |
| **Dashboard** | Streamlit app with a live, adjustable simulation tab and a precomputed experiments tab |
| **Reproducibility** | No global RNG seeding — every run takes an explicit `rng: np.random.Generator` seeded from `SimConfig.seed` |

## Architecture

```
ris_noma_sim/
  core/            channel model, RIS quantization, pairing, SIC, metrics, SimConfig
  optimization/    5 RIS phase algorithms, NOMA power allocation, shared evaluate() pipeline
  network/         topology, mobility, NetworkController (the shared simulation pipeline)
  experiments/     8 experiment scripts + run_all.py + plotting
  dashboard/       Streamlit app (sidebar, Live tab, Experiments tab)
  tests/           pytest unit tests for every module above
```

`NetworkController` (`network/controller.py`) is the **single simulation
pipeline** shared by the experiment scripts and the dashboard's Live tab —
they can never disagree about how a metric is computed, because neither one
reimplements the math.

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Requires Python 3.11+. Core dependencies: NumPy, SciPy, CVXPY (ECOS/SCS),
Matplotlib, Plotly, pandas, Streamlit, pytest.

## Running the Tests

```bash
pytest -q
```

Every core formula has a dedicated regression test in `ris_noma_sim/tests/`:
channel model, SIC cancellation direction, power allocation, RIS
optimization (statistical ordering Fixed ≤ Random ≤ Max-SNR ≤
Max-Sum-Rate), metrics, mobility, and reconfiguration triggers.

## Launching the Dashboard

```bash
streamlit run ris_noma_sim/dashboard/app.py
```

The sidebar exposes the full parameter space: RIS elements (16-256), users
(2-10), SNR (-10 to 30 dB), RIS phase resolution (1/2/3-bit or continuous),
RIS/power allocation algorithm, channel conditions (Rayleigh/Rician), user
distance, and SIC imperfection. Configs can be saved/loaded as JSON.

- **Live Simulation** tab recomputes on every parameter change (capped at a
  smaller trial count for responsiveness — the trial count is always shown
  next to each metric), with a topology plot, per-user breakdown, CSV/PNG
  downloads, and run-to-run comparison.
- **Experiments** tab loads precomputed results from `results/` — it never
  recomputes live, since a full 500-1000-trial sweep across N=16..256 is too
  slow for interactive use.

## Regenerating Experiment Results

```bash
python -m ris_noma_sim.experiments.run_all            # full run (500 trials/point, slow -- run offline)
python -m ris_noma_sim.experiments.run_all --quick     # fast smoke run (20 trials/point)
```

Writes CSVs and PNG plots to `results/`, plus a `results/summary.md` index.

## Experiment Suite

Full spec for each experiment is in [`PLAN.md` §11](PLAN.md).

| # | Experiment | What it shows |
|---|---|---|
| 1 | Does RIS help? | No-RIS vs. Random RIS vs. Optimized RIS, sum-rate vs. SNR |
| 2 | RIS phase resolution | Sum-rate vs. 1/2/3/4-bit and continuous phase resolution |
| 3 | Number of RIS elements | Sum-rate vs. RIS element count |
| 4 | Number of users | Sum-rate / avg rate / fairness / outage / EE vs. user count |
| 5 | NOMA vs. OMA | Paired comparison under identical channel realizations |
| 6 | SIC imperfection | Perfect vs. imperfect SIC vs. BER/SINR/throughput/outage |
| 7 | User distance | Power allocation vs. throughput across the coverage annulus |
| 8 | Mobility | User movement triggering RIS reconfiguration |

## Key Formulas

- **Effective channel:** `h_k = h_BU,k + h_RU,k^H @ Theta @ H_BR`, where
  `Theta = diag(exp(j*theta))` — `core/channel.py`
- **SIC SINR** (position `m`, descending channel-gain order): stronger
  users' signals are full, uncancellable interference; weaker users'
  signals are cancelled with residual `epsilon` — `core/sic.py`
- **System sum-rate:** `1/n_clusters`-normalized sum of per-cluster NOMA
  sum-rates, for an apples-to-apples comparison with OMA's `1/n_users`
  split — `core/metrics.py`
- **RIS algorithms** (random, fixed, max-SNR, max-sum-rate,
  fairness-aware) — `optimization/`
- **NOMA power allocation** (fixed, inverse-gain, fair-constrained via
  closed-form/SCA-CVXPY, max-sum-rate-QoS) — `optimization/power_allocation.py`

Full derivations and the exact formula for every quantity are in `PLAN.md`.

## Configuration

`ris_noma_sim/core/config.py`'s `SimConfig` dataclass is the **single
source of truth** for every simulation parameter — both the dashboard and
every experiment script bind to it directly, so there's never a second,
parallel parameter definition to drift out of sync. `configs/default.yaml`
is a human-readable reference copy of its defaults (not a loader input).

## License

[MIT](LICENSE) © Muhammad Rabee Umar
