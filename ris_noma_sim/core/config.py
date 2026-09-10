"""Single source of truth for every simulation parameter.

Both the Streamlit dashboard and every experiment script bind directly to
`SimConfig` -- no parallel parameter definitions should exist anywhere else
in the codebase. See PLAN.md Section 3.
"""

from __future__ import annotations

from dataclasses import dataclass

_ALLOWED_RIS_BITS = (1, 2, 3, 4, "continuous")
_ALLOWED_RIS_ALGOS = ("random", "fixed", "max_snr", "max_sumrate", "fairness_aware")
_ALLOWED_POWER_ALGOS = ("fixed", "inverse_gain", "fair_constrained", "max_sumrate_qos")
_ALLOWED_CHANNEL_TYPES = ("rayleigh", "rician")
_ALLOWED_MODULATIONS = ("bpsk", "qpsk")
_ALLOWED_RECONFIG_MODES = ("sinr_drop", "periodic")


@dataclass(frozen=True)
class SimConfig:
    n_ris: int = 64  # RIS elements: 0 (no-RIS baseline, Exp.1 only) or 16-256
    n_users: int = 4  # 2-10
    snr_db: float = 15.0  # -10 to 30
    ris_bits: int | str = 2  # 1, 2, 3, 4, or "continuous"
    ris_algo: str = "max_sumrate"
    power_algo: str = "fair_constrained"
    channel_type: str = "rician"
    rician_k_factor: float = 5.0  # dB, used only if channel_type == "rician"
    path_loss_exponent: float = 2.0  # tuned so d_max_m stays non-degenerate across the SNR range, see PLAN.md Sec 4
    cluster_size: int = 2  # NOMA pairing group size; == n_users means full-K NOMA
    sic_epsilon: float = 0.0  # 0 = perfect SIC, 0.05/0.10/0.20 = imperfect
    modulation: str = "bpsk"
    outage_rate_threshold_bps_hz: float = 0.5
    fairness_alpha: float = 0.5  # for fairness_aware RIS algo
    fairness_beta: float = 0.5
    p_element_w: float = 0.005  # RIS element static power (5 mW, literature-typical)
    p_bs_static_w: float = 1.0  # BS circuit static power
    user_distances_m: tuple[float, ...] | None = None  # None => drawn from topology defaults
    tx_power_w: float = 1.0  # P, fixed transmit power in Watts (single source of truth, see PLAN.md Sec 10)
    direct_link_blocked: bool = False  # True => h_BU,k forced to 0
    ris_offset_m: float = 10.0  # BS-to-RIS distance
    d_min_m: float = 5.0  # min RIS-to-user distance for random placement
    d_max_m: float = 15.0  # max RIS-to-user distance (kept non-degenerate, see PLAN.md Sec 4)
    power_alloc_fixed_weak: float = 0.7  # a_weak for power_algo="fixed" (2-user case); a_strong = 1 - this
    reconfig_mode: str = "sinr_drop"
    reconfig_sinr_drop_db: float = 3.0
    reconfig_period_s: float = 5.0
    user_speed_mps: float = 1.0  # mobility model speed (Experiment 8 only)
    mobility_dt_s: float = 0.1
    mobility_sim_duration_s: float = 60.0
    n_trials: int = 500
    seed: int = 42

    def __post_init__(self) -> None:
        if self.n_ris != 0 and not (16 <= self.n_ris <= 256):
            raise ValueError("n_ris must be 0 (no-RIS baseline) or in [16, 256]")
        if self.n_ris == 0 and self.direct_link_blocked:
            raise ValueError("n_ris=0 with direct_link_blocked=True is a degenerate config (no channel at all)")
        if not (2 <= self.n_users <= 10):
            raise ValueError("n_users must be in [2, 10]")
        if self.ris_bits not in _ALLOWED_RIS_BITS:
            raise ValueError(f"ris_bits must be one of {_ALLOWED_RIS_BITS}")
        if not (-10.0 <= self.snr_db <= 30.0):
            raise ValueError("snr_db must be in [-10, 30]")
        if self.ris_algo not in _ALLOWED_RIS_ALGOS:
            raise ValueError(f"ris_algo must be one of {_ALLOWED_RIS_ALGOS}")
        if self.power_algo not in _ALLOWED_POWER_ALGOS:
            raise ValueError(f"power_algo must be one of {_ALLOWED_POWER_ALGOS}")
        if self.channel_type not in _ALLOWED_CHANNEL_TYPES:
            raise ValueError(f"channel_type must be one of {_ALLOWED_CHANNEL_TYPES}")
        if self.modulation not in _ALLOWED_MODULATIONS:
            raise ValueError(f"modulation must be one of {_ALLOWED_MODULATIONS}")
        if self.reconfig_mode not in _ALLOWED_RECONFIG_MODES:
            raise ValueError(f"reconfig_mode must be one of {_ALLOWED_RECONFIG_MODES}")
        if not (0.0 <= self.sic_epsilon <= 1.0):
            raise ValueError("sic_epsilon must be in [0, 1]")
        if not (1 <= self.cluster_size <= self.n_users):
            raise ValueError("cluster_size must be in [1, n_users]")
        if self.cluster_size == 1:
            raise ValueError("cluster_size must be >= 2 (NOMA requires at least a pair per cluster)")
        # n_users need not be evenly divisible by cluster_size: pairing.py leaves
        # at most one leftover user as a singleton cluster (see PLAN.md Sec 6.1).
        if not (0.0 < self.power_alloc_fixed_weak < 1.0):
            raise ValueError("power_alloc_fixed_weak must be in (0, 1)")
        if self.tx_power_w <= 0:
            raise ValueError("tx_power_w must be positive")
        if self.p_element_w < 0 or self.p_bs_static_w < 0:
            raise ValueError("power constants must be non-negative")
        if self.d_min_m <= 0 or self.d_max_m <= self.d_min_m:
            raise ValueError("require 0 < d_min_m < d_max_m")
        if self.user_distances_m is not None and len(self.user_distances_m) != self.n_users:
            raise ValueError("user_distances_m, if given, must have exactly n_users entries")
