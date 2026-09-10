"""NOMA user clustering / pairing. See PLAN.md Section 6.1.

`pair_users` only decides cluster *membership* (which users share a NOMA
superposition slot). Decoding order and power ordering within a cluster are
determined separately, by descending channel gain, in `core/sic.py` and
`optimization/power_allocation.py` -- so the order of indices returned here
within a cluster is not itself meaningful.
"""

from __future__ import annotations

import numpy as np


def pair_users(channel_gains: np.ndarray, cluster_size: int = 2) -> list[list[int]]:
    """Sort users by descending |h_k|^2, then group into clusters of
    `cluster_size` by strong-weak interleaving: each cluster alternately
    takes the strongest and weakest remaining user (standard NOMA pairing).

    If `cluster_size == n_users`, returns a single cluster containing every
    user (full-K NOMA). If `n_users` is not evenly divisible by
    `cluster_size`, the leftover users form one final, smaller cluster (this
    handles e.g. n_users=3, cluster_size=2 as required by Experiment 4). A
    cluster of size 1 (a singleton) is valid: it carries no superposition or
    SIC, per `core/sic.py`'s K'=1 case reducing to interference-free SINR.
    """
    n = len(channel_gains)
    order = list(np.argsort(-np.asarray(channel_gains)))  # descending

    clusters: list[list[int]] = []
    remaining = order
    while remaining:
        size = min(cluster_size, len(remaining))
        cluster: list[int] = []
        front, back = True, remaining[:]
        for _ in range(size):
            cluster.append(back.pop(0) if front else back.pop())
            front = not front
        clusters.append(cluster)
        remaining = back
    return clusters
