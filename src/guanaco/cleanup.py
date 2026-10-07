"""Algorithm 1, Cleanup_epsilon (Section 4).

Interleaves pi-consistency with global uniformity. A pair S, T violates
(M, epsilon)-uniformity when maxdeg(T|S) > M^epsilon * avgdeg(T|S). The
relation F_S is split at M^(epsilon/2) * avgdeg into a light and a heavy
part, and the two branches are cleaned recursively.

The paper's Theorem 4.2 claims a linear-time implementation. This one is the
algorithm as written, with a naive degree scan, so it is correct but not linear.
"""

from __future__ import annotations

import math

from .consistency import avg_degree, establish_pi_consistency, is_empty, max_degree


def _violating_pair(config, m, epsilon):
    threshold = m ** epsilon
    for big in config.realized:
        for small in config.realized:
            if not small <= big or small == big:
                continue
            avg = avg_degree(config, big, small)
            if max_degree(config, big, small) > threshold * avg:
                return small, big
    return None


def cleanup(m, config, epsilon):
    """Cleanup_epsilon(M, C). Returns a list of (M, epsilon)-clean configurations."""
    if m < 1 or epsilon <= 0:
        raise ValueError("Cleanup assumes M >= 1 and epsilon > 0")
    establish_pi_consistency(config)
    if is_empty(config):
        return []
    pair = _violating_pair(config, m, epsilon)
    if pair is None:
        return [config]
    small, big = pair
    avg = avg_degree(config, big, small)
    cut = math.sqrt(m ** epsilon) * avg
    light = set()
    heavy = set()
    for item in config.relations[small]:
        mapping = dict(item)
        deg = sum(
            1
            for big_item in config.relations[big]
            if all(dict(big_item).get(v) == val for v, val in mapping.items())
        )
        (light if deg <= cut else heavy).add(item)
    out = []
    for part in (light, heavy):
        branch = config.copy()
        branch.relations[small] = set(part)
        out.extend(cleanup(m, branch, epsilon))
    return out
