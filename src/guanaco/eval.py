"""Algorithms 2 and 3: Guanaco and Guanaco-Recurse (Section 5).

Guanaco computes w = subw(H, X), w_plus = w(1 + delta), epsilon, and M, then
calls Guanaco-Recurse. That routine cleans, and for each clean configuration
either realizes one (M, w_plus)-realizable pair (Note 5.1) and recurses, or
finds a free-connex tree decomposition whose bags are all realized and runs
Yannakakis (Note 2.3).

Two departures from the claimed bounds, both stated in the README: subw is the
|V| <= 4 procedure in subw.py, and Yannakakis is implemented as the semijoin
reduction followed by a nested-loop enumeration over the realized bags, which
is correct but not the O(N + OUT) version.
"""

from __future__ import annotations

import math

from .cleanup import cleanup
from .config import answers, project_answers
from .consistency import establish_pi_consistency, is_pi_consistent, max_degree
from .subw import subw


def _log_m(m, value):
    if m == 1:
        return 1.0 if value == 1 else math.inf
    if value <= 0:
        return -math.inf
    return math.log(value) / math.log(m)


def realizable_pair(config, m, w_plus):
    """An (M, w_plus)-realizable pair (S, T): both realized, union not, and
    log_M(|F_S| * maxdeg(T | S intersect T)) <= w_plus."""
    realized = list(config.realized)
    for s in realized:
        for t in realized:
            union = s | t
            if union in config.realized or union == s or union == t:
                continue
            inter = s & t
            if inter not in config.realized:
                continue
            size = len(config.relations[s]) * max_degree(config, t, inter)
            if _log_m(m, size) <= w_plus:
                return s, t
    return None


def realize(config, s, t):
    """Note 5.1: add F_{S union T} = {f union g | f in F_S, g in F_T, agree on intersection}."""
    inter = s & t
    union = s | t
    joined = set()
    t_by_key = {}
    for item in config.relations[t]:
        mapping = dict(item)
        key = tuple(sorted((v, mapping[v]) for v in inter))
        t_by_key.setdefault(key, []).append(mapping)
    for item in config.relations[s]:
        mapping = dict(item)
        key = tuple(sorted((v, mapping[v]) for v in inter))
        for other in t_by_key.get(key, []):
            merged = dict(mapping)
            merged.update(other)
            joined.add(tuple(sorted(merged.items())))
    config.realized.add(union)
    config.relations[union] = joined
    return config


def _bags_realized(config):
    return [scope for scope in config.realized if scope]


def yannakakis(config, free):
    """Note 2.3, simplified: pi-consistent configuration, enumerate answers of the
    realized bags and project onto X. Correct, not the linear version."""
    if not is_pi_consistent(config):
        establish_pi_consistency(config)
    rows = answers(config)
    return project_answers(rows, free)


def guanaco_recurse(config, variables, edges, free, m, epsilon, w_plus):
    cleaned = cleanup(m, config, epsilon)
    collected = []
    for current in cleaned:
        pair = realizable_pair(current, m, w_plus)
        if pair is not None:
            nxt = current.copy()
            realize(nxt, *pair)
            collected.extend(
                guanaco_recurse(nxt, variables, edges, free, m, epsilon, w_plus)
            )
        else:
            collected.extend(yannakakis(current, free))
    return collected


def guanaco(config, free=(), delta=1.0):
    """Algorithm 2. Returns Answers(C) restricted to the free variables.

    free defaults to all variables. delta > 0 is the slack in Theorem 5.4.
    """
    if delta <= 0:
        raise ValueError("delta must be > 0")
    variables = config.variables
    edges = [scope for scope in config.realized if scope]
    free = frozenset(free) if free else variables
    if not free <= variables:
        raise ValueError("free variables must be a subset of the variable set")
    m = config.max_relation_size()
    if m == 0:
        return []
    width = subw(variables, edges, free)
    w_plus = width * (1 + delta)
    n = max(len(variables), 1)
    epsilon = min(1 - 1 / (1 + delta), 1 / (n ** 4))
    rows = guanaco_recurse(config.copy(), variables, edges, free, m, epsilon, w_plus)
    return project_answers(rows, free)


def satisfies(config):
    """Boolean case, Note 5.2: whether the configuration has any answer."""
    return len(guanaco(config, free=())) > 0 or _boolean_yes(config)


def _boolean_yes(config):
    rows = guanaco(config, free=config.variables)
    return len(rows) > 0
