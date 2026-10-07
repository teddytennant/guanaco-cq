"""pi-consistency (Proposition 2.2) and degrees (Section 2.3).

Down-close by adding projections, then drop mappings that have no extension
into a superset relation. This is the semijoin reduction the paper reduces to
Horn satisfiability; on the test sizes the fixed-point form is equivalent and
linear in the data for fixed |V|.
"""

from __future__ import annotations

from .config import restrict


def _key(mapping):
    return tuple(sorted(mapping.items()))


def project(relation, scope):
    scope = frozenset(scope)
    return {_key(restrict(dict(item), scope)) for item in relation}


def down_close(config):
    """Add every projection of every realized relation. Does not change answers."""
    pending = list(config.realized)
    seen = set(config.realized)
    while pending:
        scope = pending.pop()
        for variable in scope:
            smaller = scope - {variable}
            projected = project(config.relations[scope], smaller)
            if smaller not in seen:
                config.realized.add(smaller)
                config.relations[smaller] = projected
                seen.add(smaller)
                pending.append(smaller)
            else:
                config.relations[smaller] &= projected


def establish_pi_consistency(config):
    """Proposition 2.2: return a pi-consistent or empty tightening, same answers.

    Mutates and returns the configuration.
    """
    down_close(config)
    changed = True
    while changed:
        changed = False
        for big in list(config.realized):
            for small in list(config.realized):
                if not small < big:
                    continue
                supported = project(config.relations[big], small)
                survivors = config.relations[small] & supported
                if len(survivors) != len(config.relations[small]):
                    config.relations[small] = survivors
                    changed = True
                # Drop big-tuples whose projection is no longer in the small relation.
                kept = set()
                for item in config.relations[big]:
                    if _key(restrict(dict(item), small)) in config.relations[small]:
                        kept.add(item)
                if len(kept) != len(config.relations[big]):
                    config.relations[big] = kept
                    changed = True
    return config


def is_empty(config):
    return frozenset() in config.realized and len(config.relations[frozenset()]) == 0


def is_pi_consistent(config):
    if is_empty(config):
        return False
    if frozenset() not in config.realized or len(config.relations[frozenset()]) != 1:
        return False
    for big in config.realized:
        for small in config.realized:
            if small <= big and project(config.relations[big], small) != config.relations[small]:
                return False
    return True


def degree(config, big, mapping):
    """deg_C(T | f): number of extensions of f in F_T."""
    small = frozenset(mapping)
    count = 0
    for item in config.relations[frozenset(big)]:
        if restrict(dict(item), small) == mapping:
            count += 1
    return count


def max_degree(config, big, small):
    small, big = frozenset(small), frozenset(big)
    if not config.relations[small]:
        return 0
    return max(degree(config, big, dict(item)) for item in config.relations[small])


def avg_degree(config, big, small):
    small, big = frozenset(small), frozenset(big)
    base = len(config.relations[small])
    if base == 0:
        return 0.0
    return len(config.relations[big]) / base
