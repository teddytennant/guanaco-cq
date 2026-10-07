"""Configurations, as in Section 2.3 of arXiv:2610.05440.

A configuration on variable set V and domain D is a pair (R, (F_S)) where each
F_S is a set of mappings from S to D. An answer is a mapping f: V -> D whose
restriction to every realized set S lies in F_S.
"""

from __future__ import annotations

from dataclasses import dataclass, field


def _key(mapping):
    return tuple(sorted(mapping.items()))


@dataclass
class Configuration:
    variables: frozenset
    realized: set = field(default_factory=set)
    relations: dict = field(default_factory=dict)

    def copy(self):
        return Configuration(
            self.variables,
            set(self.realized),
            {s: set(rel) for s, rel in self.relations.items()},
        )

    def add_relation(self, variables, tuples):
        """Add a relation given as an iterable of tuples, in variable order."""
        scope = tuple(variables)
        key = frozenset(scope)
        if not key <= self.variables:
            raise ValueError(f"relation variables {set(key)} not in {set(self.variables)}")
        rel = set()
        for tup in tuples:
            if len(tup) != len(scope):
                raise ValueError("tuple arity does not match the relation variables")
            rel.add(_key(dict(zip(scope, tup))))
        self.realized.add(key)
        self.relations[key] = rel
        return self

    def relation_size(self, scope):
        return len(self.relations[frozenset(scope)])

    def max_relation_size(self):
        if not self.relations:
            return 0
        return max(len(rel) for rel in self.relations.values())

    def mappings(self, scope):
        """Yield the mappings of F_scope as dicts."""
        for item in self.relations[frozenset(scope)]:
            yield dict(item)

    def contains(self, scope, mapping):
        return _key({v: mapping[v] for v in scope}) in self.relations[frozenset(scope)]


def restrict(mapping, scope):
    return {v: mapping[v] for v in scope}


def answers(config):
    """All answers of a configuration, by nested-loop join. Test-scale only."""
    if not config.realized:
        return []
    found = []

    def rec(scopes, partial):
        if not scopes:
            found.append(dict(partial))
            return
        scope = scopes[0]
        for mapping in config.mappings(scope):
            if all(partial.get(v) == val for v, val in mapping.items() if v in partial):
                partial.update(mapping)
                rec(scopes[1:], partial)
        for v in scope:
            partial.pop(v, None)

    order = sorted(config.realized, key=lambda s: (-len(s), sorted(s)))
    rec(order, {})
    # The join above only constrains realized variables. Unconstrained variables
    # have no domain here; queries in this package always cover every variable.
    covered = set().union(*config.realized) if config.realized else set()
    if covered != set(config.variables):
        raise ValueError("answers() requires every variable to occur in some relation")
    return found


def project_answers(rows, free):
    free = list(free)
    seen = set()
    out = []
    for row in rows:
        item = _key({v: row[v] for v in free})
        if item not in seen:
            seen.add(item)
            out.append(dict(item))
    return out
