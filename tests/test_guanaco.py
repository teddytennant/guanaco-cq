"""Correctness tests for Guanaco against an independent nested-loop oracle."""

import itertools

from guanaco import Configuration, answers, guanaco, subw
from guanaco.cleanup import cleanup
from guanaco.consistency import is_pi_consistent


def oracle(config, free=None):
    """Enumerate the product of per-variable domains and keep full answers."""
    domains = {v: set() for v in config.variables}
    for scope, rel in config.relations.items():
        for item in rel:
            for v, val in item:
                domains[v].add(val)
    variables = sorted(config.variables)
    found = []
    for combo in itertools.product(*[sorted(domains[v]) for v in variables]):
        mapping = dict(zip(variables, combo))
        if all(
            tuple(sorted((v, mapping[v]) for v in scope)) in rel
            for scope, rel in config.relations.items()
        ):
            found.append(mapping)
    if free is None:
        return found
    seen = set()
    out = []
    for row in found:
        item = tuple(sorted((v, row[v]) for v in free))
        if item not in seen:
            seen.add(item)
            out.append(dict(item))
    return out


def as_set(rows, variables):
    return {tuple(row[v] for v in variables) for row in rows}


def test_single_edge_width_and_answers():
    # One edge: every tree decomposition has that edge as a bag, g(edge) <= 1,
    # so subw = 1.
    assert subw({"x", "y"}, [{"x", "y"}]) == 1.0
    c = Configuration(frozenset({"x", "y"}))
    c.add_relation(("x", "y"), [(1, 2), (1, 3), (4, 5)])
    got = guanaco(c)
    assert as_set(got, ("x", "y")) == as_set(oracle(c), ("x", "y"))


def test_path_of_two_edges_matches_oracle():
    # A tree: bags can be the edges themselves, subw = 1.
    assert subw({"x", "y", "z"}, [{"x", "y"}, {"y", "z"}]) == 1.0
    c = Configuration(frozenset({"x", "y", "z"}))
    c.add_relation(("x", "y"), [(1, 2), (1, 3), (4, 2)])
    c.add_relation(("y", "z"), [(2, 5), (2, 6), (3, 5)])
    got = guanaco(c)
    assert as_set(got, ("x", "y", "z")) == as_set(oracle(c), ("x", "y", "z"))
    assert len(got) == len(as_set(got, ("x", "y", "z")))


def test_triangle_width_and_answers():
    # A triangle has no tree decomposition whose bags all sit inside an edge:
    # any cover of the three edges needs a bag, and connectedness forces some
    # bag to hold two non-adjacent... actually a triangle's edges pairwise
    # share a vertex, so bags = the three edges IS a tree decomposition, and
    # every bag is an edge, so subw = 1. The 4-cycle is the 1.5 case.
    assert subw({"x", "y", "z"}, [{"x", "y"}, {"y", "z"}, {"x", "z"}]) == 1.0
    c = Configuration(frozenset({"x", "y", "z"}))
    c.add_relation(("x", "y"), [(1, 1), (1, 2), (2, 2)])
    c.add_relation(("y", "z"), [(1, 3), (2, 3), (2, 4)])
    c.add_relation(("x", "z"), [(1, 3), (2, 4)])
    got = guanaco(c)
    assert as_set(got, ("x", "y", "z")) == as_set(oracle(c), ("x", "y", "z"))


def test_four_cycle_width_and_projection():
    # C4: edges {x,y}, {y,z}, {z,w}, {w,x}. Every tree decomposition has a bag
    # containing two non-adjacent vertices (otherwise the cycle is not covered
    # while staying connected), and the polymatroid g(e) = 1 on edges and
    # g({x,z}) = g({y,w}) = 1.5 is edge-dominated, so subw = 1.5.
    edges = [{"x", "y"}, {"y", "z"}, {"z", "w"}, {"w", "x"}]
    assert subw({"x", "y", "z", "w"}, edges) == 1.5
    c = Configuration(frozenset({"x", "y", "z", "w"}))
    c.add_relation(("x", "y"), [(1, 1), (1, 2), (2, 2)])
    c.add_relation(("y", "z"), [(1, 3), (2, 3), (2, 4)])
    c.add_relation(("z", "w"), [(3, 5), (3, 6), (4, 6)])
    c.add_relation(("w", "x"), [(5, 1), (6, 1), (6, 2)])
    got = guanaco(c, free=("x", "z"))
    assert as_set(got, ("x", "z")) == as_set(oracle(c, free=("x", "z")), ("x", "z"))


def test_no_answer_returns_empty():
    # R says x is 1 or 2; S says x is 3. No mapping satisfies both.
    c2 = Configuration(frozenset({"x", "y"}))
    c2.add_relation(("x",), [(1,), (2,)])
    c2.add_relation(("x", "y"), [(3, 4)])
    assert guanaco(c2) == []
    assert oracle(c2) == []


def test_cleanup_is_uniform_and_represents():
    c = Configuration(frozenset({"x", "y"}))
    c.add_relation(("x", "y"), [(1, 1), (1, 2), (1, 3), (1, 4), (2, 5)])
    parts = cleanup(5, c.copy(), 0.5)
    assert parts
    assert all(is_pi_consistent(p) for p in parts)
    joined = set()
    for part in parts:
        for row in answers(part):
            joined.add((row["x"], row["y"]))
    assert joined == as_set(oracle(c), ("x", "y"))


def test_answers_are_unique():
    c = Configuration(frozenset({"x", "y", "z"}))
    c.add_relation(("x", "y"), [(1, 1), (1, 1)])
    c.add_relation(("y", "z"), [(1, 2), (1, 3)])
    got = guanaco(c)
    assert len(got) == len(as_set(got, ("x", "y", "z"))) == 2
