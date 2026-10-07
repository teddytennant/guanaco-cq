"""Tests for the Guanaco core. Expected answers come from a local join."""

from guanaco import (
    Configuration,
    cleanup,
    establish_pi_consistency,
    guanaco,
    is_clean,
    is_pi_consistent,
    parameters,
    realize_pair,
)


def brute_force(relations, free):
    """Join relations and project onto free. Independent of the library.

    relations: sequence of (variable schema, sequence of value tuples).
    free: variables to keep. Empty free is the Boolean case: the result is
    {empty assignment} when the join is non-empty, else the empty set.
    """
    parsed = []
    for schema, tuples in relations:
        schema = tuple(schema)
        rows = [dict(zip(schema, tup)) for tup in tuples]
        parsed.append(rows)
    free = tuple(free)
    found = set()

    def rec(index, assignment):
        if index == len(parsed):
            if not free:
                found.add(frozenset())
            elif all(variable in assignment for variable in free):
                found.add(frozenset((variable, assignment[variable]) for variable in free))
            return
        for row in parsed[index]:
            if any(assignment.get(variable, value) != value for variable, value in row.items()):
                continue
            merged = dict(assignment)
            merged.update(row)
            rec(index + 1, merged)

    rec(0, {})
    return found


def brute_config(config, free):
    """Join every relation stored on a configuration object."""
    relations = []
    for schema, tuples in config.relations.items():
        ordered = tuple(sorted(schema, key=str))
        rows = []
        for assignment in tuples:
            mapping = dict(assignment)
            rows.append(tuple(mapping[variable] for variable in ordered))
        relations.append((ordered, rows))
    return brute_force(relations, free)


def assignment(pairs):
    return frozenset(pairs)


def test_pi_consistency_removes_dangling_tuple_and_preserves_answers():
    relations = {
        ("X", "Y"): [(0, 0), (1, 2)],
        ("Y", "Z"): [(0, 1)],
    }
    config = Configuration(["X", "Y", "Z"], relations)
    expected = brute_force(
        [(("X", "Y"), relations[("X", "Y")]), (("Y", "Z"), relations[("Y", "Z")])],
        ["X", "Y", "Z"],
    )
    assert expected == {assignment([("X", 0), ("Y", 0), ("Z", 1)])}

    reduced = establish_pi_consistency(config)
    assert is_pi_consistent(reduced)
    xy = reduced.relations[frozenset({"X", "Y"})]
    assert assignment([("X", 1), ("Y", 2)]) not in xy
    assert assignment([("X", 0), ("Y", 0)]) in xy
    assert brute_config(reduced, ["X", "Y", "Z"]) == expected


def test_cleanup_on_skewed_relation_preserves_answers_and_is_clean():
    rows = [(0, 0), (0, 1), (0, 2), (0, 3), (1, 0), (2, 0), (3, 0)]
    config = Configuration(["X", "Y"], {("X", "Y"): rows})
    base = 7
    eps = 0.25
    leaves = cleanup(base, config, eps)
    assert len(leaves) > 1
    expected = brute_force([(("X", "Y"), rows)], ["X", "Y"])
    union = set()
    for leaf in leaves:
        assert is_clean(leaf, base, eps)
        union |= brute_config(leaf, ["X", "Y"])
    assert union == expected


def test_realize_pair_equals_pairwise_join():
    left_rows = [(0, 1), (0, 2), (1, 2)]
    right_rows = [(1, 3), (2, 4), (2, 5), (9, 9)]
    config = Configuration(
        ["X", "Y", "Z"],
        {("X", "Y"): left_rows, ("Y", "Z"): right_rows},
    )
    realized = realize_pair(config, {"X", "Y"}, {"Y", "Z"})
    got = realized.relations[frozenset({"X", "Y", "Z"})]
    expected = set()
    for x_val, y_val in left_rows:
        for y_other, z_val in right_rows:
            if y_val == y_other:
                expected.add(assignment([("X", x_val), ("Y", y_val), ("Z", z_val)]))
    assert got == expected
    assert realized.relations[frozenset({"X", "Y"})] == config.relations[frozenset({"X", "Y"})]


def _cycle(r_rows, s_rows, t_rows, u_rows):
    relations = {
        ("X", "Y"): r_rows,
        ("Y", "Z"): s_rows,
        ("Z", "W"): t_rows,
        ("W", "X"): u_rows,
    }
    config = Configuration(["X", "Y", "Z", "W"], relations)
    hypergraph = [("X", "Y"), ("Y", "Z"), ("Z", "W"), ("W", "X")]
    return hypergraph, config, relations


def _run_cycle(r_rows, s_rows, t_rows, u_rows, free=("X",)):
    hypergraph, config, relations = _cycle(r_rows, s_rows, t_rows, u_rows)
    expected = brute_force(
        [
            (("X", "Y"), relations[("X", "Y")]),
            (("Y", "Z"), relations[("Y", "Z")]),
            (("Z", "W"), relations[("Z", "W")]),
            (("W", "X"), relations[("W", "X")]),
        ],
        free,
    )
    got = guanaco(hypergraph, free, 0.5, config, 1.5)
    return got, expected, config


def test_four_cycle_satisfying():
    got, expected, _config = _run_cycle(
        [(0, 1)],
        [(1, 2)],
        [(2, 3)],
        [(3, 0)],
        free=(),
    )
    assert expected == {frozenset()}
    assert got == expected


def test_four_cycle_unsatisfying():
    # Locally consistent 4-cycle with no global answer.
    got, expected, _config = _run_cycle(
        [(0, 1), (1, 0)],
        [(1, 0), (0, 1)],
        [(0, 1), (1, 0)],
        [(0, 0), (1, 1)],
        free=(),
    )
    assert expected == set()
    assert got == expected


def _skewed_rows(width=3):
    rows = [(value, 0) for value in range(width)]
    rows += [(0, value) for value in range(1, width)]
    return rows


def test_four_cycle_skewed_splits_and_matches_brute_force():
    rows = _skewed_rows(3)
    hypergraph, config, relations = _cycle(rows, rows, rows, rows)
    expected = brute_force(
        [
            (("X", "Y"), relations[("X", "Y")]),
            (("Y", "Z"), relations[("Y", "Z")]),
            (("Z", "W"), relations[("Z", "W")]),
            (("W", "X"), relations[("W", "X")]),
        ],
        (),
    )
    # Hand-checked cycle: X=1, Y=0, Z=1, W=0.
    assert frozenset() in expected
    base, eps, _w_plus = parameters(config, 0.5, 1.5)
    leaves = cleanup(base, config, eps)
    assert len(leaves) > 1
    for leaf in leaves:
        assert is_clean(leaf, base, eps)
    got = guanaco(hypergraph, (), 0.5, config, 1.5)
    assert got == expected


def test_acyclic_two_relation_query_matches_brute_force():
    relations = {
        ("X", "Y"): [(0, 1), (1, 1), (2, 2), (3, 4)],
        ("Y", "Z"): [(1, 5), (1, 6), (2, 7), (8, 8)],
    }
    config = Configuration(["X", "Y", "Z"], relations)
    hypergraph = [("X", "Y"), ("Y", "Z")]
    expected = brute_force(
        [(("X", "Y"), relations[("X", "Y")]), (("Y", "Z"), relations[("Y", "Z")])],
        (),
    )
    assert expected == {frozenset()}
    got = guanaco(hypergraph, (), 0.5, config, 1.0)
    assert got == expected

    empty_rel = {
        ("X", "Y"): [(0, 1)],
        ("Y", "Z"): [(2, 3)],
    }
    empty_config = Configuration(["X", "Y", "Z"], empty_rel)
    empty_expected = brute_force(
        [(("X", "Y"), empty_rel[("X", "Y")]), (("Y", "Z"), empty_rel[("Y", "Z")])],
        (),
    )
    assert empty_expected == set()
    assert guanaco(hypergraph, (), 0.5, empty_config, 1.0) == empty_expected


def test_free_set_projection_is_duplicate_free():
    relations = {
        ("X", "Y"): [(0, 1), (0, 2), (1, 2), (2, 3)],
        ("Y", "Z"): [(1, 5), (2, 6), (2, 7), (3, 8), (9, 9)],
    }
    config = Configuration(["X", "Y", "Z"], relations)
    hypergraph = [("X", "Y"), ("Y", "Z")]
    free = ["X"]
    expected = brute_force(
        [(("X", "Y"), relations[("X", "Y")]), (("Y", "Z"), relations[("Y", "Z")])],
        free,
    )
    # X=0 via Y=1 and via Y=2, so a bag join could emit X=0 twice.
    assert expected == {
        assignment([("X", 0)]),
        assignment([("X", 1)]),
        assignment([("X", 2)]),
    }
    got = guanaco(hypergraph, free, 0.5, config, 1.0)
    assert got == expected
    assert len(got) == len(set(got))
    for answer in got:
        assert {variable for variable, _value in answer} <= set(free)
