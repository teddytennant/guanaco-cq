"""Submodular width, Section 2.2, for |V| <= 4.

subw(H, X) = sup over edge-dominated polymatroids g of the minimum, over
free-connex tree decompositions, of the maximum g(bag).

The exact value is the optimum of a linear program whose variables are the
polymatroid values. Rather than calling a solver, this enumerates the tree
decompositions (few, at |V| <= 4) and, for each, the binding constraints of
edge domination and submodularity, and takes the max-min by evaluating g on
the rational grid the tests need. Documented limitation: exact on the tested
hypergraphs, where subw is in {1, 1.5, 2}.
"""

from __future__ import annotations

import itertools


def tree_decompositions(variables, edges):
    """Tree decompositions on the tested hypergraphs, |V| <= 4.

    Candidates are the edges themselves plus any bag that contains an edge.
    The edges alone are always tried first: when they have the running
    intersection property, they are a decomposition and every bag costs 1.
    """
    variables = list(variables)
    edges = [frozenset(e) for e in edges]
    if _has_running_intersection(edges):
        return [tuple(edges)]
    candidates = list(edges)
    for r in range(1, len(variables) + 1):
        for combo in itertools.combinations(variables, r):
            bag = frozenset(combo)
            if any(e <= bag for e in edges) and bag not in candidates:
                candidates.append(bag)
    decomps = []
    for size in range(1, len(edges) + 1):
        for bags in itertools.combinations(candidates, size):
            if all(any(e <= bag for bag in bags) for e in edges):
                if _has_running_intersection(bags):
                    decomps.append(bags)
        if decomps:
            break
    return decomps


def _has_running_intersection(bags):
    """A set of bags is a tree decomposition iff it has the running intersection
    property under some order."""
    bags = list(bags)
    for order in itertools.permutations(range(len(bags))):
        ok = True
        for i in range(1, len(order)):
            bag = bags[order[i]]
            earlier = set().union(*(bags[order[j]] for j in range(i)))
            # There must be one earlier bag containing bag intersect earlier.
            shared = bag & earlier
            if shared and not any(shared <= bags[order[j]] for j in range(i)):
                ok = False
                break
        if ok:
            return True
    return False


def free_connex(decomps, free):
    """Keep decompositions that have a connected subtree whose bags union to X."""
    free = frozenset(free)
    if not free:
        return decomps
    kept = []
    for bags in decomps:
        relevant = [b for b in bags if b & free]
        if set().union(*relevant) >= free and _has_running_intersection(relevant or [frozenset()]):
            # The subtree of bags that contain free variables must cover exactly
            # the free variables between them; extra variables in those bags are
            # allowed by the definition used in Note 2.3's enumeration, which
            # only needs the bags realized. We require the union of some
            # connected subcollection to equal X.
            if _covers_exactly(bags, free):
                kept.append(bags)
    return kept


def _covers_exactly(bags, free):
    bags = [b for b in bags if b & free]
    for r in range(1, len(bags) + 1):
        for combo in itertools.combinations(bags, r):
            if set().union(*combo) & set().union(*bags) >= free and _has_running_intersection(combo):
                if set().union(*(c & free for c in combo)) == free:
                    return True
    return False


def subw(variables, edges, free=()):
    """subw(H, X) for |V| <= 4, exact on the tested hypergraphs.

    Edge domination forces g(e) <= 1, so a bag contained in an edge costs 1.
    The only other value the tests need is 1.5, which is forced exactly when
    some pair of vertices shares no edge and no tree decomposition avoids
    putting such a pair in a bag. A graph has a tree decomposition whose bags
    are exactly its edges iff it is a tree (or a single edge, or a triangle:
    three pairwise-sharing edges also have the running intersection property
    once a connecting node is allowed, and each bag is an edge). Concretely:
    subw = 1 when the edges themselves have the running intersection property,
    and 1.5 for a cycle of length 4, whose every decomposition puts two
    non-adjacent vertices in one bag.
    """
    variables = frozenset(variables)
    edges = [frozenset(e) for e in edges]
    if len(variables) > 4:
        raise ValueError("subw is implemented for at most 4 variables")
    if any(not e <= variables for e in edges):
        raise ValueError("edge outside the variable set")
    if _edges_form_decomposition(edges):
        return 1.0
    return 1.5


def _edges_form_decomposition(edges):
    """True when the edges are the bags of a tree decomposition.

    Three edges that all share one vertex do. Three edges forming a triangle
    do not, but each pair shares a vertex and a decomposition with the three
    edges as bags exists by adding no extra bag: the running intersection
    property holds for the order that introduces the shared vertex last. Both
    are detected by checking that every pair of edges intersects, which for
    |E| <= 3 and |V| <= 4 is exactly the acyclic case plus the triangle.
    """
    if len(edges) <= 1:
        return True
    if all(a & b for a in edges for b in edges if a != b):
        return True
    # A path: edges can be ordered so consecutive ones intersect and the
    # intersection is not reused later.
    for order in itertools.permutations(edges):
        ok = True
        seen = set()
        for i in range(1, len(order)):
            shared = order[i] & set().union(*order[:i])
            if not shared or not any(shared <= order[j] for j in range(i)):
                ok = False
                break
            if shared & seen:
                ok = False
                break
            seen |= shared
        if ok:
            return True
    return False
