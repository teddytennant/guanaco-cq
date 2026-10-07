"""Guanaco conjunctive-query evaluation (arXiv:2610.05440).

Algorithms 1, 2, and 3, plus the supporting notions from Sections 2, 4, and 5.
The caller supplies subw(H, X). The submodular-width linear programs of
Section 2.2 are not implemented. This is a correctness core: it follows the
control flow of the paper, and it does not claim the O(N^{subw+delta}) bound.
"""

from __future__ import annotations

import math
from collections import defaultdict


def _set_key(variables):
    return (len(variables), tuple(sorted((str(type(v).__name__), str(v)) for v in variables)))


def _var_key(variable):
    return (str(type(variable).__name__), str(variable))


def restrict(assignment, variables):
    """Restrict an assignment (frozenset of (variable, value) pairs) to variables."""
    if not variables:
        return frozenset()
    if not isinstance(variables, (set, frozenset)):
        variables = frozenset(variables)
    return frozenset(pair for pair in assignment if pair[0] in variables)


def _powerset(variables):
    items = list(variables)
    count = len(items)
    for mask in range(1 << count):
        yield frozenset(items[i] for i in range(count) if mask & (1 << i))


def _normalize_edge(edge):
    return frozenset(edge)


def log_base_m(size, base):
    """log_base(size), with the paper's convention log_1(1) = 1.

    Never takes a logarithm of 0. If base == 1 and size != 1, the value is
    treated as +infinity so a budget check fails closed.
    """
    if size <= 0:
        raise ValueError("log of a non-positive size is undefined")
    if base == 1:
        if size == 1:
            return 1.0
        return math.inf
    if base <= 0:
        raise ValueError("log base must be positive")
    return math.log(size) / math.log(base)


def log_budget_holds(size, base, budget):
    """Return True iff log_base(size) <= budget (size 0 is treated as -infinity)."""
    if size <= 0:
        return True
    return log_base_m(size, base) <= budget + 1e-9


class Configuration:
    """A configuration C = (R, (F_S)) over a variable set V.

    Relations are stored as a dict from frozenset(variables) to a set of
    assignments. Each assignment is a frozenset of (variable, value) pairs.
    """

    def __init__(self, variables, relations):
        self.variables = frozenset(variables)
        self.relations = {}
        for key, tuples in relations.items():
            if isinstance(key, (list, tuple)):
                schema = list(key)
            else:
                schema = list(key)
            schema_set = frozenset(schema)
            if len(schema) != len(schema_set):
                raise ValueError("relation schema contains a repeated variable")
            self.variables |= schema_set
            bucket = set()
            for tup in tuples:
                bucket.add(_assignment_from(schema, tup))
            self.relations[schema_set] = bucket

    def copy(self):
        other = Configuration.__new__(Configuration)
        other.variables = self.variables
        other.relations = {key: set(value) for key, value in self.relations.items()}
        return other

    def __repr__(self):
        parts = []
        for key in sorted(self.relations, key=_set_key):
            parts.append(f"{sorted(key, key=_var_key)}:{len(self.relations[key])}")
        return "Configuration(" + ", ".join(parts) + ")"


def _assignment_from(schema, tup):
    if isinstance(tup, dict):
        assignment = frozenset(tup.items())
    elif isinstance(tup, frozenset):
        assignment = tup
    else:
        tup = tuple(tup)
        if len(tup) != len(schema):
            raise ValueError("tuple arity does not match relation schema")
        assignment = frozenset(zip(schema, tup))
    got = {variable for variable, _value in assignment}
    if got != set(schema) and not isinstance(tup, frozenset):
        # dict form must mention exactly the schema variables
        if isinstance(tup, dict) and got != set(schema):
            raise ValueError("assignment variables do not match relation schema")
    if isinstance(tup, dict) and got != set(schema):
        raise ValueError("assignment variables do not match relation schema")
    return assignment


def is_empty(config):
    """True when the empty set is realized and F_empty is empty."""
    empty = frozenset()
    relation = config.relations.get(empty)
    return relation is not None and len(relation) == 0


def _down_close(config):
    """Realize every subset of every realized set by projection.

    Existing relations are left unchanged. Missing subsets are added.
    Linear in the input for the tiny instances this core targets.
    """
    original = list(config.relations.items())
    for variables, relation in original:
        for subset in _powerset(variables):
            if subset in config.relations:
                continue
            config.relations[subset] = {restrict(assignment, subset) for assignment in relation}


def establish_pi_consistency(config):
    """Proposition 2.2: down-close, then semijoin-reduce to a fixpoint.

    Returns a tightening that is pi-consistent or empty. Answers are unchanged.
    The input configuration is not mutated.
    """
    config = config.copy()
    _down_close(config)
    changed = True
    while changed:
        changed = False
        keys = list(config.relations.keys())
        for bigger in keys:
            bigger_rel = config.relations[bigger]
            for smaller in keys:
                if not smaller <= bigger or smaller == bigger:
                    continue
                smaller_rel = config.relations[smaller]
                kept = set()
                dropped = False
                for assignment in bigger_rel:
                    if restrict(assignment, smaller) in smaller_rel:
                        kept.add(assignment)
                    else:
                        dropped = True
                if dropped:
                    config.relations[bigger] = kept
                    bigger_rel = kept
                    changed = True
                projection = {restrict(assignment, smaller) for assignment in bigger_rel}
                if projection != smaller_rel:
                    config.relations[smaller] = smaller_rel & projection
                    changed = True
    return config


def is_pi_consistent(config):
    """Down-closed, non-empty, and F_T restricted to S equals F_S whenever Ssubseteq T."""
    empty = frozenset()
    if empty not in config.relations or not config.relations[empty]:
        return False
    for variables in config.relations:
        for subset in _powerset(variables):
            if subset not in config.relations:
                return False
    for bigger, bigger_rel in config.relations.items():
        for smaller, smaller_rel in config.relations.items():
            if smaller <= bigger:
                projection = {restrict(assignment, smaller) for assignment in bigger_rel}
                if projection != smaller_rel:
                    return False
    return True


def degree_map(config, bigger, smaller):
    """Map each f in F_smaller to deg(bigger | f)."""
    if smaller not in config.relations or bigger not in config.relations:
        raise KeyError("degree is only defined for realized sets")
    counts = defaultdict(int)
    for assignment in config.relations[bigger]:
        counts[restrict(assignment, smaller)] += 1
    return {assignment: counts.get(assignment, 0) for assignment in config.relations[smaller]}


def max_degree(config, bigger, smaller):
    degrees = degree_map(config, bigger, smaller)
    if not degrees:
        return 0
    return max(degrees.values())


def avg_degree(config, bigger, smaller):
    smaller_size = len(config.relations[smaller])
    if smaller_size == 0:
        raise ZeroDivisionError("average degree is undefined on an empty relation")
    return len(config.relations[bigger]) / smaller_size


def _uniformity_violations(config, base, eps):
    """Pairs (S, T) with S proper subset of T and maxdeg > base**eps * avgdeg."""
    violations = []
    keys = sorted(config.relations.keys(), key=_set_key)
    limit_cache = {}
    for bigger in keys:
        for smaller in keys:
            if not smaller < bigger:
                continue
            degrees = degree_map(config, bigger, smaller)
            if not degrees:
                continue
            average = len(config.relations[bigger]) / len(degrees)
            # Cache base**eps; base >= 1 and eps > 0, so this is >= 1.
            if base not in limit_cache:
                limit_cache[base] = base ** eps
            if max(degrees.values()) > limit_cache[base] * average:
                violations.append((smaller, bigger, degrees, average))
    return violations


def is_uniform(config, base, eps):
    """(M, eps)-uniform: maxdeg(T|S) <= M**eps * avgdeg(T|S) for all S subseteq T."""
    return not _uniformity_violations(config, base, eps)


def is_clean(config, base, eps):
    """(M, eps)-clean: pi-consistent and (M, eps)-uniform."""
    return is_pi_consistent(config) and is_uniform(config, base, eps)


def cleanup(base, config, eps):
    """Algorithm 1, Cleanup_eps(M, C).

    Returns a list of (M, eps)-clean configurations that represent config.
    The list is empty when every branch is empty. When M == 1 the algorithm
    does not recurse (Lemma 4.1).
    """
    if eps <= 0:
        raise ValueError("eps must be positive")
    if base < 1:
        raise ValueError("M must be >= 1")
    pending = [config]
    leaves = []
    while pending:
        current = establish_pi_consistency(pending.pop())
        if is_empty(current):
            continue
        # Lemma 4.1: no recursive split when M = 1.
        if base == 1 or not _try_split(current, base, eps, pending):
            leaves.append(current)
    return leaves


def _try_split(config, base, eps, pending):
    """Split on one uniformity violation. Return True if a split was scheduled."""
    for smaller, bigger, degrees, average in _uniformity_violations(config, base, eps):
        threshold = (base ** (eps / 2.0)) * average
        low_set = {assignment for assignment, deg in degrees.items() if deg <= threshold}
        high_set = {assignment for assignment, deg in degrees.items() if deg > threshold}
        # A genuine violation partitions F_S. If floating error fails to
        # partition, try another pair rather than recursing forever.
        if not low_set or not high_set:
            continue
        low = config.copy()
        high = config.copy()
        low.relations[smaller] = low_set
        high.relations[smaller] = high_set
        pending.append(high)
        pending.append(low)
        return True
    return False


def realize_pair(config, left, right):
    """Note 5.1: add S union T and the join of tuples that agree on S intersect T.

    The input configuration is not mutated.
    """
    left = frozenset(left)
    right = frozenset(right)
    if left not in config.relations or right not in config.relations:
        raise KeyError("both sets of a realized pair must already be realized")
    intersection = left & right
    index = defaultdict(list)
    for assignment in config.relations[right]:
        index[restrict(assignment, intersection)].append(assignment)
    joined = set()
    for assignment in config.relations[left]:
        for other in index.get(restrict(assignment, intersection), ()):
            joined.add(assignment | other)
    out = config.copy()
    out.relations[left | right] = joined
    return out


def _pair_bound(config, left, right):
    """|F_left| * maxdeg(right | left intersect right)."""
    intersection = left & right
    if intersection not in config.relations:
        return None
    return len(config.relations[left]) * max_degree(config, right, intersection)


def is_realizable_pair(config, left, right, base, w_plus):
    """(M, w_plus)-realizable: both realized, union not realized, log budget holds."""
    left = frozenset(left)
    right = frozenset(right)
    if left not in config.relations or right not in config.relations:
        return False
    if (left | right) in config.relations:
        return False
    bound = _pair_bound(config, left, right)
    if bound is None:
        return False
    return log_budget_holds(bound, base, w_plus)


def _choose_realizable_pair(config, base, w_plus):
    """Pick one realizable pair, preferring a larger union (still Note 5.1)."""
    best = None
    keys = sorted(config.relations.keys(), key=_set_key)
    for left in keys:
        for right in keys:
            if left == right:
                continue
            if not is_realizable_pair(config, left, right, base, w_plus):
                continue
            union = left | right
            rank = (len(union), tuple(sorted(str(v) for v in union)))
            if best is None or rank > best[0]:
                best = (rank, left, right)
    if best is None:
        return None
    return best[1], best[2]


def parameters(config, delta, subw):
    """M, eps, w_plus as in Algorithm 2, with caller-supplied subw(H, X)."""
    if delta <= 0:
        raise ValueError("delta must be positive")
    w_plus = subw * (1.0 + delta)
    variable_count = len(config.variables)
    if variable_count == 0:
        raise ValueError("configuration has an empty variable set")
    eps = min(1.0 - 1.0 / (1.0 + delta), 1.0 / (4.0 * variable_count))
    if config.relations:
        base = max(len(relation) for relation in config.relations.values())
    else:
        base = 0
    return base, eps, w_plus


def _running_intersection(bags, parent):
    count = len(bags)
    children = [[] for _ in range(count)]
    for index, par in enumerate(parent):
        if par >= 0:
            children[par].append(index)
    variables = set()
    for bag in bags:
        variables |= set(bag)
    for variable in variables:
        nodes = [index for index, bag in enumerate(bags) if variable in bag]
        if not _subset_connected(children, parent, nodes):
            return False
    return True


def _subset_connected(children, parent, nodes):
    target = set(nodes)
    if len(target) <= 1:
        return True
    start = nodes[0]
    seen = set()
    stack = [start]
    while stack:
        node = stack.pop()
        if node in seen:
            continue
        seen.add(node)
        neighbors = list(children[node])
        if parent[node] >= 0:
            neighbors.append(parent[node])
        for neighbor in neighbors:
            if neighbor in target and neighbor not in seen:
                stack.append(neighbor)
    return seen == target


def _join_tree(bags):
    """Parent array of a join tree, or None if the bags are not acyclic.

    A maximum spanning tree of the intersection graph is a join tree whenever
    any join tree exists.
    """
    count = len(bags)
    if count == 0:
        return None
    if count == 1:
        return [-1]
    in_tree = [False] * count
    in_tree[0] = True
    parent = [-1] * count
    for _ in range(count - 1):
        best_node = None
        best_parent = None
        best_weight = -1
        for inside in range(count):
            if not in_tree[inside]:
                continue
            for outside in range(count):
                if in_tree[outside]:
                    continue
                weight = len(bags[inside] & bags[outside])
                if best_node is None or weight > best_weight:
                    best_weight = weight
                    best_node = outside
                    best_parent = inside
        if best_node is None:
            return None
        in_tree[best_node] = True
        parent[best_node] = best_parent
    if not _running_intersection(bags, parent):
        return None
    return parent


def _free_connex(bags, parent, free):
    """Paper Section 2.1: some connected subtree has bag-union exactly X.

    X empty is the Boolean case: every tree decomposition is accepted.
    """
    if not free:
        return True
    count = len(bags)
    children = [[] for _ in range(count)]
    for index, par in enumerate(parent):
        if par >= 0:
            children[par].append(index)
    candidates = [index for index, bag in enumerate(bags) if bag <= free]
    total = len(candidates)
    if total == 0:
        return False
    # Bags in the witness subtree are few (|X| is at most |V| <= 6 in the
    # intended search). Enumerate connected subsets.
    if total > 16:
        return _free_connex_grow(bags, children, parent, free, candidates)
    for mask in range(1, 1 << total):
        subset = [candidates[bit] for bit in range(total) if mask & (1 << bit)]
        if not _subset_connected(children, parent, subset):
            continue
        union = frozenset()
        for index in subset:
            union |= bags[index]
        if union == free:
            return True
    return False


def _free_connex_grow(bags, children, parent, free, candidates):
    candidate_set = set(candidates)
    for start in candidates:
        # Grow every connected subset that contains start and stays inside
        # candidate nodes, bounded by 2**|candidates| which we only reach
        # when the cheap enumeration was refused. Stop at the first witness.
        stack = [(start, frozenset([start]), bags[start])]
        seen_states = set()
        while stack:
            _node, subset, union = stack.pop()
            key = subset
            if key in seen_states:
                continue
            seen_states.add(key)
            if union == free:
                return True
            if not union <= free:
                continue
            border = []
            for node in subset:
                neighbors = list(children[node])
                if parent[node] >= 0:
                    neighbors.append(parent[node])
                for neighbor in neighbors:
                    if neighbor in candidate_set and neighbor not in subset:
                        border.append(neighbor)
            for neighbor in border:
                stack.append((neighbor, subset | {neighbor}, union | bags[neighbor]))
    return False


def _extend_free(bags, parent, free, realized):
    """Attach realized subsets of X until the free-connex witness exists."""
    if _free_connex(bags, parent, free):
        return list(bags), list(parent)
    if not free:
        return None
    extras = [bag for bag in realized if bag and bag <= free]
    limit = len(free)

    def rec(current_bags, current_parent, depth):
        if _free_connex(current_bags, current_parent, free):
            return current_bags, current_parent
        if depth == 0:
            return None
        existing = frozenset().union(*current_bags) if current_bags else frozenset()
        used = set(current_bags)
        for bag in extras:
            if bag in used:
                continue
            for anchor in range(len(current_bags)):
                if bag & existing <= current_bags[anchor]:
                    found = rec(current_bags + [bag], current_parent + [anchor], depth - 1)
                    if found is not None:
                        return found
        return None

    return rec(list(bags), list(parent), limit)


def _covering_bag_sets(edges, candidates):
    """Yield distinct sets of realized bags that cover every hyperedge of H."""
    supersets = []
    for edge in edges:
        covers = [bag for bag in candidates if edge <= bag]
        covers.sort(key=lambda bag: (-len(bag), _set_key(bag)))
        if not covers:
            return
        supersets.append(covers)
    seen = set()

    def rec(index, chosen):
        if index == len(supersets):
            if chosen not in seen:
                seen.add(chosen)
                yield chosen
            return
        for bag in supersets[index]:
            yield from rec(index + 1, chosen | frozenset([bag]))

    yield from rec(0, frozenset())


def _fallback_tree(candidates, edges, free, variable_count):
    """Complete search for |V| <= 6: grow a tree of realized bags.

    A new leaf B may be attached to parent P only when
    B intersect (variables already in the tree) is a subset of P, which is
    necessary and sufficient to preserve running intersection.
    """
    if variable_count > 6:
        return None
    ordered = sorted(candidates, key=lambda bag: (-len(bag), _set_key(bag)))
    depth_limit = max(variable_count, 1) + max(len(edges), 1)

    def covers(bags):
        return all(any(edge <= bag for bag in bags) for edge in edges)

    def rec(bags, parent):
        if bags and covers(bags) and _free_connex(bags, parent, free):
            return bags, parent
        if len(bags) >= depth_limit:
            return None
        existing = frozenset().union(*bags) if bags else frozenset()
        used = set(bags)
        for bag in ordered:
            if bag in used:
                continue
            # Prefer bags that cover a still-uncovered edge, but also allow
            # subsets of X (free-connex witnesses) and any other realized bag
            # so the search stays complete within the depth limit.
            if not bags:
                found = rec([bag], [-1])
                if found is not None:
                    return found
                continue
            for anchor in range(len(bags)):
                if bag & existing <= bags[anchor]:
                    found = rec(bags + [bag], parent + [anchor])
                    if found is not None:
                        return found
        return None

    return rec([], [])


def find_free_connex_td(config, hypergraph, free):
    """Search for a free-connex tree decomposition of (H, X) with realized bags.

    Returns (bags, parent) or None. Intended for |V| <= 6.
    """
    edges = [_normalize_edge(edge) for edge in hypergraph]
    free = frozenset(free)
    candidates = [bag for bag in config.relations.keys() if bag]
    # Fast path: one covering bag per hyperedge, then attach subsets of X.
    for chosen in _covering_bag_sets(edges, candidates):
        bags = list(chosen)
        parent = _join_tree(bags)
        if parent is None:
            continue
        extended = _extend_free(bags, parent, free, candidates)
        if extended is not None:
            return extended
    return _fallback_tree(candidates, edges, free, len(config.variables))


def _children_of(parent):
    children = [[] for _ in range(len(parent))]
    root = None
    for index, par in enumerate(parent):
        if par < 0:
            root = index
        else:
            children[par].append(index)
    if root is None:
        raise ValueError("tree decomposition has no root")
    return children, root


def _agrees(left, right):
    right_map = dict(right)
    for variable, value in left:
        other = right_map.get(variable)
        if other is not None and other != value:
            return False
    return True


def yannakakis(config, free, decomposition):
    """Full reducer along the tree of realized bags, then enumerate the join.

    Boolean case (X empty): return {empty assignment} if the join is non-empty,
    otherwise the empty set. Non-empty X: each answer on X once.
    """
    bags, parent = decomposition
    free = frozenset(free)
    relations = []
    for bag in bags:
        if bag not in config.relations:
            raise KeyError("Yannakakis bag is not realized")
        relations.append(set(config.relations[bag]))
    children, root = _children_of(parent)
    edges = [(index, par) for index, par in enumerate(parent) if par >= 0]
    changed = True
    while changed:
        changed = False
        for left, right in edges:
            intersection = bags[left] & bags[right]
            right_proj = {restrict(assignment, intersection) for assignment in relations[right]}
            reduced_left = {
                assignment
                for assignment in relations[left]
                if restrict(assignment, intersection) in right_proj
            }
            if len(reduced_left) != len(relations[left]):
                relations[left] = reduced_left
                changed = True
            left_proj = {restrict(assignment, intersection) for assignment in relations[left]}
            reduced_right = {
                assignment
                for assignment in relations[right]
                if restrict(assignment, intersection) in left_proj
            }
            if len(reduced_right) != len(relations[right]):
                relations[right] = reduced_right
                changed = True

    # Acyclic join after a full reducer: non-empty iff every bag is non-empty.
    # Boolean evaluation only needs that fact (Note 5.2). Non-empty X still
    # enumerates, because the projection is not determined by non-emptiness.
    if not free:
        if bags and all(relations):
            return {frozenset()}
        return set()

    def enumerate_subtree(node, incoming):
        outputs = []
        for assignment in relations[node]:
            if incoming and not _agrees(assignment, incoming):
                continue
            merged = incoming | assignment if incoming else assignment
            partials = [merged]
            failed = False
            for child in children[node]:
                extended = []
                for partial in partials:
                    extended.extend(enumerate_subtree(child, partial))
                partials = extended
                if not partials:
                    failed = True
                    break
            if not failed:
                outputs.extend(partials)
        return outputs

    projected = set()
    for assignment in enumerate_subtree(root, frozenset()):
        projected.add(restrict(assignment, free))
    return projected


def guanaco(hypergraph, free, delta, config, subw):
    """Algorithm 2, Guanaco_{H,X,delta}(C0), with w = subw passed by the caller.

    H is the set of input atoms (query edges), not the set of all realized
    sets. Returns Answers(C0) restricted to X. Boolean X = empty yields a set
    containing the empty assignment when the join is non-empty.
    """
    edges = [_normalize_edge(edge) for edge in hypergraph]
    for edge in edges:
        if edge not in config.relations:
            raise KeyError("every hyperedge of H must be a relation of C0")
    free = frozenset(free)
    if not free <= config.variables:
        raise ValueError("free set X is not a subset of the variable set")
    base, eps, w_plus = parameters(config, delta, subw)
    if base == 0:
        return set()
    return _guanaco_recurse(edges, free, base, eps, w_plus, config)


def _guanaco_recurse(edges, free, base, eps, w_plus, config):
    """Algorithm 3, Guanaco-Recurse."""
    leaves = cleanup(base, config, eps)
    answers = set()
    cached_td = None
    cached = False
    for leaf in leaves:
        pair = _choose_realizable_pair(leaf, base, w_plus)
        if pair is not None:
            left, right = pair
            child = realize_pair(leaf, left, right)
            # The logged bound is an upper bound on the join. Refuse a pair
            # whose realized join exceeds M**w_plus (Note 5.1).
            joined = child.relations[left | right]
            if not log_budget_holds(len(joined), base, w_plus):
                raise RuntimeError("realized pair exceeded the M**w_plus join bound")
            answers |= _guanaco_recurse(edges, free, base, eps, w_plus, child)
            continue
        if not cached:
            cached_td = find_free_connex_td(leaf, edges, free)
            cached = True
        if cached_td is None:
            raise RuntimeError(
                "no free-connex tree decomposition with all bags realized "
                f"({len(leaf.relations)} realized sets); "
                "the paper claims one exists when no realizable pair remains"
            )
        answers |= yannakakis(leaf, free, cached_td)
    return answers
