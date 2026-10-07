# Guanaco

Faithful correctness core of Guanaco, a conjunctive-query algorithm that alternates pi-consistency with global uniformity (https://arxiv.org/abs/2610.05440).

Run the tests from this directory with `PYTHONPATH=src python -m pytest -q`.

The caller passes subw(H, X). General submodular-width computation is not included. This is a correctness core, not an implementation of the O(N^{subw+delta}) bound.
