Guanaco evaluates conjunctive queries by interleaving pi-consistency, global uniformity, and bounded joins, following Algorithms 1 to 3 of Abo Khamis and Chen (https://arxiv.org/abs/2610.05440).

Run tests with `PYTHONPATH=src python -m pytest -q` from this directory. subw is computed for at most 4 variables and is exact on the tested hypergraphs, where it is 1 or 1.5. Cleanup and Yannakakis follow the paper's steps but use naive scans and nested-loop enumeration, so there is no claim of the O(N^{subw + delta}) bound.
