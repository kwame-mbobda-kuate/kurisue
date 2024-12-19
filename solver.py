import networkx as nx
import scipy
from typing import Dict, Tuple
import collections

Function = collections.namedtuple("Function", ["antideriv", "main", "deriv"])

null_function = Function(lambda x: 0, lambda x: 0, lambda x: 0)


def get_linear_function(a: float, b: float):
    return Function(lambda x: a * x**2 / 2 + b * x, lambda x: a * x + b, lambda x: a)


def get_bpr_function(a: float, b: float, c: float, t0: float):
    return Function(
        lambda x: t0 * (1 + a * x ** (b + 1) / c**b),
        lambda x: t0 * (1 + a * (x / c) ** b),
        lambda x: t0 * (a * b * x ** (b - 1) / c**b),
    )


def solve(
    graph: nx.DiGraph,
    od_matrix: scipy.sparse.dok_array,
    functions: Dict[Tuple[int, int], Function],
    tol: float = 0.1,
    max_steps: int = 100,
    mode: str = "UE",
) -> dict:
    sources, targets = od_matrix.nonzero()
    new_sources, new_targets = [], []
    edges = graph.edges()
    null_dict = dict()
    x = dict()
    t = dict()

    for u, v in edges:
        null_dict[(u, v)] = 0
        t[(u, v)] = functions[(u, v)].main(0)

    x = null_dict.copy()
    for source, target in zip(sources, targets):
        try:
            shortest_path = nx.shortest_path(
                graph, source=source, target=target, weight=lambda u, v, _: t[(u, v)]
            )
        except nx.exception.NetworkXNoPath:
            continue
        new_sources.append(source)
        new_targets.append(target)
        for k in range(len(shortest_path) - 1):
            x[(shortest_path[k], shortest_path[k + 1])] += od_matrix[source, target]

    sources, targets = new_sources, new_targets

    j = 0
    eps = tol + 1
    while j < max_steps and eps > tol:
        t = null_dict.copy()
        for u, v in functions:
            t[(u, v)] = functions[(u, v)].main(x[u, v])

        y = null_dict.copy()
        for source, target in zip(sources, targets):
            shortest_path = nx.shortest_path(
                graph, source=source, target=target, weight=lambda u, v, _: t[(u, v)]
            )
            for k in range(len(shortest_path) - 1):
                y[(shortest_path[k], shortest_path[k + 1])] += od_matrix[source, target]

        def f(alpha):
            return sum(
                functions[(u, v)].antideriv((1 - alpha) * x[(u, v)] + alpha * y[(u, v)])
                for u, v in edges
            )

        def f_p(alpha):
            return sum(
                (y[u, v] - x[u, v])
                * functions[(u, v)].main((1 - alpha) * x[(u, v)] + alpha * y[(u, v)])
                for u, v in edges
            )

        res = scipy.optimize.minimize(f, x0=1 / 2, jac=f_p, bounds=[(0, 1)])
        alpha = res.x
        eps = alpha * max(abs(y[(u, v)] - x[(u, v)]) for u, v in edges)
        x = {(u, v): x[(u, v)] + alpha * (y[(u, v)] - x[(u, v)]) for u, v in edges}
        j += 1
    return x
