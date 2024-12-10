import numpy as np
import networkx as nx
import scipy
import collections

Function = collections.namedtuple("Function", ["antideriv", "main", "deriv"])

null_function = Function(lambda x: 0, lambda x: 0, lambda x: 0)


def get_linear_function(a: float, b: float):
    return Function(lambda x: a * x**2 / 2 + b * x, lambda x: a * x + b, lambda x: a)


def solve(
    graph: nx.MultiDiGraph,
    od_demand: dict,
    functions,
    tol: float = 0.1,
    max_steps: int = 100,
    mode: str = "UE",
):
    x = dict()
    t = dict()
    for u, v in functions:
        t[(u, v)] = functions[(u, v)].main(x.get((u, v), 0))
    shortest_paths = dict(
        nx.shortest_path(graph, weight=lambda u, v, _: t.get((u, v), 0))
    )
    for u in shortest_paths:
        for v in shortest_paths[u]:
            if (u, v) in od_demand:
                path = shortest_paths[u][v]
                for k in range(len(path) - 1):
                    x[(path[k], path[k + 1])] = (
                        x.get((path[k], path[k + 1]), 0) + od_demand[(u, v)]
                    )
    j = 0
    eps = tol + 1
    while j < max_steps and eps > tol:
        t = dict()
        for u, v in functions:
            t[(u, v)] = functions[(u, v)].main(x.get((u, v), 0))
        shortest_paths = dict(
            nx.shortest_path(graph, weight=lambda u, v, _: t.get((u, v), 0))
        )
        y = dict()
        for u in shortest_paths:
            for v in shortest_paths[v]:
                if (u, v) in od_demand:
                    path = shortest_paths[u][v]
                    for k in range(len(path) - 1):
                        y[(path[k], path[k + 1])] = (
                            y.get((path[k], path[k + 1]), 0) + od_demand[(u, v)]
                        )

        def f(alpha):
            return sum(
                functions.get((u, v), null_function).antideriv(
                    (1 - alpha) * x.get((u, v), 0) + alpha * y.get((u, v), 0)
                )
                for u, v in graph.edges()
            )

        def f_p(alpha):
            return sum(
                (y.get((u, v), 0) - x.get((u, v), 0))
                * functions.get((u, v), null_function).main(
                    (1 - alpha) * x.get((u, v), 0) + alpha * y.get((u, v), 0)
                )
                for u, v in graph.edges()
            )

        res = scipy.optimize.minimize(f, x0=1 / 2, jac=f_p, bounds=[(0, 1)])
        alpha = res.x
        eps = alpha * max(abs(y.get((u, v), 0) - x.get((u, v), 0)) for u, v in graph.edges())
        x = {(u, v): x.get((u, v), 0) + alpha * (y.get((u, v), 0) - x.get((u, v), 0)) for u, v in graph.edges()}
        j += 1
    return x
