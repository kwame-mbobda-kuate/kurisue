import numpy as np
import networkx as nx
import scipy
import collections

Function = collections.namedtuple("Function", ["antideriv", "main", "deriv"])

null_function = Function(lambda x: 0, lambda x: 0, lambda x: 0)


def sparse_solver(
    graph: nx.MultiDiGraph,
    od_demand: scipy.sparse.csr_array,
    functions,
    tol: float = 0.1,
    max_steps: int = 100,
    mode: str = "UE",
):
    x = scipy.sparse.dok_array((graph.order, graph.order))
    t = scipy.sparse.dok_array((graph.order, graph.order))
    for u, v in functions:
        t[u, v] = functions[(u, v)].main(x[u, v])
    shortest_paths = dict(nx.shortest_path(graph, weight=lambda u, v, _: t[u, v]))
    for u in shortest_paths:
        for v in shortest_paths[u]:
            path = shortest_paths[u][v]
            for k in range(len(path) - 1):
                x[path[k], path[k + 1]] += od_demand[u, v]
    j = 0
    eps = tol + 1
    while j < max_steps and eps > tol:
        t = scipy.sparse.dok_array((graph.order, graph.order))
        for u, v in functions:
            t[u, v] = functions.get[(u, v)].main(x[u, v])
        shortest_paths = dict(nx.shortest_path(graph, weight=lambda u, v, _: t[u, v]))
        y = scipy.sparse.dok_array((graph.order(), graph.order()))
        for u in shortest_paths:
            for v in shortest_paths[v]:
                path = shortest_paths[u][v]
                for k in range(len(path) - 1):
                    y[path[k], path[k + 1]] += od_demand[u, v]

        def f(alpha):
            return sum(
                functions[(u, v)].antideriv((1 - alpha) * x[u, v] - y[u, v])
                for (u, v) in functions
            )

        def f_p(alpha):
            return sum(
                (y[u, v] - x[u, v])
                * functions[(u, v)].main((1 - alpha) * x[u, v] - y[u, v])
                for (u, v) in functions
            )

        res = scipy.optimize.minimize(f, x0=1 / 2, jac=f_p, bounds=[(0, 1)])
        alpha = res.x
        eps = np.max(np.abs(alpha * (y - x)))
        x = x + alpha * (y - x)
        j += 1
    return x


def solve(
    graph: nx.MultiDiGraph,
    od_demand: np.array,
    t0: np.array,
    a: np.array,
    b: np.array,
    c: np.array,
    tol: float = 0.1,
    max_steps: int = 100,
    mode: str = "UE",
):
    x = np.zeros((graph.order(), graph.order()))
    shortest_paths = dict(nx.shortest_path(graph, weight=lambda u, v, _: t0[u, v]))
    for u in shortest_paths:
        for v in shortest_paths[u]:
            path = shortest_paths[u][v]
            for k in range(len(path) - 1):
                x[path[k], path[k + 1]] += od_demand[u, v]
    j = 0
    eps = tol + 1
    while j < max_steps and eps > tol:
        t = t0 * (1 + a * (x / c) ** b)
        # ln((t / t0) - 1) = ln(a) + b * ln(x / c)
        shortest_paths = dict(nx.shortest_path(graph, weight=lambda u, v, _: t[u, v]))
        y = np.zeros((graph.order(), graph.order()))
        for u in shortest_paths:
            for v in shortest_paths[v]:
                path = shortest_paths[u][v]
                for k in range(len(path) - 1):
                    y[path[k], path[k + 1]] += od_demand[u, v]

        def f(alpha):
            z = x + alpha * (y - x)
            return np.sum(t0 * (z + a * z ** (b + 1) / (c**b * (b + 1))))

        def f_p(alpha):
            return np.sum((y - x) * t0 * (1 + a * ((x + alpha * (y - x)) / c) ** b))

        res = scipy.optimize.minimize(f, x0=1 / 2, jac=f_p, bounds=[(0, 1)])
        alpha = res.x
        eps = np.max(np.abs(alpha * (y - x)))
        x = x + alpha * (y - x)
        j += 1
    return x
