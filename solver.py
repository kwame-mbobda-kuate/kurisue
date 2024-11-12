import os
import osmnx
import numpy as np
import networkx as nx
from scipy.spatial import cKDTree
import pandas as pd
import scipy
from pyproj import Transformer


def solve(
    graph: nx.MultiDiGraph,
    od_demand: np.array,
    a: np.array,
    b: np.array,
    c: np.array,
    tol: float = 0.1,
    max_steps: int = 100,
    mode: str = "UE",
):
    t0 = np.zeros((graph.size, graph.size))
    x = np.zeros((graph.size, graph.size))
    for u, v, time in graph.edges(data="travel_time"):
        t0[u, v] = time
    shortest_paths = nx.shortest_path(graph, weight="travel_time")
    for u in shortest_paths:
        for v in shortest_paths[v]:
            path = shortest_paths[u][v]
            for k in range(len(path) - 1):
                x[path[k], path[k + 1]] += od_demand[u, v]
    k = 0
    eps = tol + 1
    while k < max_steps or eps < tol:
        t = t0 * (1 + a * (x / c) ** b)
        shortest_paths = nx.shortest_path(graph, weight=lambda u, v, k, d: t[u, v])
        y = np.zeros((graph.size, graph.size))
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

        res = scipy.optimize.minimize(f, x0=1 / 2, jac=f_p, bounds=(0, 1))
        alpha = res.x
        eps = np.max(np.abs(alpha * (y - x)))
        x = x + alpha * (y - x)
        k += 1
    return x
