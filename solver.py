import numpy as np
import networkx as nx
import scipy


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


"""
# Exemple de Wikipédia
N = 10**5
G = nx.Graph()
start = 0
a = 1
b = 2
end = 3
G.add_edges_from([(start, a), (start, b), (a, end), (b, end)])
od_demand = np.zeros((4, 4))
od_demand[start, end] = 4000
c = np.ones((4, 4))
a_mat = np.zeros((4, 4))
b_mat = np.ones((4, 4))
t0 = np.zeros((4, 4))
t0[start, a] = t0[b, end] = 1 / N
t0[a, end] = t0[start, b] = 45
od_demand[start, end] = 4000
a_mat[b, end] = a_mat[start, a] = N / 100

print(solve(G, od_demand, t0, a_mat, b_mat, c))

G.add_edge(a, b)
print(solve(G, od_demand, t0, a_mat, b_mat, c))"""
