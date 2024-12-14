import time
import networkx as nx
import numpy as np
import solver
import scipy

t = time.perf_counter()

N = 10  # Number of solvings
n = 100  # Number of nodes of the graph
p = 0.01  # Probability that two nodes are connected
seed = 1130426  # S;G
np.random.seed(seed)
graph = nx.fast_gnp_random_graph(n, p, seed=1, directed=True)

t0 = 1
a = 0.15  # alpha
b = 4  # beta
c = 1  # capacity
bpr = solver.Function(
    lambda x: t0 * (1 + a * x ** (b + 1) / c**b),
    lambda x: t0 * (1 + a * (x / c) ** b),
    lambda x: t0 * (a * b * x ** (b - 1) / c**b),
)

scale = 3  # Scale parameter of the exponential law
od_matrix = scipy.sparse.dok_array((n, n))
functions = {}

for i in range(n):
    for j in range(n):
        if graph.has_edge(i, j):
            functions[(i, j)] = bpr
            od_matrix[i, j] = np.random.exponential(scale)

t = time.perf_counter()
for _ in range(N):
    solver.solve(graph, od_matrix, functions)
print((time.perf_counter() - t) / N)
