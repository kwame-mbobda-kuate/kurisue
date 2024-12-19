import time
import networkx as nx
import numpy as np
import solver
import scipy
import graph_tool.all as gt
import itapas

t = time.perf_counter()

N = 10  # Number of solvings
n = 800  # Number of nodes of the graph
p = 0.01  # Probability that two nodes are connected
seed = 1130426  # S;G
np.random.seed(seed)
graph = nx.fast_gnp_random_graph(n, p, seed=seed, directed=True)

t0 = 1  # free flow travel time
a = 0.15  # alpha
b = 4  # beta
c = 1  # capacity
bpr = solver.get_bpr_function(a, b, c, t0)

scale = 1/10  # Scale parameter of the exponential law
od_matrix = scipy.sparse.dok_array((n, n))
functions = {}

def bpr_function(x, params):
    # a, b, c, t0
    return (
        params[3] * (1 + params[0] * x ** (params[1] + 1) / params[2]**params[1]),
        params[3] * (1 + params[0] * (x / params[2]) ** params[1]),
        params[3] * (params[0] * params[1] * x ** (params[1] - 1) / params[2]**params[1]),
    )

demand = {}
nb_edges = 0

for i in range(n):
    for j in range(n):
        if graph.has_edge(i, j):
            nb_edges += 1
            functions[(i, j)] = bpr
            rd = np.random.exponential(scale)
            od_matrix[i, j] = rd
            demand[(i, j)] = rd

t = time.perf_counter()
for _ in range(N):
    solver.solve(graph, od_matrix, functions)
print((time.perf_counter() - t) / N)

graph = gt.Graph(nx.to_scipy_sparse_array(graph))
A, B, C, T0 = graph.new_edge_property("float"), graph.new_edge_property("float"), graph.new_edge_property("float"), graph.new_edge_property("float")
A.a = a * np.ones(nb_edges)
B.a = b * np.ones(nb_edges)
C.a = c * np.ones(nb_edges)
T0.a = t0 * np.ones(nb_edges)
params = [A, B, C, T0]
back = graph.new_edge_property("float")
back.a = np.zeros(nb_edges)
itapas_solver = itapas.ITAPAS(graph, params, back, bpr_function)

t = time.perf_counter()
for _ in range(N):
    itapas_solver.assign(demand)
print((time.perf_counter() - t) / N)
