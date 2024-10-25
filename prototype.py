import numpy as np
import networkx as nx
from scipy i
import collections

UE = 0
SO = 1

def get_linear_function(a, b):

    def int(x):
        return a*x**2 / 2 + b * x
    
    def main(x):
        return a*x + b

    def derivative(x):
        return a

    return collections.namedtuple("Function (linear)", function = main, derivative = derivative)

null_function = get_linear_function(0, 0)

class NetworkGraph:

    def __init__(self, graph, od_demand, cost_functions):
        self.graph = graph
        self.od_demand = od_demand
        self.cost_functions = cost_functions
        self.order = graph.order

    def solve(self, mode, max_steps = 20):
        if mode == UE:
            t = {edge: self.cost_functions[edge].main(0) for edge in G.edges}
            shortest_path = nx.shortest_path(self.graph, weight = lambda i, j, ...: t[(i, j)])
            x = np.zeros((order, order))
            for (i, j) in G.edges:
                path = shortest_path[i][j]
                for k in range(len(path) - 1):
                    x[path[k]][path[k+1]] += self.od_demand[i, j]
        if mode == SO:
            ...
        for k in range(max_steps):
            t = [self.cost_functions[edge].main(x[*edge]) for edge in G.edges]
            shortest_path = nx.shortest_path(self.graph, weight = lambda i, j, ...: t[i, j])
            y = np.zeros((order, order))
            for (i, j) in G.edges:
                path = shortest_path[i][j]
                for k in range(len(path) - 1):
                    y[path[k]][path[k+1]] += self.od_demand[i, j]

            def f(alpha):
                z = x + alpha * (y - x)
                return np.sum(self.cost_functions[edge].int(z[edge]) for edge in self.graph.edges)

            def f_p(alpha):
                z = x + alpha * (y - x)
                return np.sum((y[edge] - x[edge]) * self.cost_functions[edge].main(z[edge]) for edge in self.graph.edges)

            res = scipy.optimize.minimize_scalar(f, bounds = (0, 1))
            alpha = res.x
            x = x + alpha * (y - x)
        return x

G = nx.graph()
G.add_edges_from([(0, 1), (0, 2), (1, 3), (2, 3)])
od_demand = np.zeros((4, 4))
od_demand[0, 3] = 4000
cost_functions = {
    (0, 1) : get_linear_function(1/100, 0),
    (0, 2) : get_linear_function(0, 45),
    (1, 3) : get_linear_function(0, 45),
    (0, 2) : get_linear_function(1/100, 0)
                 }

network_graph = NetworkGraph(G, od_demand, cost_function)
print(network_graph.solve(UE))
