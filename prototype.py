import numpy as np
import networkx as nx
import scipy
import collections
import warnings

warnings.filterwarnings("ignore")

UE = 0
SO = 1

Function = collections.namedtuple("Function", ["antideriv", "main", "deriv"])

def get_linear_function(a: float, b: float):
    def antideriv(x):
        return a*x**2 / 2 + b * x
    
    def main(x):
        return a*x + b

    def deriv(x):
        return a

    return  Function(antideriv, main, deriv)

null_function = get_linear_function(0, 0)

class NetworkGraph:

    def __init__(self, graph, od_demand, cost_functions):
        self.graph = graph
        self.od_demand = od_demand
        self.cost_functions = cost_functions
        self.order = graph.order()

    def solve(self, mode, max_steps = 20):
        if mode == UE:
            t = {edge: self.cost_functions[edge].main(0) for edge in G.edges}
            shortest_path = dict(nx.shortest_path(self.graph, weight = lambda i, j, data: t.get((i, j), 0)))
            x = np.zeros((self.order, self.order))
            for i in shortest_path:
                for j in shortest_path[i]:
                    path = shortest_path[i][j]
                    for k in range(len(path) - 1):
                        x[path[k]][path[k+1]] += self.od_demand[i, j]
        if mode == SO:
            ...
        for k in range(max_steps):
            t = {edge: self.cost_functions[edge].main(x[*edge]) for edge in G.edges}
            shortest_path = dict(nx.shortest_path(self.graph, weight = lambda i, j, data: t.get((i, j), 0)))
            y = np.zeros((self.order, self.order))
            for i in shortest_path:
                for j in shortest_path[i]:
                    path = shortest_path[i][j]
                    for k in range(len(path) - 1):
                        y[path[k]][path[k+1]] += self.od_demand[i, j]

            def f(alpha):
                z = x + alpha * (y - x)
                return sum(self.cost_functions[edge].antideriv(z[edge]) for edge in self.graph.edges)

            def f_p(alpha):
                z = x + alpha * (y - x)
                return sum((y[edge] - x[edge]) * self.cost_functions[edge].main(z[edge]) for edge in self.graph.edges)

            res = scipy.optimize.minimize_scalar(f, bounds = (0, 1))
            alpha = res.x
            x = x + alpha * (y - x)
        return x

G = nx.Graph()
start = 0
a = 1
b = 2
end = 3
G.add_edges_from([(start, a), (start, b), (a, end), (b, end)])
od_demand = np.zeros((4, 4))
od_demand[start, end] = 4000
cost_functions = {
    (start, a) : get_linear_function(1/100, 0),
    (start, b) : get_linear_function(0, 45),
    (a, end) : get_linear_function(0, 45),
    (b, end) : get_linear_function(1/100, 0)
                 }

network_graph = NetworkGraph(G, od_demand, cost_functions)
print(network_graph.solve(UE))

G.add_edge(a, b)
cost_functions[(a, b)] = null_function
network_graph = NetworkGraph(G, od_demand, cost_functions)
print(network_graph.solve(UE))