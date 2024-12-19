import unittest
import networkx as nx
import solver
import numpy as np
import itapas
import graph_tool.all as gt
import scipy


class TestSolver(unittest.TestCase):

    def test_naive_solver(self):
        graph = nx.DiGraph()
        start = 0
        a = 1
        b = 2
        end = 3
        graph.add_edges_from([(start, a), (start, b), (a, end), (b, end)])
        od_demand = scipy.sparse.dok_array((4, 4))
        od_demand[start, end] = 4000
        cost_functions = {
            (start, a): solver.get_linear_function(1 / 100, 0),
            (start, b): solver.get_linear_function(0, 45),
            (a, end): solver.get_linear_function(0, 45),
            (b, end): solver.get_linear_function(1 / 100, 0),
        }

        solution = solver.solve(graph, od_demand, cost_functions)
        ground_truth = dict()
        ground_truth[(0, 1)] = ground_truth[(0, 2)] = 2000
        ground_truth[(1, 3)] = ground_truth[(2, 3)] = 2000
        norm = max(
            abs(solution.get((u, v), 0) - ground_truth.get((u, v), 0))
            for u in range(4)
            for v in range(4)
        )
        self.assertLessEqual(norm, 0.1)

        graph.add_edge(a, b)
        cost_functions[(a, b)] = solver.null_function

        solution = solver.solve(graph, od_demand, cost_functions)
        ground_truth = dict()
        ground_truth[(0, 1)] = ground_truth[(1, 2)] = ground_truth[(2, 3)] = 4000
        norm = max(
            abs(solution.get((u, v), 0) - ground_truth.get((u, v), 0))
            for u in range(4)
            for v in range(4)
        )
        self.assertLessEqual(norm, 0.1)
    
    def test_itapas(self):
        start = 0
        a = 1
        b = 2
        end = 3
        demand = {(start, end): 4000}
        graph = gt.Graph()
        e1 = graph.add_edge(start, a)
        e2 = graph.add_edge(start, b)
        e3 = graph.add_edge(a, end)
        e4 = graph.add_edge(b, end)
        link_param_a = graph.new_edge_property("float")
        link_param_b = graph.new_edge_property("float")
        link_background = graph.new_edge_property("float")
        link_background.a = np.zeros(4)
        link_param_a[e1], link_param_b[e1] = 1/100, 0
        link_param_a[e2], link_param_b[e2] = 0, 45
        link_param_a[e3], link_param_b[e3] = 0, 45
        link_param_a[e4], link_param_b[e4] = 1/100, 0
        link_param = [link_param_a, link_param_b]

        def linear_function(x, param):
            return (param[0] * x**2 / 2 + param[1] * x, param[0] * x + param[1], param[0])
        
        itapas_solver = itapas.ITAPAS(graph, link_param, link_background, linear_function)
        link_flow, _ = itapas_solver.assign(demand)
        ground_truth = np.ones(4) * 2000
        self.assertLessEqual(np.max(np.abs(link_flow - ground_truth)), 0.1)

        e5 = graph.add_edge(a, b)
        link_param_a[e5], link_param_b[e5], link_background[e5] = 0, 0, 0
        itapas_solver = itapas.ITAPAS(graph, link_param, link_background, linear_function)
        link_flow, _ = itapas_solver.assign(demand)
        ground_truth = 4000 * np.array([1, 0, 0, 1, 1])
        self.assertLessEqual(np.max(np.abs(link_flow - ground_truth)), 0.1)

if __name__ == "__main__":
    unittest.main()
