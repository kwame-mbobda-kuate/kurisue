import unittest
import networkx as nx
import solver
import numpy as np
import itapas
import graph_tool.all as gt
import scipy


class TestSolver(unittest.TestCase):

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
        link_param_a[e1], link_param_b[e1] = 1 / 100, 0
        link_param_a[e2], link_param_b[e2] = 0, 45
        link_param_a[e3], link_param_b[e3] = 0, 45
        link_param_a[e4], link_param_b[e4] = 1 / 100, 0
        link_param = [link_param_a, link_param_b]

        link_flow = solver.solve(graph, demand, solver.linear_function, link_param)
        ground_truth = np.ones(4) * 2000
        self.assertLessEqual(np.max(np.abs(link_flow - ground_truth)), 0.1)

        e5 = graph.add_edge(a, b)
        link_param_a[e5], link_param_b[e5] = 0, 0
        link_flow = solver.solve(graph, demand, solver.linear_function, link_param)
        ground_truth = 4000 * np.array([1, 0, 0, 1, 1])
        self.assertLessEqual(np.max(np.abs(link_flow - ground_truth)), 0.1)


if __name__ == "__main__":
    unittest.main()
