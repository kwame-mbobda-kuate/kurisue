import unittest
import networkx as nx
import solver
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


if __name__ == "__main__":
    unittest.main()
