import unittest
import networkx as nx
import numpy as np
import solver
import osmnx
import pandas as pd
import data_processing
import time


class TestSolver(unittest.TestCase):

    def generic_test_solver(self, solver):
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

        solution = solver(G, od_demand, t0, a_mat, b_mat, c)
        ground_truth = np.zeros((4, 4))
        ground_truth[0, 1] = ground_truth[0, 2] = 2000
        ground_truth[1, 3] = ground_truth[2, 3] = 2000
        norm = np.max(np.abs(ground_truth - solution))
        self.assertLessEqual(norm, 0.1)

        G.add_edge(a, b)

        solution = solver(G, od_demand, t0, a_mat, b_mat, c)
        ground_truth = np.zeros((4, 4))
        ground_truth[0, 1] = ground_truth[1, 2] = ground_truth[2, 3] = 4000
        norm = np.max(np.abs(ground_truth - solution))
        self.assertLessEqual(norm, 0.1)

    def test_naive_solver(self):
        self.generic_test_solver(solver.solve)


if __name__ == "__main__":
    unittest.main()
