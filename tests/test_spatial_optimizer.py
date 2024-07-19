import unittest
from simple3D.spatial_optimizer import SpatialOptimizer

class TestSpatialOptimizer(unittest.TestCase):
    def setUp(self):
        self.N = 10
        self.neighbor_pairs = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7), (7, 8), (8, 9)]
        self.desired_distances = [(0, 2, 5.0), (1, 3, 7.0), (4, 6, 8.0), (5, 7, 6.0), (8, 9, 3.0)]
        self.optimizer = SpatialOptimizer(self.N, self.neighbor_pairs, self.desired_distances)

    def test_strong_restraint(self):
        initial_guess = np.random.rand(self.N * 3) * 10
        cost = self.optimizer.strong_restraint(initial_guess)
        self.assertTrue(isinstance(cost, float))

    def test_run_single_optimization(self):
        initial_guess = np.random.rand(self.N * 3) * 10
        final_coords, score = self.optimizer.run_single_optimization(initial_guess)
        self.assertEqual(len(final_coords), self.N * 3)
        self.assertTrue(isinstance(score, float))

if __name__ == '__main__':
    unittest.main()
