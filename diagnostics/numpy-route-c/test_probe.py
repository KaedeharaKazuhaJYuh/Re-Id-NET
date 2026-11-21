import unittest
import numpy as np

from probe import MODES, cov_penalty, rank_penalty, retrieval, run, synth, triplet, unit, unit_grad


class ProbeTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(46)
        self.a, self.b = rng.normal(size=(12, 4)), rng.normal(size=(12, 4))
        self.y = np.repeat(range(3), 4)

    def assert_directional_grad(self, f, a, b):
        rng = np.random.default_rng(19)
        da, db = rng.normal(size=a.shape), rng.normal(size=b.shape)
        _, ga, gb = f(a, b)
        h = 1e-6
        ref = (f(a + h*da, b + h*db)[0] - f(a - h*da, b - h*db)[0])/(2*h)
        self.assertAlmostEqual(ref, (ga*da).sum() + (gb*db).sum(), places=6)

    def test_covariance_gradient(self):
        self.assert_directional_grad(cov_penalty, self.a, self.b)

    def test_ratr_proxy_gradient(self):
        self.assert_directional_grad(lambda a, b: rank_penalty(a, b, self.y), self.a, self.b)

    def test_unit_gradient(self):
        def f(a, b):
            z, n = unit(a)
            return (z*b).sum(), unit_grad(z, n, b), z
        self.assert_directional_grad(f, self.a, self.b)

    def test_triplet_gradient(self):
        def f(a, b):
            value, grad = triplet(a, self.y, margin=3.0)
            return value, grad, np.zeros_like(b)
        self.assert_directional_grad(f, self.a, self.b)

    def test_perfect_retrieval_and_missing_positive(self):
        result = retrieval(np.eye(3), np.arange(3), np.arange(3))
        self.assertEqual(result, {"mAP": 1.0, "rank1": 1.0})
        with self.assertRaises(ValueError):
            retrieval(np.eye(1), np.array([1]), np.array([2]))
        # Two positives at ranks 1 and 3 imply AP = (1 + 2/3) / 2.
        hit = retrieval(np.array([[3, 2, 1]]), np.array([2]), np.array([2, 7, 2]))
        self.assertAlmostEqual(hit["mAP"], 5/6)

    def test_identity_disjoint_splits_and_camera_support(self):
        t, v, s = synth(5)
        ids = [set(d[1]) for d in (t, v, s)]
        self.assertEqual(len(set.union(*ids)), sum(map(len, ids)))
        for _, labels, cameras in (t, v, s):
            self.assertEqual(set(labels[cameras == 0]), set(labels[cameras == 1]))

    def test_smoke_is_deterministic(self):
        a = run([7], 3)
        self.assertEqual(a, run([7], 3))
        self.assertEqual(set(a["aggregate"]), set(MODES))
        for mode in MODES:
            test = a["aggregate"][mode]["test"]
            for branch in ("branch_a", "branch_b", "fusion"):
                self.assertGreaterEqual(test[branch]["mAP"]["mean"], 0.0)
                self.assertLessEqual(test[branch]["mAP"]["mean"], 1.0)


if __name__ == "__main__":
    unittest.main()
