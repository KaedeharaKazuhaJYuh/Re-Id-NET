import unittest

import torch

from data import evaluate_reid, PKBatchSampler
from losses import objective
from model import ReID
from train import args_from, run


class PipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_stage_geometry_and_gradients(self):
        model = ReID(6)
        x = torch.rand(12, 3, 256, 128)
        with torch.no_grad():
            self.assertEqual([tuple(v.shape[-2:]) for v in model.backbone(x)],
                             [(64,32),(32,16),(16,8),(16,8)])
        y = torch.arange(3).repeat_interleave(4)
        out = model(torch.rand(12,3,64,32))
        self.assertEqual(len(out["aux"]), 2)
        self.assertEqual(out["retrieval"]["combined"].shape, (12,64))
        objective(out, y)["loss"].backward()
        for name, p in model.named_parameters():
            if p.requires_grad:
                self.assertIsNotNone(p.grad, name)
                self.assertTrue(torch.isfinite(p.grad).all(), name)

    def test_cross_camera_retrieval_protocol(self):
        q = torch.tensor([[1.,0.],[0.,1.]])
        g = torch.tensor([[1.,0.],[1.,0.],[0.,1.],[.8,.6]])
        result = evaluate_reid(q,[1,2],[1,1],g,[-1,1,2,1],[2,1,2,3])
        self.assertEqual(result, {"mAP":1., "rank1":1., "valid_queries":2})

    def test_pk_balanced_reproducible(self):
        groups = [i for i in range(8) for _ in range(4)]
        s = PKBatchSampler(groups, 4, 4, seed=33)
        batches = list(s)
        self.assertEqual(batches, list(s))
        self.assertEqual(len(batches), 2)
        for batch in batches:
            labels = [groups[i] for i in batch]
            self.assertEqual(sorted(labels.count(y) for y in set(labels)), [4]*4)

    def test_train_step_and_skip_eval(self):
        for mode in ("none","ratr","cov","cov_consistency"):
            report = run(args_from(["--variant",mode,"--max-steps","1","--skip-eval"]))
            self.assertEqual(report["steps"], 1)
            self.assertIsNone(report["evaluation"])


if __name__ == "__main__":
    unittest.main()
