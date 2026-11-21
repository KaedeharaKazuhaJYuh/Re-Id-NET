"""Metadata-only fixtures: opened JPEGs and official test folders are not needed."""
from contextlib import redirect_stderr
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import torch
from data import Market1501, market_train_holdout
from train import args_from, resume_config, run


class ValidationHoldoutTest(unittest.TestCase):
    def setUp(self):
        self.tmp = TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        train = self.root / "bounding_box_train"
        train.mkdir()
        for pid in range(1, 23):
            for cam in ((1,) if pid == 22 else (1, 2)):
                for frame in range(2):
                    (train / f"{pid:04d}_c{cam}s1_{frame:06d}_00.jpg").touch()
        (train / "-1_c1s1_000000_00.jpg").touch()

    def raw_ids(self, dataset):
        return {int(Market1501.pattern.match(p.name)[1]) for p, _, _ in dataset.items}

    def test_source_camera_and_identity_discipline(self):
        tr, q, g, spec = market_train_holdout(self.root, 4, 31)
        self.assertFalse((self.root / 'query').exists())
        self.assertFalse((self.root / 'bounding_box_test').exists())
        self.assertEqual(self.raw_ids(q), self.raw_ids(g))
        self.assertEqual(self.raw_ids(q), set(spec["validation_ids"]))
        self.assertFalse(self.raw_ids(tr) & self.raw_ids(q))
        self.assertIn(22, self.raw_ids(tr))
        self.assertNotIn(-1, self.raw_ids(tr))
        self.assertEqual(set(tr.pids), set(range(tr.classes)))
        self.assertEqual(len(q), 4)
        self.assertEqual(len(g), 12)
        for query_path, qpid, qcam in q.items:
            self.assertNotIn(query_path, [path for path, _, _ in g.items])
            self.assertTrue(any(pid == qpid and cam != qcam for _, pid, cam in g.items))
        self.assertEqual(market_train_holdout(self.root, 4, 31)[-1], spec)
        self.assertNotEqual(market_train_holdout(self.root, 4, 32)[-1], spec)
        with self.assertRaisesRegex(ValueError, 'remaining training identities'):
            market_train_holdout(self.root, 7, 31, min_train_ids=16)

    def test_resume_with_changed_seed_refused_before_model_or_image_open(self):
        state_path = self.root / 'resume.pt'
        cli = ['--market-root', str(self.root), '--backend', 'official-tiny',
               '--device', 'cuda', '--imagenet-checkpoint', str(self.root / 'dummy.pth'),
               '--validation-identities', '4', '--validation-split-seed', '31',
               '--resume-from', str(state_path), '--skip-eval']
        args = args_from(cli)
        tr, _, _, spec = market_train_holdout(self.root, 4, 31)
        old_config = resume_config(args, tr, torch.device('cuda'), spec)
        torch.save({'version': 1, 'config': old_config}, state_path)
        args.validation_split_seed = 32
        with (patch('torch.cuda.is_available', return_value=True),
              patch('train.ReID', side_effect=AssertionError('model instantiated')),
              self.assertRaisesRegex(ValueError, 'training configuration differs')):
            run(args)
        args.validation_identities = 0
        with (patch('torch.cuda.is_available', return_value=True),
              patch('train.ReID', side_effect=AssertionError('model instantiated')),
              self.assertRaisesRegex(ValueError, 'training configuration differs')):
            run(args)


    def test_step_limit_eval_cli_rejects_official_or_unsaved_protocol(self):
        base = ['--market-root', str(self.root), '--backend', 'official-tiny',
                '--device', 'cuda', '--imagenet-checkpoint', str(self.root / 'unused.pth')]
        for bad in (['--max-steps', '1', '--resume-state', 'unused.pt'],
                    ['--validation-identities', '4', '--max-steps', '1'],
                    ['--validation-identities', '4', '--resume-state', 'unused.pt'],
                    ['--validation-identities', '4', '--max-steps', '1',
                     '--resume-state', 'unused.pt', '--skip-eval']):
            with self.subTest(bad=bad), redirect_stderr(StringIO()) as err:
                with self.assertRaises(SystemExit):
                    args_from(base + bad + ['--eval-validation-at-limit'])
                self.assertIn('--eval-validation-at-limit requires', err.getvalue())

    def test_existing_checkpoint_at_cap_can_evaluate_without_training(self):
        src = self.root / 'resumed.pt'
        dst = self.root / 'durable.pt'
        args = args_from([
            '--market-root', str(self.root), '--backend', 'official-tiny',
            '--device', 'cuda', '--imagenet-checkpoint', str(self.root / 'unused.pth'),
            '--p', '3', '--validation-identities', '4', '--max-steps', '1',
            '--validation-split-seed', '31', '--resume-from', str(src),
            '--eval-validation-at-limit', '--resume-state', str(dst)])
        # Make this a metadata-only CPU test with no attempt at an actual model fit.
        args.device, args.backend = 'cpu', 'reference'
        tr, _, _, spec = market_train_holdout(self.root, 4, 31)
        tiny = torch.nn.Linear(1, 1)
        opt = torch.optim.AdamW(tiny.parameters(), lr=args.lr, weight_decay=.05)
        torch.save(dict(version=1, config=resume_config(args, tr, torch.device('cpu'), spec),
                        model=tiny.state_dict(), optimizer=opt.state_dict(),
                        rng_cpu=torch.get_rng_state(), rng_cuda=None,
                        progress=dict(steps=1, epoch=0, offset=1, totals={'loss': 1.},
                                      epoch_log=[], window_steps=0, window_loss=0.)), src)

        def evaluate_only_after_save(model, query, gallery, device, market):
            self.assertTrue(dst.is_file())
            self.assertEqual(torch.load(dst, weights_only=True)['progress']['steps'], 1)
            self.assertEqual(self.raw_ids(query.dataset), self.raw_ids(gallery.dataset))
            return {'heldout': True}

        with (patch('train.ReID', return_value=tiny),
              patch.object(Market1501, '__getitem__', side_effect=AssertionError('image opened')),
              patch('train.evaluate', side_effect=evaluate_only_after_save) as evaluate):
            out = run(args)
            self.assertTrue(out['validation_evaluated_at_step_limit'])
            self.assertEqual(out['evaluation'], {'heldout': True})
            self.assertEqual(out['steps'], 1)
            evaluate.assert_called_once()
            args.eval_validation_at_limit = False
            self.assertIsNone(run(args)['evaluation'])
            evaluate.assert_called_once()


if __name__ == '__main__':
    unittest.main()
