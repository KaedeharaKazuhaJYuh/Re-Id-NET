import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from data import PKBatchSampler
from train import args_from, atomic_checkpoint, run


class ResumeTests(unittest.TestCase):
    def nested_equal(self, first, second):
        if torch.is_tensor(first):
            self.assertTrue(torch.equal(first, second))
        elif isinstance(first, dict):
            self.assertEqual(first.keys(), second.keys())
            for key in first:
                self.nested_equal(first[key], second[key])
        elif isinstance(first, (list, tuple)):
            self.assertEqual(len(first), len(second))
            for lhs, rhs in zip(first, second):
                self.nested_equal(lhs, rhs)
        else:
            self.assertEqual(first, second)

    def test_sampler_offset(self):
        sampler = PKBatchSampler([x // 4 for x in range(32)], 2, 2, seed=7)
        sampler.set_epoch(2)
        full = list(sampler)
        sampler.set_epoch(2, 3)
        self.assertEqual(list(sampler), full[3:])
        self.assertEqual(len(sampler), len(full) - 3)

    def test_mid_epoch_and_epoch_boundary_resume_and_report_once(self):
        base = ['--epochs', '2', '--p', '2', '--k', '2', '--skip-eval',
                '--cpu-threads', '1', '--progress-every', '10']
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            def execute(*extra):
                console = io.StringIO()
                with contextlib.redirect_stdout(console):
                    result = run(args_from([*base, *map(str, extra)]))
                reports = [json.loads(line) for line in console.getvalue().splitlines()]
                return result, reports

            full_path = directory / 'full.pt'
            full, full_reports = execute('--resume-state', full_path)
            self.assertEqual([r['step'] for r in full_reports], [10])
            self.assertEqual(full_reports[0]['window_steps'], 10)
            full_state = torch.load(full_path, weights_only=True)
            for split in (3, 8):
                path = directory / f'split-{split}.pt'
                cut, previous_reports = execute('--resume-state', path, '--max-steps', split)
                self.assertEqual(cut['steps'], split)
                self.assertEqual(cut['epochs'][-1].get('partial', False), split == 3)
                done, reports = execute('--resume-from', path)
                self.assertEqual(full['steps'], done['steps'])
                self.assertEqual(full['epochs'], done['epochs'])
                self.assertEqual(full_reports, previous_reports + reports)
                self.nested_equal(full_state, torch.load(path, weights_only=True))

    def test_input_output_collision_and_existing_state_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            src, dst = Path(tmp) / 'from.pt', Path(tmp) / 'out.pt'
            src.write_bytes(b'keep')
            with contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    args_from(['--resume-from', str(src), '--resume-state', str(dst), '--out', str(src)])
                with self.assertRaises(SystemExit):
                    args_from(['--resume-state', str(src)])
            self.assertEqual(src.read_bytes(), b'keep')

    def test_atomic_save_keeps_prior_checkpoint_after_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'resume.pt'
            atomic_checkpoint(path, {'step': 7})
            with patch('train.os.replace', side_effect=OSError('test')):
                with self.assertRaises(OSError):
                    atomic_checkpoint(path, {'step': 8})
            self.assertEqual(torch.load(path, weights_only=True), {'step': 7})
            self.assertEqual(list(Path(tmp).iterdir()), [path])


if __name__ == '__main__':
    unittest.main()
