"""验证共享评分、完整 checkpoint 和 RNG 恢复，不代替雷达模型实验。"""

import copy
from pathlib import Path
import random
import tempfile
import unittest

import numpy as np
import torch
import yaml

from evaluate import score_forecasts
from train import load_checkpoint, save_checkpoint


CONFIG = yaml.safe_load((Path(__file__).resolve().parent / "config.yaml").read_text())


class ForecastScoreTests(unittest.TestCase):
    def test_hand_calculated_shared_event_and_field_credit(self):
        history = torch.zeros((1, 13, 1, 64, 64))
        future = torch.zeros((1, 12, 1, 64, 64))
        future[:, :3, :, :8, :8] = 181/255
        batch = {"history": history, "future": future, "history_valid": torch.ones_like(history, dtype=torch.bool),
                 "future_valid": torch.ones_like(future, dtype=torch.bool)}
        predictions = torch.stack([future, torch.zeros_like(future)], dim=1)
        metrics, credit = score_forecasts(batch, predictions, CONFIG)
        for value in metrics.values():
            torch.testing.assert_close(value, torch.zeros(1))
        expected_field = (181/255)/2 * (3*64)/(12*64*64)
        torch.testing.assert_close(credit, torch.tensor([[1+expected_field, 0.]]))

    def test_unknown_truth_does_not_remove_members_and_bad_prediction_rejects(self):
        history = torch.zeros((1, 13, 1, 64, 128))
        future = torch.zeros((1, 12, 1, 64, 128))
        mask = torch.ones_like(future, dtype=torch.bool)
        mask[:, :, :, :, :64] = False
        batch = {"history": history, "future": future, "history_valid": torch.ones_like(history, dtype=torch.bool),
                 "future_valid": mask}
        predictions = torch.zeros((1, 2, 12, 1, 64, 128))
        predictions[:, :, :, :, :, :64] = 1.
        metrics, credit = score_forecasts(batch, predictions, CONFIG)
        self.assertEqual(metrics["weighted_score"].item(), 0.)
        torch.testing.assert_close(credit, torch.zeros((1, 2)))
        predictions[0, 0, 0, 0, 0, 0] = float("nan")
        with self.assertRaisesRegex(ValueError, "坏成员"):
            score_forecasts(batch, predictions, CONFIG)


class CheckpointTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name)/"step_000001.pth"
        self.model = torch.nn.Module()
        self.model.denoiser = torch.nn.Linear(2, 1)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=.01, weight_decay=0.)
        self.generator = torch.Generator().manual_seed(7)
        self.config = {"run": {"batch_size": 2}, "sampling": {"steps": 20},
                       "training": {"output_dir": "first", "resume": None, "max_updates": 1, "step_batch_size": 4}}
        self.identity = {"code_sha256": {"source": "fixed"}}
        self.update()
        save_checkpoint(self.path, self.model, self.optimizer, self.generator, 1, self.config, self.identity)

    def update(self):
        self.optimizer.zero_grad(set_to_none=True)
        self.model.denoiser(torch.ones((2, 2))).square().mean().backward()
        self.optimizer.step()

    def test_model_optimizer_cursor_and_all_used_rng_restore_exactly(self):
        state = copy.deepcopy(self.model.state_dict())
        optimizer = copy.deepcopy(self.optimizer.state_dict())
        expected = [random.random(), np.random.random(), torch.randn(3), torch.randn(3, generator=self.generator)]
        self.update()
        changed = copy.deepcopy(self.config)
        changed["training"].update(output_dir="second", max_updates=2, resume=str(self.path))
        step = load_checkpoint(self.path, self.model, self.optimizer, self.generator, changed, self.identity)
        self.assertEqual(step, 1)
        for key, value in self.model.state_dict().items():
            torch.testing.assert_close(value, state[key], rtol=0, atol=0)
        restored = self.optimizer.state_dict()
        self.assertEqual(restored["param_groups"], optimizer["param_groups"])
        for index, values in optimizer["state"].items():
            for key, value in values.items():
                torch.testing.assert_close(restored["state"][index][key], value, rtol=0, atol=0)
        self.assertEqual(random.random(), expected[0])
        self.assertEqual(np.random.random(), expected[1])
        torch.testing.assert_close(torch.randn(3), expected[2], rtol=0, atol=0)
        torch.testing.assert_close(torch.randn(3, generator=self.generator), expected[3], rtol=0, atol=0)

    def test_changed_source_or_experiment_rejects_before_model_write(self):
        state = copy.deepcopy(self.model.state_dict())
        with self.assertRaises(ValueError):
            load_checkpoint(self.path, self.model, self.optimizer, self.generator, self.config, {"code_sha256": "changed"})
        changed = copy.deepcopy(self.config)
        changed["training"]["step_batch_size"] = 2
        with self.assertRaises(ValueError):
            load_checkpoint(self.path, self.model, self.optimizer, self.generator, changed, self.identity)
        for key, value in self.model.state_dict().items():
            torch.testing.assert_close(value, state[key], rtol=0, atol=0)

    def test_existing_checkpoint_is_not_overwritten(self):
        before = self.path.read_bytes()
        with self.assertRaises(FileExistsError):
            save_checkpoint(self.path, self.model, self.optimizer, self.generator, 2, self.config, self.identity)
        self.assertEqual(before, self.path.read_bytes())


if __name__ == "__main__":
    unittest.main()
