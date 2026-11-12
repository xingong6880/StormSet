"""检查严格权重加载与模型输入边界；真实前向证据另存 outputs/。"""

import hashlib
from pathlib import Path
import tempfile
import unittest

import torch
import yaml

from model import CasCast, _load_weights


ROOT = Path(__file__).resolve().parent


class WeightTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / "weights.pth"

    def entry(self, state):
        torch.save({"model": {"component": state}}, self.path)
        return {"path": str(self.path), "sha256": hashlib.sha256(self.path.read_bytes()).hexdigest()}

    def test_matching_weights_load_strictly(self):
        source = torch.nn.Linear(3, 2)
        target = torch.nn.Linear(3, 2)
        _load_weights(target, self.entry(source.state_dict()), "component")
        for key, value in source.state_dict().items():
            torch.testing.assert_close(value, target.state_dict()[key])

    def test_ddp_prefix_is_removed(self):
        source = torch.nn.Linear(3, 2)
        target = torch.nn.Linear(3, 2)
        entry = self.entry({"module."+key: value for key, value in source.state_dict().items()})
        _load_weights(target, entry, "component")
        torch.testing.assert_close(source.weight, target.weight)

    def test_bad_digest_rejects_before_deserialization(self):
        self.path.write_bytes(b"not a checkpoint")
        with self.assertRaisesRegex(ValueError, "校验值"):
            _load_weights(torch.nn.Linear(3, 2), {"path": str(self.path), "sha256": "0"*64}, "component")

    def test_incomplete_extra_or_wrong_shape_parameters_fail(self):
        source = torch.nn.Linear(3, 2)
        for state in [{"weight": source.weight}, {**source.state_dict(), "extra": torch.zeros(1)},
                      {"weight": torch.zeros((2, 4)), "bias": source.bias}]:
            with self.subTest(keys=list(state)), self.assertRaises(RuntimeError):
                _load_weights(torch.nn.Linear(3, 2), self.entry(state), "component")


@unittest.skipUnless((ROOT / "data/reference/cascast/upstream/.git").exists()
                     and all((ROOT / f"checkpoints/cascast/{name}.pth").exists()
                             for name in ["earthformer", "autoencoder", "casformer"]),
                     "需要本地官方源码与三个真实预训练权重")
class RealComponentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        config = yaml.safe_load((ROOT / "config.yaml").read_text())
        cls.model = CasCast(config["model"], device="cpu")

    @classmethod
    def tearDownClass(cls):
        del cls.model

    def test_frozen_components_positions_and_eval_behavior(self):
        self.model.train()
        self.assertTrue(all(not part.training for part in self.model.modules()))
        self.assertTrue(all(not value.requires_grad for value in self.model.predictor.parameters()))
        self.assertTrue(all(not value.requires_grad for value in self.model.autoencoder.parameters()))
        self.assertTrue(any(value.requires_grad for value in self.model.denoiser.parameters()))
        self.assertFalse(self.model.denoiser.model.pos_embed.requires_grad)

    def test_missing_or_incompatible_history_is_not_silently_filled(self):
        history = torch.zeros((1, 13, 1, 384, 384))
        valid = torch.ones_like(history, dtype=torch.bool)
        valid[0, 0, 0, 0, 0] = False
        with self.assertRaisesRegex(ValueError, "缺测"):
            self.model.condition(history, valid, 2)
        valid.fill_(True)
        for bad in [history[:, :12], history.double(), history+float("nan"), history+1.1]:
            with self.assertRaises(ValueError):
                self.model.condition(bad, valid, 2)
        with self.assertRaises(ValueError):
            self.model.condition(history, valid, 0)

    def test_latent_member_and_timestep_axes_are_not_broadcast(self):
        state = torch.zeros((1, 2, 12, 4, 48, 48))
        time = torch.full((1, 2), 500, dtype=torch.int64)
        with self.assertRaisesRegex(ValueError, "同形"):
            self.model.predict_noise(state, time, state[:, :1])
        for bad in [time.float(), time+500, time-501, time[:, :1]]:
            with self.assertRaisesRegex(ValueError, "时间步"):
                self.model.predict_noise(state, bad, state)
        with self.assertRaises(ValueError):
            self.model.decode(state.double())


if __name__ == "__main__":
    unittest.main()
