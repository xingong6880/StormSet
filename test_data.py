"""检查数据切片、来源、时间及缺测风险；所有临时缓存仅为可手算的测试输入。"""

import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from data import SEVIRDataset, normalize_vil, restore_vil
from events import extract_events
from scores import fair_brier, masked_mean


ROOT = Path(__file__).resolve().parent
EVENTS = yaml.safe_load((ROOT / "config.yaml").read_text())["events"]


class NormalizationTests(unittest.TestCase):
    def test_all_valid_codes_roundtrip_without_threshold_shift(self):
        raw = torch.arange(255, dtype=torch.uint8)
        normalized, valid = normalize_vil(raw, 255.)
        restored = restore_vil(normalized, valid, 255.)
        self.assertEqual(normalized.dtype, torch.float32)
        self.assertTrue(valid.all())
        torch.testing.assert_close(restored, raw.float(), atol=0, rtol=0)
        for threshold in [133, 181, 219]:
            self.assertTrue(torch.equal(restored >= threshold, raw >= threshold))

    def test_missing_is_separate_from_zero_and_generated_values(self):
        normalized, valid = normalize_vil(torch.tensor([0, 255], dtype=torch.uint8), 255.)
        torch.testing.assert_close(normalized, torch.zeros(2))
        self.assertEqual(valid.tolist(), [True, False])
        restored = restore_vil(normalized, valid, 255.)
        self.assertEqual(restored[0].item(), 0)
        self.assertTrue(torch.isnan(restored[1]))
        # 还原生成值不按 255 推断缺测，也不裁剪到观测编码范围。
        values = torch.tensor([-.1, 1., 1.1], requires_grad=True)
        restored = restore_vil(values, torch.ones(3, dtype=torch.bool), 255.)
        self.assertTrue(torch.isfinite(restored).all())
        torch.testing.assert_close(restored, torch.tensor([-25.5, 255., 280.5]))
        restored.sum().backward()
        torch.testing.assert_close(values.grad, torch.full((3,), 255.))

    def test_invalid_scale_type_and_mask_fail(self):
        for scale in [0., -1., float("nan"), float("inf")]:
            with self.assertRaises(ValueError):
                normalize_vil(torch.zeros(1, dtype=torch.uint8), scale)
            with self.assertRaises(ValueError):
                restore_vil(torch.zeros(1), torch.ones(1, dtype=torch.bool), scale)
        with self.assertRaises(ValueError):
            normalize_vil(torch.zeros(1), 255.)
        with self.assertRaises(ValueError):
            restore_vil(torch.zeros(1), torch.ones(1), 255.)
        with self.assertRaises(ValueError):
            restore_vil(torch.zeros(2), torch.ones(1, dtype=torch.bool), 255.)


class DataTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        self.root = self.base / "cache"
        (self.root / "samples").mkdir(parents=True)
        self.settings = {"root": str(self.root), "split": "train", "height": 64, "width": 64,
                         "normalization_scale": 255., "split_files": {}}
        for name, index in [("train", 0), ("val", 1), ("test", 2)]:
            p = self.base / (name + ".txt")
            p.write_text(f"vil-2017-test.h5-{index}-0.npy\n")
            self.settings["split_files"][name] = str(p)
        self.record = {"event_id": "R0001", "file_name": "vil/2017/test.h5", "file_index": 0,
                       "cascast_window": "vil-2017-test.h5-0-0.npy", "time_utc": "2017-01-01 02:00:00",
                       "minute_offsets": ":".join(str(i) for i in range(-120, 121, 5))}
        self.selection = {"sample_size": 1, "window_part": 0, "history_slice": "0:13", "future_slice": "13:25",
                          "train_list_sha256": hashlib.sha256(Path(self.settings["split_files"]["train"]).read_bytes()).hexdigest(),
                          "records": [self.record]}
        self.frames = np.zeros((25, 64, 64), dtype=np.uint8)
        self.frames[0], self.frames[12], self.frames[13:16], self.frames[24] = 16, 74, 181, 219
        self.write_cache()

    def write_cache(self):
        self.sample = self.root / "samples/R0001.npz"
        np.savez_compressed(self.sample, vil=self.frames)
        p = self.root / "selection.json"
        p.write_text(json.dumps(self.selection))
        digest = hashlib.sha256(p.read_bytes()).hexdigest()
        self.acquisition = {"status": "complete", "selection_sha256": digest,
                            "records": [{"event_id": "R0001", "selection_sha256": digest, "file_index": 0,
                                         "source_url": "https://sevir.s3.us-west-2.amazonaws.com/data/vil/2017/test.h5",
                                         "shape": [25, self.settings["height"], self.settings["width"]], "dtype": "uint8",
                                         "sha256": hashlib.sha256(self.sample.read_bytes()).hexdigest()}]}
        (self.root / "acquisition.json").write_text(json.dumps(self.acquisition))

    def test_slices_identity_timing_and_collation(self):
        dataset = SEVIRDataset(self.settings, EVENTS)
        self.assertEqual(len(dataset), 1)
        batch = next(iter(DataLoader(dataset, batch_size=1, shuffle=False)))
        self.assertEqual(batch["history"].shape, (1, 13, 1, 64, 64))
        self.assertEqual(batch["future"].shape, (1, 12, 1, 64, 64))
        raw_h = restore_vil(batch["history"], batch["history_valid"], dataset.scale)
        raw_f = restore_vil(batch["future"], batch["future_valid"], dataset.scale)
        self.assertEqual(raw_h[0, 0, 0, 0, 0].item(), 16)
        self.assertEqual(raw_h[0, -1, 0, 0, 0].item(), 74)
        self.assertEqual(raw_f[0, 0, 0, 0, 0].item(), 181)
        self.assertEqual(raw_f[0, -1, 0, 0, 0].item(), 219)
        self.assertEqual(batch["event_id"], ["R0001"])
        self.assertEqual(batch["window_name"], [self.record["cascast_window"]])
        self.assertEqual(batch["forecast_origin_utc"], ["2017-01-01T01:00:00+00:00"])
        torch.testing.assert_close(batch["future_lead_minutes"][0], torch.arange(5, 61, 5).float())

    def test_relative_paths_are_project_relative(self):
        settings = {**self.settings, "root": "cache", "split_files": {name: name+".txt" for name in ["train", "val", "test"]}}
        with patch("data.PROJECT", self.base):
            self.assertEqual(SEVIRDataset(settings, EVENTS)[0]["event_id"], "R0001")

    def test_training_cache_cannot_be_validation(self):
        with self.assertRaisesRegex(ValueError, "划分"):
            SEVIRDataset({**self.settings, "split": "val"}, EVENTS)

    def test_overlapping_source_windows_cannot_cross_splits(self):
        Path(self.settings["split_files"]["val"]).write_text("vil-2017-test.h5-0-1.npy\n")
        with self.assertRaisesRegex(ValueError, "共享源记录"):
            SEVIRDataset(self.settings, EVENTS)

    def test_official_list_and_selection_integrity(self):
        p = Path(self.settings["split_files"]["train"])
        original = p.read_bytes()
        p.write_bytes(original+b"\n")
        with self.assertRaisesRegex(ValueError, "训练清单"):
            SEVIRDataset(self.settings, EVENTS)
        p.write_bytes(original)
        p = self.root / "selection.json"
        p.write_bytes(p.read_bytes()+b"\n")
        with self.assertRaisesRegex(ValueError, "抽样清单"):
            SEVIRDataset(self.settings, EVENTS)

    def test_duplicate_source_records_are_rejected(self):
        self.selection["records"].append({**self.record, "event_id": "R0002"})
        self.selection["sample_size"] = 2
        self.write_cache()
        self.acquisition["records"].append({**self.acquisition["records"][0], "event_id": "R0002"})
        (self.root / "acquisition.json").write_text(json.dumps(self.acquisition))
        with self.assertRaisesRegex(ValueError, "重复"):
            SEVIRDataset(self.settings, EVENTS)

    def test_bad_checksum_shape_and_dtype_are_rejected(self):
        dataset = SEVIRDataset(self.settings, EVENTS)
        self.sample.write_bytes(b"wrong sample")
        with self.assertRaisesRegex(ValueError, "校验值"):
            dataset[0]
        for frames in [np.zeros((24, 64, 64), dtype=np.uint8), self.frames.astype(np.float32)]:
            self.frames = frames
            self.write_cache()
            with self.assertRaisesRegex(ValueError, "实际帧数"):
                SEVIRDataset(self.settings, EVENTS)[0]

    def test_irregular_timing_and_different_window_are_rejected(self):
        offsets = self.record["minute_offsets"].split(":")
        offsets[13] = str(int(offsets[13])+1)
        self.record["minute_offsets"] = ":".join(offsets)
        self.write_cache()
        with self.assertRaisesRegex(ValueError, "帧间隔"):
            SEVIRDataset(self.settings, EVENTS)
        self.selection["window_part"] = 1
        self.write_cache()
        with self.assertRaisesRegex(ValueError, "part 0"):
            SEVIRDataset(self.settings, EVENTS)

    def test_data_event_score_mask_contract(self):
        self.settings["width"] = 128
        self.frames = np.zeros((25, 64, 128), dtype=np.uint8)
        self.frames[13:16, :8, 64:72] = 181
        self.frames[14, 0, 0] = 255
        self.write_cache()
        sample = SEVIRDataset(self.settings, EVENTS)[0]
        self.assertFalse(sample["future_valid"][1, 0, 0, 0])
        self.assertEqual(sample["future"][1, 0, 0, 0].item(), 0)
        history = restore_vil(sample["history"], sample["history_valid"], 255.)[None].numpy()
        future = restore_vil(sample["future"], sample["future_valid"], 255.)[None, None].numpy()
        observed, valid = extract_events(history, future, sample["history_valid"][None].numpy(),
                                         sample["future_valid"][None, None].numpy(), EVENTS)
        # 两个手工成员仅验证接口：一个全零，一个保留可知值；没有模型预测。
        members = np.concatenate([np.zeros_like(future), np.nan_to_num(future)], axis=1)
        predicted, _ = extract_events(history, members, sample["history_valid"][None].numpy(),
                                       np.ones_like(members, dtype=bool), EVENTS)
        score, _ = fair_brier(torch.from_numpy(predicted), torch.from_numpy(observed[:, 0]), torch.from_numpy(valid[:, 0]))
        self.assertTrue(torch.isnan(score[0, :3]).all())
        self.assertEqual(masked_mean(score, torch.from_numpy(valid[:, 0]), dim=-1).item(), 0)


if __name__ == "__main__":
    unittest.main()
