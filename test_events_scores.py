"""验证事件边界、有限集合修正和梯度方向；运行 python -m unittest -v。"""

import csv
import itertools
import json
from pathlib import Path
import unittest

import numpy as np
import torch
import yaml

from events import EVENT_NAMES, extract_events
from scores import fair_brier, fair_crps, masked_mean
from data import SEVIRDataset, restore_vil


ROOT = Path(__file__).resolve().parent
CONFIG = yaml.safe_load((ROOT / "config.yaml").read_text())
SETTINGS = CONFIG["events"]


def scene(height=64, width=64, batch=1, members=1):
    """只构造事件算子的数值输入，不模拟模型或训练过程。"""
    history = np.zeros((batch, 13, 1, height, width), dtype=np.float32)
    futures = np.zeros((batch, members, 12, 1, height, width), dtype=np.float32)
    return history, futures, np.ones_like(history, dtype=bool), np.ones_like(futures, dtype=bool)


class EventTests(unittest.TestCase):
    def test_exact_threshold_area_duration(self):
        for value, side, count, expected in [(181, 8, 3, [1, 1, 1]),
                                             (180.99, 8, 3, [0, 0, 0]),
                                             (181, 7, 3, [0, 0, 0]),
                                             (181, 8, 2, [0, 1, 1])]:
            with self.subTest(value=value, side=side, count=count):
                args = scene()
                args[1][0, 0, :count, 0, :side, :side] = value
                values, valid = extract_events(*args, SETTINGS)
                np.testing.assert_array_equal(values[0, 0], expected)
                self.assertTrue(valid.all())

    def test_horizon_inclusive_and_late_persistence(self):
        for start, expected in [(5, [0, 1, 1]), (6, [0, 0, 1]), (9, [1, 0, 1])]:
            args = scene()
            args[1][0, 0, start:start+(3 if start == 9 else 1), 0, :8, :8] = 181
            values, _ = extract_events(*args, SETTINGS)
            np.testing.assert_array_equal(values[0, 0], expected)

    def test_history_eligibility_is_shared_across_members(self):
        args = scene(members=2)
        args[0][0, -1, 0, :8, :8] = 181
        args[1][0, :, :3, 0, :8, :8] = 181
        values, valid = extract_events(*args, SETTINGS)
        self.assertTrue(np.isnan(values[0, :, 1]).all())
        self.assertFalse(valid[0, :, 1].any())
        np.testing.assert_array_equal(values[0, :, [0, 2]], 1)

    def test_earlier_activity_does_not_disqualify_return(self):
        args = scene()
        args[0][0, 0, 0, :8, :8] = 181
        args[1][0, 0, 0, 0, :8, :8] = 181
        values, valid = extract_events(*args, SETTINGS)
        self.assertEqual(values[0, 0, 1], 1)
        self.assertTrue(valid.all())

    def test_total_area_is_not_connected_area(self):
        args = scene()
        args[1][0, 0, :3, 0, :24:3, :24:3] = 181
        values, _ = extract_events(*args, SETTINGS)
        np.testing.assert_array_equal(values[0, 0], [1, 1, 0])

    def test_eight_connectivity_and_tile_boundaries(self):
        args = scene()
        for i in range(64):
            args[1][0, 0, 0, 0, i, i] = 181
        values, _ = extract_events(*args, SETTINGS)
        np.testing.assert_array_equal(values[0, 0], [0, 1, 1])
        args = scene(height=128)
        args[1][0, 0, 0, 0, 60:68, :8] = 181
        values, _ = extract_events(*args, SETTINGS)
        # 全图中连通的64像素分处两个格网，各自只有32，不能跨格网拼接。
        np.testing.assert_array_equal(values, 0)

    def test_row_major_order_and_batch_member_dimensions(self):
        args = scene(height=128, width=128, batch=2, members=2)
        args[1][1, 1, :3, 0, 64:72, :8] = 181
        values, valid = extract_events(*args, SETTINGS)
        self.assertEqual(values.shape, (2, 2, 12))
        expected = np.zeros((2, 2, 12), dtype=np.float32)
        expected[1, 1, 6:9] = 1
        np.testing.assert_array_equal(values, expected)
        self.assertTrue(valid.all())
        self.assertEqual(EVENT_NAMES, ("sustained_3obs", "trigger_by_30min", "connected_area_any"))

    def test_missing_and_nonfinite_are_unknown(self):
        for which, value in [(0, 255), (1, 255), (1, float("nan"))]:
            args = scene()
            args[which].flat[0] = value
            if np.isfinite(value):
                args[which+2].flat[0] = False
            values, valid = extract_events(*args, SETTINGS)
            self.assertTrue(np.isnan(values).all())
            self.assertFalse(valid.any())

    def test_generated_255_is_not_automatically_missing(self):
        args = scene()
        args[1][0, 0, :3, 0, :8, :8] = 255
        values, valid = extract_events(*args, SETTINGS)
        np.testing.assert_array_equal(values, 1)
        self.assertTrue(valid.all())

    def test_invalid_shapes_masks_and_configuration(self):
        args = scene(height=65)
        with self.assertRaises(ValueError):
            extract_events(*args, SETTINGS)
        args = scene()
        with self.assertRaises(ValueError):
            extract_events(args[0], args[1][0], args[2], args[3], SETTINGS)
        with self.assertRaises(ValueError):
            extract_events(args[0], args[1], args[2].astype(float), args[3], SETTINGS)
        for key, value in [("consecutive_frames", 0), ("onset_horizon_frames", 13),
                           ("tile_size", 0), ("area_pixels", 4097), ("history_frames", 12),
                           ("consecutive_frames", 4), ("onset_horizon_frames", 5),
                           ("frame_interval_minutes", 10), ("version", "unimplemented_v2")]:
            with self.subTest(key=key), self.assertRaises(ValueError):
                extract_events(*args, {**SETTINGS, key: value})


class ScoreTests(unittest.TestCase):
    def test_hand_calculated_brier_and_reward_credit(self):
        samples = torch.tensor([[0., 0., 1., 1.]], dtype=torch.float64)
        score, credit = fair_brier(samples, torch.ones(1, dtype=torch.float64))
        torch.testing.assert_close(score, torch.tensor([1/6], dtype=torch.float64))
        torch.testing.assert_close(credit, torch.tensor([[0., 0., 1/3, 1/3]], dtype=torch.float64))

    def test_hand_calculated_crps_and_reward_credit(self):
        x = torch.tensor([[-1., 0., 2.]], dtype=torch.float64, requires_grad=True)
        score, credit = fair_crps(x, torch.ones(1, dtype=torch.float64))
        torch.testing.assert_close(score, torch.tensor([1/3], dtype=torch.float64))
        torch.testing.assert_close(credit, torch.tensor([[0., 1/6, .5]], dtype=torch.float64))
        self.assertTrue(score.requires_grad)
        self.assertFalse(credit.requires_grad)

    def test_crps_value_gradient_and_tie_symmetry(self):
        x = torch.tensor([[-1., .2, 2.5]], dtype=torch.float64, requires_grad=True)
        self.assertTrue(torch.autograd.gradcheck(lambda z: fair_crps(z, torch.tensor([1.3], dtype=z.dtype))[0], (x,)))
        ties = torch.zeros((1, 2), dtype=torch.float64, requires_grad=True)
        score, _ = fair_crps(ties, torch.ones(1, dtype=torch.float64))
        torch.testing.assert_close(torch.autograd.grad(score.sum(), ties)[0], torch.full_like(ties, -.5))

    def test_binary_expectation_and_policy_gradient_by_exact_enumeration(self):
        q = .3
        for k, p in itertools.product([2, 3, 4], [.05, .3, .8]):
            loss = gradient = joint = 0.
            for events in itertools.product([0., 1.], repeat=k):
                x = torch.tensor([events], dtype=torch.float64)
                probability = p**sum(events) * (1-p)**(k-sum(events))
                for target in [0., 1.]:
                    weight = probability * (q if target else 1-q)
                    score, credit = fair_brier(x, torch.tensor([target], dtype=torch.float64))
                    loss += weight * score.item()
                    gradient += weight * (credit * (x-p)).sum().item()
                    joint += weight * -score.item() * (sum(events)-k*p)
            self.assertAlmostEqual(loss, (p-q)**2 + q*(1-q), places=12)
            self.assertAlmostEqual(gradient, -2*(p-q)*p*(1-p), places=12)
            self.assertAlmostEqual(gradient, joint, places=12)

    def test_crps_expectation_and_policy_gradient_by_exact_enumeration(self):
        support = torch.tensor([-1., .5, 2.], dtype=torch.float64)
        p = torch.tensor([.2, .5, .3], dtype=torch.float64)
        q = torch.tensor([.45, .25, .3], dtype=torch.float64)
        distances = (support[:, None]-support[None, :]).abs()
        logits = p.log().requires_grad_()
        probabilities = logits.softmax(0)
        population = probabilities @ distances @ q - .5 * probabilities @ distances @ probabilities
        expected_gradient = torch.autograd.grad(-population, logits)[0]
        for k in [2, 4]:
            loss = 0.
            gradient = torch.zeros(3, dtype=torch.float64)
            joint = torch.zeros_like(gradient)
            for indices in itertools.product(range(3), repeat=k):
                x = support[list(indices)][None]
                logit_scores = torch.eye(3, dtype=torch.float64)[list(indices)] - p
                probability = p[list(indices)].prod().item()
                for target in range(3):
                    weight = probability * q[target].item()
                    score, credit = fair_crps(x, support[target:target+1])
                    loss += weight * score.item()
                    gradient += weight * (credit[0, :, None]*logit_scores).sum(0)
                    joint += weight * -score.item() * logit_scores.sum(0)
            self.assertAlmostEqual(loss, population.item(), places=12)
            torch.testing.assert_close(gradient, expected_gradient, atol=1e-12, rtol=0)
            torch.testing.assert_close(gradient, joint, atol=1e-12, rtol=0)

    def test_masks_exclude_unknown_without_removing_members(self):
        x = torch.tensor([[[0., float("nan")], [1., float("nan")]]])
        y = torch.tensor([[1., float("nan")]])
        valid = torch.tensor([[True, False]])
        for function in [fair_brier, fair_crps]:
            score, credit = function(x, y, valid)
            self.assertTrue(torch.isnan(score[0, 1]))
            self.assertTrue((credit[:, :, 1] == 0).all())
            self.assertEqual(masked_mean(score, valid, dim=-1).item(), 0.)
            self.assertEqual(masked_mean(credit, valid[:, None], dim=-1).shape, (1, 2))
            with self.assertRaises(ValueError):
                function(x, y, torch.ones_like(valid))
            with self.assertRaises(ValueError):
                function(x, y, valid[:, None].expand_as(x))

    def test_empty_mask_and_unknown_prediction_fail_loudly(self):
        x = torch.tensor([[[float("nan")], [1.]]])
        y = torch.ones((1, 1))
        for function in [fair_brier, fair_crps]:
            with self.assertRaises(ValueError):
                function(x, y)
            valid = torch.zeros_like(y, dtype=torch.bool)
            score, credit = function(x, y, valid)
            self.assertTrue(torch.isnan(score).all())
            self.assertFalse(credit.any())
            with self.assertRaises(ValueError):
                masked_mean(score, valid, dim=-1)
        with self.assertRaises(ValueError):
            masked_mean(torch.ones((2, 2, 3)), torch.ones((2, 3), dtype=torch.bool), dim=-1)

    def test_reject_single_member_and_soft_binary_labels(self):
        for function in [fair_brier, fair_crps]:
            with self.assertRaises(ValueError):
                function(torch.zeros((1, 1)), torch.zeros(1))
        with self.assertRaises(ValueError):
            fair_brier(torch.tensor([[0., .5]]), torch.zeros(1))

    def test_integer_inputs_and_low_precision_are_promoted(self):
        for dtype in [torch.uint8, torch.float16, torch.bfloat16]:
            score, credit = fair_crps(torch.tensor([[0, 255]], dtype=dtype), torch.tensor([127], dtype=dtype))
            self.assertEqual(score.dtype, torch.float32)
            self.assertEqual(credit.dtype, torch.float32)
            torch.testing.assert_close(score, torch.zeros_like(score))

    def test_member_permutation_preserves_score_and_credit(self):
        x = torch.tensor([[[0., 1.], [1., 0.], [1., 1.], [0., 0.]]], dtype=torch.float64)
        y = torch.tensor([[1., 0.]], dtype=torch.float64)
        permutation = [2, 0, 3, 1]
        for function in [fair_brier, fair_crps]:
            score, credit = function(x, y)
            permuted_score, permuted_credit = function(x[:, permutation], y)
            torch.testing.assert_close(score, permuted_score)
            torch.testing.assert_close(credit[:, permutation], permuted_credit)

    def test_unknown_coordinates_do_not_poison_value_gradients(self):
        x = torch.tensor([[[0., float("nan")], [2., float("nan")]]], requires_grad=True)
        valid = torch.tensor([[True, False]])
        score, _ = fair_crps(x, torch.tensor([[1., float("nan")]]), valid)
        masked_mean(score, valid, dim=-1).sum().backward()
        self.assertTrue(torch.isfinite(x.grad).all())
        self.assertTrue((x.grad[:, :, 1] == 0).all())


@unittest.skipUnless((ROOT / "data/sevir_event100/acquisition.json").exists(), "本地100个训练记录未下载；跳过真实数据回归。")
class RealDataTests(unittest.TestCase):
    def test_100_records_match_independently_audited_labels(self):
        reference = {}
        with (ROOT / "outputs/event100/tile_event_labels.csv").open() as handle:
            for row in csv.DictReader(handle):
                if float(row["threshold"]) == SETTINGS["threshold"] and int(row["area_pixels"]) == SETTINGS["area_pixels"]:
                    reference.setdefault(row["event_id"], []).append(row)
        selection = json.loads((ROOT / "data/sevir_event100/selection.json").read_text())
        dataset = SEVIRDataset(CONFIG["data"], SETTINGS)
        self.assertEqual(len(dataset), 100)
        loader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=False)
        totals = np.zeros(3, dtype=int)
        counts = np.zeros(3, dtype=int)
        seen = []
        for batch in loader:
            history = restore_vil(batch["history"], batch["history_valid"], dataset.scale).numpy()
            future = restore_vil(batch["future"], batch["future_valid"], dataset.scale).numpy()[:, None]
            values, valid = extract_events(history, future, batch["history_valid"].numpy(),
                                           batch["future_valid"].numpy()[:, None], SETTINGS)
            self.assertEqual(values.shape, (len(batch["event_id"]), 1, 108))
            for index, event_id in enumerate(batch["event_id"]):
                seen.append(event_id)
                group = sorted(reference[event_id], key=lambda row: (int(row["tile_row"]), int(row["tile_column"])))
                expected = np.array([[row[name] == "True" for name in EVENT_NAMES] for row in group], dtype=np.float32)
                known = np.array([row["valid"] == "True" for row in group])
                eligible = np.array([row["onset_eligible"] == "True" for row in group])
                expected_valid = np.stack([known, eligible & known, known], axis=-1)
                expected[~expected_valid] = np.nan
                np.testing.assert_array_equal(valid[index, 0].reshape(36, 3), expected_valid)
                np.testing.assert_allclose(values[index, 0].reshape(36, 3), expected, atol=0, rtol=0, equal_nan=True)
            totals += np.nansum(values.reshape(-1, 3), axis=0).astype(int)
            counts += valid.reshape(-1, 3).sum(axis=0)
        self.assertEqual(seen, [row["event_id"] for row in selection["records"]])
        np.testing.assert_array_equal(totals, [266, 105, 289])
        np.testing.assert_array_equal(counts, [3600, 3402, 3600])


if __name__ == "__main__":
    unittest.main()
