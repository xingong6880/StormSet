"""验证 RL 的概率、停止梯度与期望更新方向；不运行雷达模型或训练。"""

import itertools
import math
import unittest

import numpy as np
import torch
from torch.distributions import Independent, Normal

from rl import (backward_trajectory, ddim_mean, ddim_schedule, ensemble_policy_loss,
                gaussian_transition_log_prob, sample_gaussian_transition, sample_trajectories)
from scores import fair_brier, fair_crps, masked_mean


class GaussianTests(unittest.TestCase):
    def test_joint_density_matches_torch_distribution(self):
        mean = torch.linspace(-1, 1, 48, dtype=torch.float64).reshape(2, 3, 2, 4)
        action = mean.square() + .2
        for std in [.7, torch.linspace(.3, 1, 6, dtype=mean.dtype).reshape(2, 3, 1, 1)]:
            expected = Independent(Normal(mean, std), 2).log_prob(action)
            actual = gaussian_transition_log_prob(action, mean, std)
            self.assertEqual(actual.shape, (2, 3))
            torch.testing.assert_close(actual, expected)

    def test_action_stops_gradient_while_mean_keeps_score_gradient(self):
        mean = torch.tensor([[[.2, -.3], [.5, .8]]], dtype=torch.float64, requires_grad=True)
        action = mean + torch.tensor([[[.4, -.6], [.1, .3]]], dtype=mean.dtype)
        result = gaussian_transition_log_prob(action, mean, .5)
        grad_mean, grad_action = torch.autograd.grad(result.sum(), (mean, action), allow_unused=True)
        self.assertIsNone(grad_action)
        torch.testing.assert_close(grad_mean, (action.detach()-mean.detach()) / .5**2)

    def test_action_coordinates_are_summed(self):
        action = torch.zeros((1, 2, 7), dtype=torch.float64)
        actual = gaussian_transition_log_prob(action, torch.zeros_like(action), 1.)
        torch.testing.assert_close(actual, torch.full((1, 2), -3.5*math.log(2*math.pi), dtype=action.dtype))

    def test_sampling_uses_one_noise_array_and_stops_gradient(self):
        mean = torch.arange(12., dtype=torch.float64).reshape(2, 3, 2).requires_grad_()
        generator = torch.Generator().manual_seed(17)
        action = sample_gaussian_transition(mean, .4, generator)
        expected_noise = torch.randn(mean.shape, dtype=mean.dtype, generator=torch.Generator().manual_seed(17))
        torch.testing.assert_close(action, mean.detach()+.4*expected_noise)
        self.assertFalse(action.requires_grad)
        self.assertFalse(torch.equal(expected_noise[:, 0], expected_noise[:, 1]))
        repeat = sample_gaussian_transition(mean, .4, torch.Generator().manual_seed(17))
        torch.testing.assert_close(action, repeat)

    def test_half_precision_density_uses_float32(self):
        for dtype in [torch.float16, torch.bfloat16]:
            mean = torch.zeros((1, 2, 32), dtype=dtype, requires_grad=True)
            action = torch.full_like(mean, .125)
            actual = gaussian_transition_log_prob(action, mean, .01)
            expected = gaussian_transition_log_prob(action.float(), mean.float(), .01)
            self.assertEqual(actual.dtype, torch.float32)
            torch.testing.assert_close(actual, expected)
            self.assertTrue(torch.isfinite(torch.autograd.grad(actual.sum(), mean)[0]).all())
            self.assertEqual(sample_gaussian_transition(mean, .01).dtype, torch.float32)

    def test_rejects_deterministic_nonfinite_and_ambiguous_scale(self):
        mean = torch.zeros((2, 3, 2))
        for std in [0., -1., float("nan"), float("inf"), torch.ones((2, 3)), torch.ones((2, 4, 1))]:
            for function in [lambda: sample_gaussian_transition(mean, std),
                             lambda: gaussian_transition_log_prob(mean, mean, std)]:
                with self.subTest(std=std), self.assertRaises(ValueError):
                    function()
        for bad in [torch.zeros((2, 3)), torch.zeros((0, 3, 2)), mean.long(), mean+float("nan")]:
            with self.assertRaises(ValueError):
                sample_gaussian_transition(bad, .5)
        for bad in [mean[:, :, :1], mean.long(), mean+float("inf")]:
            with self.assertRaises(ValueError):
                gaussian_transition_log_prob(bad, mean, .5)
        with self.assertRaises(ValueError):
            gaussian_transition_log_prob(mean+1e30, mean, 1e-30)

    def test_learned_scale_is_not_silently_detached(self):
        mean = torch.zeros((1, 2, 3))
        for dtype in [torch.float32, torch.float64]:
            std = torch.tensor(.5, dtype=dtype, requires_grad=True)
            for function in [lambda: sample_gaussian_transition(mean, std),
                             lambda: gaussian_transition_log_prob(mean, mean, std)]:
                with self.subTest(dtype=dtype), self.assertRaises(ValueError):
                    function()


class PolicyTests(unittest.TestCase):
    def test_member_sum_batch_mean_and_detached_credit(self):
        log_prob = torch.tensor([[-1., -2., -3.], [-4., -5., -6.]], requires_grad=True)
        credit = torch.tensor([[.1, .2, .3], [-.2, .1, .4]], requires_grad=True)
        loss = ensemble_policy_loss(log_prob, credit)
        self.assertAlmostEqual(loss.item(), 1.75, places=6)
        grad_prob, grad_credit = torch.autograd.grad(loss, (log_prob, credit), allow_unused=True)
        torch.testing.assert_close(grad_prob, -credit.detach()/2)
        self.assertIsNone(grad_credit)
        permutation = [2, 0, 1]
        torch.testing.assert_close(loss, ensemble_policy_loss(log_prob[:, permutation], credit[:, permutation]))

    def test_gaussian_event_update_matches_exact_expected_brier_gradient(self):
        # 给定越阈事件后，Gaussian 的均值 score 对动作是线性的。
        # 用截断高斯的条件均值代表各事件，枚举全部成员/真值组合可精确积分梯度。
        q, threshold, std = .3, .2, .7
        for location, members in itertools.product([-.6, .4, 1.2], [2, 4]):
            z = (location-threshold)/std
            p = .5*(1+math.erf(z/math.sqrt(2)))
            phi = math.exp(-.5*z*z)/math.sqrt(2*math.pi)
            conditional_means = [location-std*phi/(1-p), location+std*phi/p]
            expected = 2*(p-q)*phi/std  # 最小化 E[fair Brier] 的解析导数
            integrated = 0.
            for events in itertools.product([0., 1.], repeat=members):
                probability = p**sum(events)*(1-p)**(members-sum(events))
                action = torch.tensor([[[conditional_means[int(e)]] for e in events]], dtype=torch.float64)
                for target in [0., 1.]:
                    weight = probability*(q if target else 1-q)
                    mean = torch.tensor(location, dtype=torch.float64, requires_grad=True)
                    log_prob = gaussian_transition_log_prob(action, mean.expand_as(action), std)
                    _, credit = fair_brier(torch.tensor([events], dtype=mean.dtype), torch.tensor([target], dtype=mean.dtype))
                    loss = ensemble_policy_loss(log_prob, credit)
                    integrated += weight*torch.autograd.grad(loss, mean)[0].item()
            self.assertAlmostEqual(integrated, expected, places=12)

    def test_two_step_trajectory_gradient_matches_analytic_expectation(self):
        # 两步线性高斯链的二次终端损失可解析积分；这不是扩散模型替身或训练结果。
        # 三节点 Gauss-Hermite 对这里的低阶多项式精确，两个成员的噪声用乘积分布。
        nodes, weights = np.polynomial.hermite.hermgauss(3)
        nodes, weights = nodes*math.sqrt(2), weights/math.sqrt(math.pi)
        location, coupling, first_std, second_std = .4, .6, .7, .3
        integrated = 0.
        for indices in itertools.product(range(3), repeat=4):
            weight = math.prod(weights[i] for i in indices)
            noise = torch.tensor(nodes[list(indices)], dtype=torch.float64).reshape(1, 2, 2)
            first = (location + first_std*noise[:, :, 0:1]).requires_grad_()
            second = (location + coupling*first.detach() + second_std*noise[:, :, 1:2]).requires_grad_()
            mean = torch.tensor(location, dtype=torch.float64, requires_grad=True)
            first_log = gaussian_transition_log_prob(first, mean.expand_as(first), first_std)
            second_log = gaussian_transition_log_prob(second, mean+coupling*first.detach(), second_std)
            credit = -second.squeeze(-1).square()/2
            loss = ensemble_policy_loss(first_log+second_log, credit)
            gradient, grad_first, grad_second = torch.autograd.grad(loss, (mean, first, second), allow_unused=True)
            self.assertIsNone(grad_first)
            self.assertIsNone(grad_second)
            integrated += weight*gradient.item()
        self.assertAlmostEqual(integrated, 2*location*(1+coupling)**2, places=12)

    def test_score_masks_aggregate_before_policy_loss(self):
        samples = torch.tensor([[[0., float("nan"), 1.], [1., float("nan"), 1.]]])
        target = torch.tensor([[1., float("nan"), 0.]])
        valid = torch.tensor([[True, False, True]])
        log_prob = torch.tensor([[-2., -3.]], requires_grad=True)
        for function in [fair_brier, fair_crps]:
            _, credit = function(samples, target, valid)
            combined = masked_mean(credit, valid[:, None], dim=-1)
            loss = ensemble_policy_loss(log_prob, combined)
            torch.testing.assert_close(loss, -(combined*log_prob).sum())
            torch.testing.assert_close(torch.autograd.grad(loss, log_prob)[0], -combined)

    def test_half_precision_policy_loss_uses_float32(self):
        log_prob = torch.full((1, 4), -1000., dtype=torch.float16, requires_grad=True)
        loss = ensemble_policy_loss(log_prob, torch.full_like(log_prob, 100.))
        self.assertEqual(loss.dtype, torch.float32)
        self.assertEqual(loss.item(), 400000.)
        torch.testing.assert_close(torch.autograd.grad(loss, log_prob)[0], torch.full_like(log_prob, -100.))

    def test_invalid_policy_inputs_are_not_broadcast_or_filtered(self):
        log_prob = torch.zeros((2, 3))
        for bad_log, bad_credit in [(log_prob, torch.ones((2, 1))),
                                    (log_prob[:, :1], log_prob[:, :1]),
                                    (log_prob[:, :, None], log_prob[:, :, None]),
                                    (log_prob[:0], log_prob[:0]),
                                    (log_prob.long(), log_prob), (log_prob, log_prob.long()),
                                    (log_prob+float("nan"), log_prob), (log_prob, log_prob+float("inf")),
                                    (log_prob+1e30, log_prob+1e30)]:
            with self.subTest(shape=bad_log.shape), self.assertRaises(ValueError):
                ensemble_policy_loss(bad_log, bad_credit)


class LinearEpsilon(torch.nn.Module):
    """仅用于精确梯度检查的线性 epsilon 算子，不是雷达模型或训练替身。"""

    def __init__(self):
        super().__init__()
        self.denoiser = torch.nn.Linear(1, 1, bias=False, dtype=torch.float64)
        with torch.no_grad():
            self.denoiser.weight.fill_(.1)

    def predict_noise(self, state, timesteps, condition):
        return self.denoiser(state) + .02*condition


class TrajectoryTests(unittest.TestCase):
    settings = {"steps": 4, "eta": 1., "terminal_std": .01}

    def test_schedule_terminal_definition_and_invalid_settings(self):
        schedule = ddim_schedule({**self.settings, "steps": 20})
        torch.testing.assert_close(schedule["timesteps"], torch.arange(950, -1, -50))
        self.assertTrue((schedule["std"] > 0).all())
        self.assertAlmostEqual(schedule["std"][-1].item(), .01)
        self.assertEqual(schedule["direction"][-1].item(), 0.)
        self.assertEqual(schedule["previous"][-1].item(), 1.)
        for key, value in [("steps", 3), ("steps", True), ("steps", 1), ("eta", 0.),
                           ("eta", 1.1), ("terminal_std", 0.), ("terminal_std", float("nan"))]:
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                ddim_schedule({**self.settings, key: value})

    def test_mean_clips_x0_but_not_sampled_terminal_action(self):
        schedule = ddim_schedule(self.settings)
        state = torch.full((1, 2, 1), 100., dtype=torch.float64)
        epsilon = torch.zeros_like(state)
        mean = ddim_mean(state, epsilon, schedule, 3)
        torch.testing.assert_close(mean, torch.full_like(mean, 13.))
        action = sample_gaussian_transition(mean, schedule["std"][-1], torch.Generator().manual_seed(2))
        self.assertTrue((action > 13).any())
        torch.testing.assert_close(mean, ddim_mean(state, epsilon, schedule, torch.tensor([3])))

    def test_rollout_reproducibility_fixed_condition_and_recorded_density(self):
        model = LinearEpsilon()
        condition = torch.zeros((2, 3, 2, 1), dtype=torch.float64, requires_grad=True)
        schedule = ddim_schedule(self.settings)
        first = sample_trajectories(model, condition, schedule, torch.Generator().manual_seed(7))
        second = sample_trajectories(model, condition, schedule, torch.Generator().manual_seed(7))
        torch.testing.assert_close(first["states"], second["states"], rtol=0, atol=0)
        self.assertEqual(first["states"].shape, (5, 2, 3, 2, 1))
        self.assertFalse(first["states"].requires_grad)
        self.assertFalse(first["condition"].requires_grad)
        with torch.no_grad():
            condition.fill_(20.)
        self.assertTrue((first["condition"] == 0).all())
        for i, timestep in enumerate(schedule["timesteps"]):
            state, action = first["states"][i:i+2]
            epsilon = model.predict_noise(state, timestep.expand(2, 3), first["condition"])
            mean = ddim_mean(state, epsilon, schedule, i)
            density = gaussian_transition_log_prob(action, mean, schedule["std"][i])
            torch.testing.assert_close(density, first["log_prob"][i])

    def test_batched_steps_and_accumulated_steps_have_same_complete_gradient(self):
        model = LinearEpsilon()
        condition = torch.zeros((2, 3, 2, 1), dtype=torch.float64)
        credit = torch.tensor([[.1, -.2, .3], [.2, .4, -.1]], dtype=torch.float64, requires_grad=True)
        gradients, losses = [], []
        for chunk in [4, 2, 1]:
            model.zero_grad(set_to_none=True)
            trajectory = sample_trajectories(model, condition, ddim_schedule(self.settings), torch.Generator().manual_seed(7))
            trajectory["states"].requires_grad_()
            result = backward_trajectory(model, trajectory, credit, chunk)
            gradients.append(model.denoiser.weight.grad.clone())
            losses.append(result["policy_loss"])
            self.assertIsNone(trajectory["states"].grad)
            self.assertIsNone(credit.grad)
            self.assertLess(result["log_prob_max_abs_error"], 1e-10)
            with self.assertRaisesRegex(ValueError, "轨迹"):
                backward_trajectory(model, trajectory, credit, chunk)
        for value in gradients[1:]:
            torch.testing.assert_close(value, gradients[0], rtol=1e-12, atol=1e-12)
        for value in losses[1:]:
            self.assertAlmostEqual(value, losses[0], places=11)
        # 单独按定义逐步相加，核对分块代码没有引入额外时间/成员平均。
        schedule = trajectory["schedule"]
        joint = torch.zeros_like(credit)
        for i, timestep in enumerate(schedule["timesteps"]):
            state, action = trajectory["states"][i:i+2].detach()
            mean = ddim_mean(state, model.predict_noise(state, timestep.expand(2, 3), condition), schedule, i)
            joint = joint + gaussian_transition_log_prob(action, mean, schedule["std"][i])
        direct_loss = -(credit.detach()*joint).sum(dim=1).mean()
        direct_gradient = torch.autograd.grad(direct_loss, model.denoiser.weight)[0]
        torch.testing.assert_close(direct_gradient, gradients[0], rtol=1e-12, atol=1e-12)
        self.assertAlmostEqual(direct_loss.item(), losses[0], places=11)

    def test_changed_policy_and_mismatched_credit_reject_before_backward(self):
        model = LinearEpsilon()
        condition = torch.zeros((1, 2, 2, 1), dtype=torch.float64)
        trajectory = sample_trajectories(model, condition, ddim_schedule(self.settings), torch.Generator().manual_seed(7))
        with self.assertRaises(ValueError):
            backward_trajectory(model, trajectory, torch.ones((1, 1)), 2)
        with torch.no_grad():
            model.denoiser.weight.add_(.01)
        with self.assertRaisesRegex(ValueError, "策略参数"):
            backward_trajectory(model, trajectory, torch.ones((1, 2)), 2)


if __name__ == "__main__":
    unittest.main()
