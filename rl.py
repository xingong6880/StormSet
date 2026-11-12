"""随机 DDIM 轨迹、固定正方差转移与集合策略梯度。

采样时不建图；更新时用已 detach 的状态重算均值，再计算记录动作的密度。
状态/动作均为 [B,K,...]，后面的维度共同构成一个动作，密度在这些维度求和。
各步 log probability 相加得到 [B,K] 的轨迹对数概率，不对时间或动作坐标平均。
末步显式定义额外高斯输出转移，区别于官方确定性末步。
只支持新轨迹的一次 on-policy 更新；不实现 PPO、KL 或优化器。
"""

import math

import torch


def _gaussian_parameters(mean, std):
    """检查动作形状与固定标准差；半精度密度至少使用 float32 计算。"""
    if mean.ndim < 3 or not mean.is_floating_point() or any(size == 0 for size in mean.shape):
        raise ValueError("高斯均值必须是非空的浮点 [B,K,...]，至少有一个动作维度。")
    if isinstance(std, torch.Tensor) and std.requires_grad:
        raise ValueError("当前只支持固定标准差，不支持学习转移方差。")
    mean = mean if mean.dtype == torch.float64 else mean.float()
    std = torch.as_tensor(std, device=mean.device, dtype=mean.dtype)
    if std.ndim not in (0, mean.ndim):
        raise ValueError("标准差必须是标量或与均值同维数；变化维度请显式插入单例维。")
    try:
        std = torch.broadcast_to(std, mean.shape)
    except RuntimeError as error:
        raise ValueError("标准差不能广播到动作形状。") from error
    if not torch.isfinite(mean).all() or not torch.isfinite(std).all() or (std <= 0).any():
        raise ValueError("均值必须有限，标准差必须有限且严格为正；不能将确定性转移当作高斯密度。")
    return mean, std


@torch.no_grad()
def sample_gaussian_transition(mean, std, generator=None):
    """逐坐标独立采样未裁剪的动作，不保留路径梯度；可传入 Torch generator。

    整个 batch/ensemble 一次抽取噪声；不要在成员间重置到同一个随机种子。
    独立成员还要求条件固定、均值计算不混合成员，完整轨迹由后续模型接口组织。
    """
    mean, std = _gaussian_parameters(mean, std)
    noise = torch.randn(mean.shape, dtype=mean.dtype, device=mean.device, generator=generator)
    action = mean + std * noise
    if not torch.isfinite(action).all():
        raise ValueError("高斯采样出现非有限值，停止使用该轨迹。")
    return action


def gaussian_transition_log_prob(action, mean, std):
    """重算固定动作的联合对数密度，返回 [B,K]；仅均值保留参数梯度。

    mean 必须由已 detach 的采样状态计算，动作在本函数中 detach。不能裁剪动作、
    把零标准差夹到 epsilon，或省略仍依赖参数的确定性末步后声称目标不变。
    """
    mean, std = _gaussian_parameters(mean, std)
    if (action.shape != mean.shape or action.device != mean.device
            or not action.is_floating_point() or not torch.isfinite(action).all()):
        raise ValueError("记录动作必须是与均值同形、同设备的有限浮点张量。")
    residual = (action.detach().to(mean.dtype) - mean) / std
    per_coordinate = -.5 * residual.square() - std.log() - .5 * math.log(2 * math.pi)
    log_prob = per_coordinate.flatten(start_dim=2).sum(dim=-1)
    if not torch.isfinite(log_prob).all():
        raise ValueError("转移对数概率溢出，停止更新，不能静默更改密度的约简方式。")
    return log_prob


def ensemble_policy_loss(trajectory_log_prob, credit):
    """最大化成员奖励的损失：成员求和、批次平均，credit 强制停止梯度。

    两者均为 [B,K]，K>=2。credit 由 scores.py 先按共享有效坐标聚合，
    已经含 1/K，不能再次按成员平均或用本批其他成员均值作为独立基线。
    """
    if (trajectory_log_prob.ndim != 2 or trajectory_log_prob.shape[0] == 0
            or trajectory_log_prob.shape[1] < 2 or credit.shape != trajectory_log_prob.shape
            or credit.device != trajectory_log_prob.device
            or not trajectory_log_prob.is_floating_point() or not credit.is_floating_point()):
        raise ValueError("轨迹对数概率和已聚合的成员奖励系数必须是同设备的浮点 [B,K]，K>=2。")
    log_prob = trajectory_log_prob if trajectory_log_prob.dtype == torch.float64 else trajectory_log_prob.float()
    credit = credit.detach().to(log_prob.dtype)
    if not torch.isfinite(log_prob).all() or not torch.isfinite(credit).all():
        raise ValueError("策略损失不能包含非有限对数概率或奖励系数。")
    loss = -(credit * log_prob).sum(dim=1).mean()
    if not torch.isfinite(loss):
        raise ValueError("策略损失溢出，停止更新。")
    return loss


def ddim_schedule(settings, device="cpu"):
    """固定 SEVIR 线性 beta、leading 时间格；末步噪声是新策略定义的一部分。

    非末步使用官方 DDIM 的 eta 方差与裁剪 x0 均值。末步均值为裁剪 x0，
    标准差为 terminal_std；不将原本的零方差夹成 epsilon 后冒充原采样分布。
    """
    steps, eta, terminal = settings["steps"], settings["eta"], settings["terminal_std"]
    if isinstance(steps, bool) or not isinstance(steps, int) or not 2 <= steps <= 1000 or 1000 % steps:
        raise ValueError("当前 leading 时间格要求 steps 为 1000 的整数因子，且至少两步。")
    if not 0 < eta <= 1 or not math.isfinite(terminal) or terminal <= 0:
        raise ValueError("随机策略要求 0<eta<=1，以及有限、严格为正的 terminal_std。")
    # 按官方实现先在 CPU 生成 float32 alpha，再搬到目标设备，避免累计乘积算法不同。
    cumulative = (1-torch.linspace(.0001, .02, 1000)).cumprod(0)
    timesteps = torch.arange(steps-1, -1, -1) * (1000 // steps)
    alpha = cumulative[timesteps]
    previous = torch.cat([alpha[1:], torch.ones(1)])
    std = eta * ((1-previous)/(1-alpha)*(1-alpha/previous)).sqrt()
    direction = (1-previous-std.square()).clamp_min(0).sqrt()
    std[-1] = terminal
    return {"timesteps": timesteps.to(device), "alpha": alpha.to(device),
            "previous": previous.to(device), "std": std.to(device), "direction": direction.to(device)}


def ddim_mean(state, epsilon, schedule, index):
    """给定记录状态计算高斯均值；index 可为标量或与 batch 对应的步索引。

    固定 checkpoint 对应 epsilon prediction、x0 裁剪范围 13；不裁剪动作。
    向量 index 用于同时重算多步，绝不把时间步并入成员轴。
    """
    def coefficient(name):
        return schedule[name][index].reshape((-1,) + (1,)*(state.ndim-1))
    alpha = coefficient("alpha")
    x0 = ((state-(1-alpha).sqrt()*epsilon)/alpha.sqrt()).clamp(-13, 13)
    return coefficient("previous").sqrt()*x0 + coefficient("direction")*epsilon


def _versions(model):
    return tuple(parameter._version for parameter in model.denoiser.parameters())


@torch.no_grad()
def sample_trajectories(model, condition, schedule, generator=None):
    """独立采样整组轨迹；只接收条件 latent，不接收未来真值。

    states 为 [S+1,B,K,...]，保存未裁剪动作和固定条件供更新重算。
    初始噪声与条件后验均不依赖可训练参数，其密度无需进入策略梯度。
    """
    condition = condition.detach().clone()
    schedule = {key: value.detach().clone() for key, value in schedule.items()}
    state = torch.randn(condition.shape, device=condition.device, dtype=condition.dtype, generator=generator)
    states, log_probs = [state], []
    for i, timestep in enumerate(schedule["timesteps"]):
        times = timestep.expand(condition.shape[:2])
        epsilon = model.predict_noise(state, times, condition)
        mean = ddim_mean(state, epsilon, schedule, i)
        state = sample_gaussian_transition(mean, schedule["std"][i], generator)
        log_probs.append(gaussian_transition_log_prob(state, mean, schedule["std"][i]))
        states.append(state)
    return {"states": torch.stack(states), "condition": condition, "schedule": schedule,
            "log_prob": torch.stack(log_probs), "versions": _versions(model), "model_id": id(model), "used": False}


def backward_trajectory(model, trajectory, credit, step_batch_size):
    """重算记录轨迹并累计完整策略梯度；不调用 optimizer.step 或再次采样。

    先尝试整批时间步；显存不足时可按实测选择分块。分块只拆时间步计算，
    各块损失相加，保持全部 S 步、全部 K 成员和逻辑 batch 的原有权重。
    """
    if trajectory["used"] or trajectory["model_id"] != id(model) or trajectory["versions"] != _versions(model):
        raise ValueError("轨迹已使用或策略参数已改变；需要重新进行 on-policy rollout。")
    if isinstance(step_batch_size, bool) or not isinstance(step_batch_size, int) or step_batch_size < 1:
        raise ValueError("step_batch_size 必须为正整数。")
    states, condition, schedule = trajectory["states"], trajectory["condition"], trajectory["schedule"]
    steps, batch, members = states.shape[0]-1, *condition.shape[:2]
    if credit.shape != (batch, members):
        raise ValueError("成员奖励系数必须先聚合为 [B,K]。")
    loss_total, max_error = 0., 0.
    for begin in range(0, steps, step_batch_size):
        end = min(begin+step_batch_size, steps)
        count = end-begin
        state = states[begin:end].detach().flatten(0, 1)
        action = states[begin+1:end+1].detach().flatten(0, 1)
        fixed_condition = condition.detach().unsqueeze(0).expand(count, *condition.shape).flatten(0, 1)
        times = schedule["timesteps"][begin:end, None, None].expand(count, batch, members).flatten(0, 1)
        indices = torch.arange(begin, end, device=state.device).repeat_interleave(batch)
        epsilon = model.predict_noise(state, times, fixed_condition)
        mean = ddim_mean(state, epsilon, schedule, indices)
        std = schedule["std"][indices].reshape((-1,) + (1,)*(state.ndim-1))
        log_prob = gaussian_transition_log_prob(action, mean, std).reshape(count, batch, members)
        recorded = trajectory["log_prob"][begin:end]
        # 不同步长 batch 的 GPU 算法可能有舍入差异；记录绝对误差，不声称逐位一致。
        if not torch.allclose(log_prob.detach(), recorded, rtol=1e-5, atol=.125):
            raise ValueError("重算概率与采样概率不一致，停止更新。")
        max_error = max(max_error, (log_prob.detach()-recorded).abs().max().item())
        loss = ensemble_policy_loss(log_prob.sum(dim=0), credit)
        loss.backward()
        loss_total += loss.detach().item()
    trajectory["used"] = True
    return {"policy_loss": loss_total, "log_prob_max_abs_error": max_error}
