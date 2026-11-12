"""有限集合的 fair score 与奖励方向的成员贡献；不组织 rollout 或优化器。

fair score 使用 Ferro (2014), doi:10.1002/qj.2270 的有限样本修正。
两个评分函数均返回 (score, credit)：score 为 [B,...]，越小越好；
credit 为 [B,K,...]，已 detach，用于最大化 reward=-score 的策略梯度。
按同一坐标权重聚合 credit 后，策略损失为 -(credit * log_probability).sum(K)。
credit 已含 1/K，不能再对 K 求均值；它也不是评价指标。
"""

import torch


def _prepare(samples, observed, valid):
    """只接受共享的观测掩码；不能丢弃坏成员再改变集合大小。"""
    if samples.ndim < 2 or samples.shape[1] < 2:
        raise ValueError("fair score 需要 [B,K,...] 且 K>=2。")
    if observed.shape != samples.shape[:1] + samples.shape[2:]:
        raise ValueError("observed 必须为 [B,...]，与 samples 的非成员维度一致。")
    if observed.device != samples.device or samples.is_complex() or observed.is_complex():
        raise ValueError("输入必须是同一设备上的实数 Torch 张量。")
    if not samples.is_floating_point() or samples.dtype in (torch.float16, torch.bfloat16):
        samples = samples.float()
    observed = observed.to(dtype=samples.dtype)
    if valid is None:
        valid = torch.ones_like(observed, dtype=torch.bool)
    if valid.shape != observed.shape or valid.dtype != torch.bool or valid.device != samples.device:
        raise ValueError("valid 必须是同设备的 [B,...] bool 观测掩码，不能按成员筛选。")
    samples = torch.where(valid.unsqueeze(1), samples, 0)
    observed = torch.where(valid, observed, 0)
    if not torch.isfinite(samples).all() or not torch.isfinite(observed).all():
        raise ValueError("可评分位置存在非有限值；不能通过删除预测成员来评分。")
    return samples, observed, valid


def fair_brier(samples, observed, valid=None):
    """二元事件的 fair Brier 和奖励 credit；输入必须是 0/1，不能传软事件。"""
    samples, observed, valid = _prepare(samples, observed, valid)
    if not ((samples == 0) | (samples == 1)).all() or not ((observed == 0) | (observed == 1)).all():
        raise ValueError("fair Brier 的事件值及观测必须为 0 或 1。")
    members = samples.shape[1]
    mean = samples.mean(dim=1)
    score = (mean - observed).square() - mean * (1 - mean) / (members - 1)
    events, target = samples.detach(), observed.detach()
    other_mean = (events.sum(dim=1, keepdim=True) - events) / (members - 1)
    credit = (2 / members) * events * (target.unsqueeze(1) - other_mean)
    return score.masked_fill(~valid, float("nan")), credit.masked_fill(~valid.unsqueeze(1), 0)


def fair_crps(samples, observed, valid=None):
    """实数场的 fair CRPS 和奖励 credit；score 保留对场数值的梯度。

    每个无序成员对只计算一次，分母为 K(K-1)。逐对计算避免构建
    [B,K,K,T,H,W] 大张量；保持 abs 在相同成员值处的对称零梯度。
    """
    samples, observed, valid = _prepare(samples, observed, valid)
    members = samples.shape[1]
    distance = (samples - observed.unsqueeze(1)).abs()
    score = distance.mean(dim=1)
    credit = -distance.detach() / members
    for i in range(members):
        for j in range(i):
            pair = (samples[:, i] - samples[:, j]).abs() / (members * (members - 1))
            score = score - pair
            credit[:, i] += pair.detach()
            credit[:, j] += pair.detach()
    return score.masked_fill(~valid, float("nan")), credit.masked_fill(~valid.unsqueeze(1), 0)


def masked_mean(values, valid, dim):
    """仅在明确指定的维度平均已知值；无有效值时抛错，不伪造零损失。

    聚合 score 时 valid 为 [B,...]；聚合 credit 时可传 valid[:,None,...]，
    只约简事件/场坐标，保留 K 维。非均匀权重由调用方显式处理。
    """
    if valid.dtype != torch.bool or valid.device != values.device or valid.ndim != values.ndim:
        raise ValueError("valid 必须为同设备、同维数的 bool 掩码；保留 K 时请显式插入该维度。")
    valid = torch.broadcast_to(valid, values.shape)
    count = valid.sum(dim=dim)
    if (count == 0).any():
        raise ValueError("指定的平均范围没有有效观测。")
    values = torch.where(valid, values, 0)
    if not torch.isfinite(values).all():
        raise ValueError("有效位置包含非有限值。")
    return values.sum(dim=dim) / count
