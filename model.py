"""直接使用固定版本的官方 CasCast 组件；不复制网络，不组织去噪轨迹。

只支持官方 SEVIR 384×384 配置与 float32。条件只来自历史；Earthformer
与 autoencoder 冻结，CasFormer 保留其原有可训练参数。完整采样由 rl.py 对接。
"""

import hashlib
from pathlib import Path
import subprocess
import sys

import torch
from torch import nn
import yaml


PROJECT = Path(__file__).resolve().parent


def _load_weights(module, entry, key):
    """核对已下载权重，严格加载指定组件；不接受缺失或多余参数。"""
    path = PROJECT / entry["path"]
    with path.open("rb") as handle:
        digest = hashlib.file_digest(handle, "sha256").hexdigest()
    if digest != entry["sha256"]:
        raise ValueError(f"{key} 权重校验值不符，停止加载。")
    state = torch.load(path, map_location="cpu", weights_only=True)["model"][key]
    state = {name.removeprefix("module."): value for name, value in state.items()}
    module.load_state_dict(state, strict=True)


class CasCast(nn.Module):
    """官方 Earthformer、autoencoder 与 CasFormer 的最小适配器。

    settings 来自 config.yaml 的 model 节。设备由调用方指定；模型只接受
    已放到同一设备的张量，不静默搬运数据、填补缺测或改变归一化方式。
    """

    def __init__(self, settings, device="cpu"):
        super().__init__()
        root = (PROJECT / settings["upstream_root"]).resolve()
        commit = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
        dirty = subprocess.check_output(["git", "-C", str(root), "status", "--porcelain", "--untracked-files=no"], text=True)
        if commit != settings["upstream_commit"] or dirty:
            raise ValueError("需要固定 commit 且未改动的官方 CasCast checkout。")
        loaded = sys.modules.get("networks")
        if loaded is not None and root / "networks" not in [Path(p).resolve() for p in loaded.__path__]:
            raise ImportError("进程已加载另一个 networks 包，请在独立进程加载 CasCast。")
        if str(root) not in sys.path:
            sys.path.insert(0, str(root))
        from networks.earthformer_xy import EarthFormer_xy
        from networks.autoencoder_kl import autoencoder_kl
        from networks.casformer import CasFormer

        # 架构直接读取固定版本的官方配置，避免在本库复制两套网络超参数。
        config_dir = root / "configs/sevir_used"
        architecture = yaml.safe_load((config_dir / "cascast_diffusion.yaml").read_text())["model"]["params"]["sub_model"]
        predictor_args = yaml.safe_load((config_dir / "EarthFormer.yaml").read_text())["model"]["params"]["sub_model"]["EarthFormer_xy"]
        self.predictor = EarthFormer_xy(**predictor_args)
        self.autoencoder = autoencoder_kl(architecture["autoencoder_kl"])
        self.denoiser = CasFormer(**architecture["casformer"])
        for module, key in [(self.predictor, "EarthFormer_xy"), (self.autoencoder, "autoencoder_kl"),
                            (self.denoiser, "casformer")]:
            _load_weights(module, settings["checkpoints"][key], key)
        self.predictor.requires_grad_(False)
        self.autoencoder.requires_grad_(False)
        # 不对整个 denoiser 调用 requires_grad_(True)，保留固定位置编码的冻结状态。
        self.latent_scale = settings["latent_scale"]
        if not isinstance(self.latent_scale, (float, int)) or not 0 < self.latent_scale < float("inf"):
            raise ValueError("latent_scale 必须是有限正数。")
        self.to(device=device, dtype=torch.float32)
        self.eval()

    def train(self, mode=True):
        """保留参数梯度，但固定各组件的推理行为，避免 rollout/重算随机层不同。"""
        return super().train(False)

    def _check_latents(self, values):
        if (values.ndim != 6 or values.shape[0] == 0 or values.shape[1] == 0
                or tuple(values.shape[2:]) != (12, 4, 48, 48)
                or values.dtype != torch.float32 or values.device != next(self.parameters()).device
                or not torch.isfinite(values).all()):
            raise ValueError("latent 必须为模型同设备的有限 float32 [B,K,12,4,48,48]。")

    @torch.no_grad()
    def condition(self, history, history_valid, members, generator=None):
        """只用过去预测粗场，再独立抽取 K 份条件 latent；返回 (coarse, condition)。

        history/掩码为 [B,13,1,384,384]，数值使用 data.py 的 /255 尺度。
        coarse 为 [B,12,1,384,384]，condition 为 [B,K,12,4,48,48]。
        不将同一个随机后验样本复制给所有成员；当前不支持缺测历史输入。
        """
        if (history.ndim != 5 or history.shape[0] == 0 or tuple(history.shape[1:]) != (13, 1, 384, 384)
                or history.dtype != torch.float32 or history.device != next(self.parameters()).device
                or not torch.isfinite(history).all() or (history < 0).any() or (history > 1).any()):
            raise ValueError("历史必须为模型同设备的有限 float32 [B,13,1,384,384]，范围为 [0,1]。")
        if (history_valid.shape != history.shape or history_valid.dtype != torch.bool
                or history_valid.device != history.device or not history_valid.all()):
            raise ValueError("预训练模型尚无缺测处理协议；历史掩码必须同形、同设备且全部有效。")
        if not isinstance(members, int) or members < 1:
            raise ValueError("成员数必须为正整数。")
        coarse = self.predictor(history)
        posterior = self.autoencoder.net.encode(coarse.flatten(0, 1))
        batch = history.shape[0]
        mean = posterior.mean.reshape(batch, 1, 12, 4, 48, 48)
        std = posterior.std.reshape(batch, 1, 12, 4, 48, 48)
        noise = torch.randn((batch, members, 12, 4, 48, 48), device=mean.device, dtype=mean.dtype, generator=generator)
        condition = (mean + std * noise) * self.latent_scale
        self._check_latents(condition)
        return coarse, condition

    def predict_noise(self, latents, timesteps, condition):
        """CasFormer epsilon 预测，保留参数/输入梯度；RL 调用方先 detach 记录状态。

        latent 与 condition 为 [B,K,12,4,48,48]，timesteps 为 [B,K] 整数，
        范围 0–999。成员与 batch 合并后调用官方网络，不在成员间做 attention。
        """
        self._check_latents(latents)
        self._check_latents(condition)
        if condition.shape != latents.shape:
            raise ValueError("条件 latent 必须与采样状态同形，不能静默广播成员。")
        if (timesteps.shape != latents.shape[:2] or timesteps.device != latents.device
                or timesteps.dtype not in (torch.int32, torch.int64)
                or (timesteps < 0).any() or (timesteps >= 1000).any()):
            raise ValueError("去噪时间步必须是同设备的 [B,K] 整数，范围为 0–999。")
        result = self.denoiser(x=latents.flatten(0, 1), timesteps=timesteps.flatten(),
                               cond=condition.detach().flatten(0, 1))
        return result.reshape_as(latents)

    def decode(self, latents):
        """解码为 /255 尺度的 [B,K,12,1,384,384]，不裁剪；保留 latent 梯度。

        RL 评分时由调用方使用 no_grad；可微评分对照可通过冻结 decoder 反传。
        默认普通批量前向，不预设逐帧 slicing、CPU offload 或激活重算。
        """
        self._check_latents(latents)
        batch, members = latents.shape[:2]
        pixels = self.autoencoder.net.decode((latents / self.latent_scale).flatten(0, 2))
        return pixels.reshape(batch, members, 12, 1, 384, 384)
