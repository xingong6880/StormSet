"""共享的真值评分与固定预算评估入口；当前默认只跑训练集单批诊断。"""

import json
import math
from pathlib import Path
import time

import numpy as np
import torch
from torch.utils.data import DataLoader
import yaml

from data import SEVIRDataset, restore_vil
from events import extract_events
from model import CasCast
from rl import ddim_schedule, sample_trajectories
from scores import fair_brier, fair_crps, masked_mean


PROJECT = Path(__file__).resolve().parent


@torch.no_grad()
def score_forecasts(batch, predictions, config):
    """按共享真值掩码评分，返回逐样本指标与已聚合的 [B,K] 奖励系数。

    场 CRPS 使用 /255 尺度，事件先还原编码尺度再提取。只在这里使用真值，
    不筛选成员或由预测有效性缩小评价区域；当前三个事件按已知坐标均匀平均。
    """
    if predictions.ndim != 6 or not torch.isfinite(predictions).all():
        raise ValueError("预测必须是有限 [B,K,T,1,H,W]；不能删除坏成员后评分。")
    weights = config["objective"]
    if (not all(math.isfinite(weights[key]) and weights[key] >= 0 for key in ["event_weight", "field_weight"])
            or weights["event_weight"]+weights["field_weight"] == 0):
        raise ValueError("评分权重必须有限、非负，且至少一项为正。")
    device, scale = predictions.device, config["data"]["normalization_scale"]
    history = restore_vil(batch["history"], batch["history_valid"], scale).cpu().numpy()
    target = batch["future"].to(device)
    target_valid = batch["future_valid"].to(device)
    truth = restore_vil(target, target_valid, scale).cpu().numpy()
    generated = restore_vil(predictions, torch.ones_like(predictions, dtype=torch.bool), scale).cpu().numpy()
    observed, valid = extract_events(history, truth[:, None], batch["history_valid"].cpu().numpy(),
                                     target_valid[:, None].cpu().numpy(), config["events"])
    events, member_valid = extract_events(history, generated, batch["history_valid"].cpu().numpy(),
                                         np.ones_like(generated, dtype=bool), config["events"])
    if (valid & ~member_valid).any():
        raise ValueError("真值可评分位置出现未知预测事件，不能通过缩小掩码来继续。")
    observed = torch.as_tensor(observed[:, 0], device=device)
    valid = torch.as_tensor(valid[:, 0], device=device)
    event_score, event_credit = fair_brier(torch.as_tensor(events, device=device), observed, valid)
    field_score, field_credit = fair_crps(predictions, target, target_valid)
    event_score = masked_mean(event_score, valid, dim=1)
    field_score = masked_mean(field_score, target_valid, dim=(1, 2, 3, 4))
    credit = (weights["event_weight"]*masked_mean(event_credit, valid[:, None], dim=2)
              + weights["field_weight"]*masked_mean(field_credit, target_valid[:, None], dim=(2, 3, 4, 5)))
    metrics = {"event_fair_brier": event_score, "field_fair_crps_normalized": field_score,
               "weighted_score": weights["event_weight"]*event_score+weights["field_weight"]*field_score}
    return metrics, credit


def main():
    config = yaml.safe_load((PROJECT / "config.yaml").read_text())
    runtime, settings = config["run"], config["evaluation"]
    output = PROJECT / settings["output_dir"]
    output.mkdir(parents=True, exist_ok=False)
    device = torch.device(runtime["device"])
    torch.manual_seed(runtime["seed"])
    generator = torch.Generator(device=device).manual_seed(runtime["seed"])
    dataset = SEVIRDataset(config["data"], config["events"])
    model = CasCast(config["model"], device=device)
    if settings["checkpoint"] is not None:
        checkpoint = torch.load(PROJECT / settings["checkpoint"], map_location="cpu", weights_only=True)
        for key in ["model", "sampling", "events", "objective"]:
            if checkpoint["config"][key] != config[key]:
                raise ValueError(f"checkpoint 的 {key} 与评估配置不同。")
        model.denoiser.load_state_dict(checkpoint["denoiser"], strict=True)
    schedule = ddim_schedule(config["sampling"], device)
    rows = []
    for index, batch in enumerate(DataLoader(dataset, batch_size=runtime["batch_size"], shuffle=False)):
        if index >= settings["max_batches"]:
            break
        started = time.perf_counter()
        with torch.no_grad():
            _, condition = model.condition(batch["history"].to(device), batch["history_valid"].to(device),
                                           config["sampling"]["members"], generator)
            trajectory = sample_trajectories(model, condition, schedule, generator)
            predictions = model.decode(trajectory["states"][-1])
            metrics, _ = score_forecasts(batch, predictions, config)
        row = {"event_id": batch["event_id"], "seconds": time.perf_counter()-started,
               **{key: value.cpu().tolist() for key, value in metrics.items()}}
        rows.append(row)
        print(json.dumps(row, ensure_ascii=False), flush=True)
    if not rows:
        raise ValueError("评估没有读取任何批次。")
    result = {"scope": "固定缓存的有限批次诊断，不能当作完整测试集结果", "config": config, "rows": rows}
    (output / "metrics.json").write_text(json.dumps(result, ensure_ascii=False, indent=2)+"\n")


if __name__ == "__main__":
    main()
