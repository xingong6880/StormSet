"""最小 on-policy 训练循环；默认只做一次工程验证更新，不是论文实验预算。"""

import copy
import hashlib
import json
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch.utils.data import default_collate
import yaml

from data import SEVIRDataset
from evaluate import score_forecasts
from model import CasCast
from rl import backward_trajectory, ddim_schedule, sample_trajectories


PROJECT = Path(__file__).resolve().parent


def source_identity(config):
    """保存源码、固定抽样清单和精度身份；冻结组件由配置中的权重校验值标识。"""
    paths = ["data.py", "model.py", "events.py", "scores.py", "rl.py", "train.py", "evaluate.py", "requirements.txt"]
    selection = PROJECT / config["data"]["root"] / "selection.json"
    return {"code_sha256": {name: hashlib.sha256((PROJECT/name).read_bytes()).hexdigest() for name in paths},
            "selection_sha256": hashlib.sha256(selection.read_bytes()).hexdigest(),
            "precision": {"matmul_tf32": torch.backends.cuda.matmul.allow_tf32,
                          "cudnn_tf32": torch.backends.cudnn.allow_tf32,
                          "matmul_precision": torch.get_float32_matmul_precision(),
                          "deterministic_algorithms": torch.are_deterministic_algorithms_enabled()},
            "torch_version": str(torch.__version__)}


def save_checkpoint(path, model, optimizer, generator, step, config, identity, metrics=None):
    """保存模型、优化器、数据游标和全部已使用 RNG；不覆盖已有 checkpoint。"""
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"不覆盖已有 checkpoint：{path}")
    numpy_state = np.random.get_state()
    device = next(model.parameters()).device
    checkpoint = {"denoiser": model.denoiser.state_dict(), "optimizer": optimizer.state_dict(),
                  "step": step, "next_record": step*config["run"]["batch_size"],
                  "config": copy.deepcopy(config), "identity": identity, "metrics": metrics,
                  "rng": {"python": random.getstate(),
                          "numpy": (numpy_state[0], numpy_state[1].tolist(), *numpy_state[2:]),
                          "torch": torch.get_rng_state(), "generator": generator.get_state(),
                          "cuda": torch.cuda.get_rng_state(device) if device.type == "cuda" else None}}
    temporary = path.with_suffix(".tmp")
    torch.save(checkpoint, temporary)
    temporary.replace(path)


def load_checkpoint(path, model, optimizer, generator, config, identity):
    """严格恢复相同实验和执行配置；仅允许延长预算、换输出目录或指定恢复文件。"""
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    def experiment(value):
        value = copy.deepcopy(value)
        for key in ["output_dir", "resume", "max_updates"]:
            value["training"].pop(key, None)
        value.pop("evaluation", None)
        return value
    if checkpoint["identity"] != identity or experiment(checkpoint["config"]) != experiment(config):
        raise ValueError("checkpoint 的源码、数据、精度或实验配置不同，不能静默恢复。")
    if checkpoint["next_record"] != checkpoint["step"]*config["run"]["batch_size"]:
        raise ValueError("checkpoint 的数据游标与更新步数不一致。")
    model.denoiser.load_state_dict(checkpoint["denoiser"], strict=True)
    optimizer.load_state_dict(checkpoint["optimizer"])
    rng = checkpoint["rng"]
    random.setstate(rng["python"])
    np.random.set_state((rng["numpy"][0], np.asarray(rng["numpy"][1], dtype=np.uint32), *rng["numpy"][2:]))
    torch.set_rng_state(rng["torch"])
    generator.set_state(rng["generator"])
    if rng["cuda"] is not None:
        torch.cuda.set_rng_state(rng["cuda"], next(model.parameters()).device)
    return checkpoint["step"]


def run(config):
    """按固定清单顺序循环读取训练记录，每批新轨迹只对应一次 optimizer update。"""
    runtime, settings = config["run"], config["training"]
    if config["data"]["split"] != "train":
        raise ValueError("训练只能使用明确的 train 划分。")
    if runtime["batch_size"] < 1 or settings["max_updates"] < 1 or config["sampling"]["members"] < 2:
        raise ValueError("需要正 batch/更新预算，以及至少两个独立成员。")
    if not 0 < settings["learning_rate"] < float("inf"):
        raise ValueError("learning_rate 必须是有限正数。")
    output = PROJECT / settings["output_dir"]
    output.mkdir(parents=True, exist_ok=False)
    device = torch.device(runtime["device"])
    random.seed(runtime["seed"])
    np.random.seed(runtime["seed"])
    torch.manual_seed(runtime["seed"])
    generator = torch.Generator(device=device).manual_seed(runtime["seed"])
    dataset = SEVIRDataset(config["data"], config["events"])
    model = CasCast(config["model"], device=device)
    parameters = [value for value in model.denoiser.parameters() if value.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=settings["learning_rate"], weight_decay=0., fused=settings["fused_adamw"])
    schedule = ddim_schedule(config["sampling"], device)
    identity = source_identity(config)
    step = 0
    if settings["resume"] is not None:
        step = load_checkpoint(PROJECT/settings["resume"], model, optimizer, generator, config, identity)
    if step >= settings["max_updates"]:
        raise ValueError("恢复点已达到更新预算，没有新的更新可执行。")
    (output/"run.json").write_text(json.dumps({"config": config, "identity": identity}, ensure_ascii=False, indent=2)+"\n")
    save_checkpoint(output/f"step_{step:06d}.pth", model, optimizer, generator, step, config, identity)
    for step in range(step, settings["max_updates"]):
        if device.type == "cuda":
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)
        started = time.perf_counter()
        indices = [(step*runtime["batch_size"]+i) % len(dataset) for i in range(runtime["batch_size"])]
        batch = default_collate([dataset[index] for index in indices])
        optimizer.zero_grad(set_to_none=True)
        _, condition = model.condition(batch["history"].to(device), batch["history_valid"].to(device),
                                       config["sampling"]["members"], generator)
        trajectory = sample_trajectories(model, condition, schedule, generator)
        with torch.no_grad():
            predictions = model.decode(trajectory["states"][-1])
            metrics, credit = score_forecasts(batch, predictions, config)
        rollout_record = {"step_pending": step+1, "event_id": batch["event_id"],
                          **{key: value.mean().item() for key, value in metrics.items()},
                          "credit": credit.cpu().tolist()}
        (output/f"rollout_{step+1:06d}.json").write_text(json.dumps(rollout_record, ensure_ascii=False, indent=2)+"\n")
        diagnostic = backward_trajectory(model, trajectory, credit, settings["step_batch_size"])
        if any(value.grad is None for value in parameters):
            raise ValueError("部分可训练参数没有梯度，停止更新。")
        grad_norm = torch.linalg.vector_norm(torch.stack([torch.linalg.vector_norm(value.grad) for value in parameters]))
        if not torch.isfinite(grad_norm) or grad_norm == 0:
            raise ValueError("完整策略梯度非有限或全部为零，停止更新。")
        optimizer.step()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        row = {"step": step+1, "event_id": batch["event_id"], "seconds_before_checkpoint": time.perf_counter()-started,
               **{key: value.mean().item() for key, value in metrics.items()}, **diagnostic,
               "gradient_norm": grad_norm.item(), "credit": credit.cpu().tolist(),
               "peak_allocated_mib": torch.cuda.max_memory_allocated(device)/1024**2 if device.type == "cuda" else None,
               "peak_reserved_mib": torch.cuda.max_memory_reserved(device)/1024**2 if device.type == "cuda" else None}
        save_checkpoint(output/f"step_{step+1:06d}.pth", model, optimizer, generator, step+1, config, identity, row)
        row["seconds_including_checkpoint"] = time.perf_counter()-started
        with (output/"metrics.jsonl").open("a") as handle:
            handle.write(json.dumps(row, ensure_ascii=False)+"\n")
        print(json.dumps(row, ensure_ascii=False), flush=True)
    return row


if __name__ == "__main__":
    run(yaml.safe_load((PROJECT/"config.yaml").read_text()))
