"""读取已核对的 SEVIR NPZ 缓存；保持官方划分，不下载数据或调用模型。"""

from datetime import datetime, timedelta, timezone
import hashlib
import io
import json
import math
from pathlib import Path
import re

import numpy as np
import torch
from torch.utils.data import Dataset


PROJECT = Path(__file__).resolve().parent


def normalize_vil(encoded, scale):
    """原始 uint8 观测除以固定 scale；255 缺测填零，并单独返回掩码。"""
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError("归一化尺度必须是有限正数。")
    encoded = torch.as_tensor(encoded)
    if encoded.dtype != torch.uint8:
        raise ValueError("原始 SEVIR 观测必须是 uint8；生成值不使用观测缺测编码。")
    valid = encoded != 255
    return (encoded.float() / scale).masked_fill(~valid, 0), valid


def restore_vil(normalized, valid, scale):
    """还原到 VIL 编码尺度，未知位置为 NaN；不裁剪或取整，保留数值梯度。

    valid 必须显式提供。生成值由生成过程提供有效性，不能因还原值为 255
    而误作缺测；观测则使用 normalize_vil 返回的掩码。
    """
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError("归一化尺度必须是有限正数。")
    if not normalized.is_floating_point() or valid.dtype != torch.bool:
        raise ValueError("还原需要浮点 Torch 张量和 bool 有效性掩码。")
    if valid.shape != normalized.shape or valid.device != normalized.device:
        raise ValueError("有效性掩码必须与数值同形、同设备。")
    return (normalized * scale).masked_fill(~valid, float("nan"))


class SEVIRDataset(Dataset):
    """读取 selection.json/acquisition.json 所固定的一个划分子集。

    每个 samples/<event_id>.npz 只含 vil:[T_in+T_out,H,W]，目前为 part 0。
    settings 来自 config.yaml 的 data 节，event_settings 来自 events 节；
    相对路径按本仓库根目录解释。完整 HDF5、上游 NPY 及其他窗口尚未对接。
    """

    def __init__(self, settings, event_settings):
        self.root = (PROJECT / settings["root"]).resolve()
        self.scale = settings["normalization_scale"]
        if not math.isfinite(self.scale) or self.scale <= 0:
            raise ValueError("归一化尺度必须是有限正数。")
        self.past = event_settings["history_frames"]
        self.steps = self.past + event_settings["future_frames"]
        self.shape = (self.steps, settings["height"], settings["width"])
        selected_bytes = (self.root / "selection.json").read_bytes()
        selection = json.loads(selected_bytes)
        acquisition = json.loads((self.root / "acquisition.json").read_text())
        digest = hashlib.sha256(selected_bytes).hexdigest()
        if acquisition["status"] != "complete" or acquisition["selection_sha256"] != digest:
            raise ValueError("下载记录不完整，或抽样清单与下载记录不一致。")
        if (selection["window_part"] != 0 or selection["history_slice"] != f"0:{self.past}"
                or selection["future_slice"] != f"{self.past}:{self.steps}"):
            raise ValueError("当前只支持与缓存切片一致的 part 0，不能静默切换窗口或帧数。")

        # 按源记录检查划分，防止把同一记录的不同重叠窗口放入不同集合。
        if set(settings["split_files"]) != {"train", "val", "test"} or settings["split"] not in settings["split_files"]:
            raise ValueError("需要明确的 train/val/test 清单和有效的 split。")
        splits = {}
        for name, filename in settings["split_files"].items():
            payload = (PROJECT / filename).read_bytes()
            if name == "train" and hashlib.sha256(payload).hexdigest() != selection["train_list_sha256"]:
                raise ValueError("官方训练清单与抽样时固定的校验值不同。")
            splits[name] = set(payload.decode().splitlines())
        sources = {name: {window.rsplit("-", 1)[0] for window in windows} for name, windows in splits.items()}
        for a, b in [("train", "val"), ("train", "test"), ("val", "test")]:
            if sources[a] & sources[b]:
                raise ValueError(f"{a}/{b} 共享源记录，不能用重叠窗口跨划分。")

        self.records = selection["records"]
        self.metadata = {row["event_id"]: row for row in acquisition["records"]}
        ids = [row["event_id"] for row in self.records]
        keys = [row["cascast_window"].rsplit("-", 1)[0] for row in self.records]
        if (not ids or len(ids) != selection["sample_size"] or len(set(ids)) != len(ids) or len(set(keys)) != len(keys)
                or len(self.metadata) != len(acquisition["records"]) or set(ids) != set(self.metadata)):
            raise ValueError("缓存记录为空、源记录重复，或下载元数据与抽样记录不匹配。")
        self.timings = {}
        for row in self.records:
            event_id = row["event_id"]
            canonical = f'{row["file_name"].replace("/", "-")}-{row["file_index"]}-0.npy'
            meta = self.metadata[event_id]
            if not re.fullmatch(r"[RS][0-9]+", event_id) or row["cascast_window"] != canonical:
                raise ValueError("源事件标识、文件名、索引与窗口名称不一致。")
            if canonical not in splits[settings["split"]]:
                raise ValueError("缓存记录不属于请求的官方划分，不能将训练缓存用作验证或测试集。")
            if (meta["selection_sha256"] != digest or meta["file_index"] != row["file_index"]
                    or not meta["source_url"].endswith("/data/" + row["file_name"])
                    or tuple(meta["shape"]) != self.shape or meta["dtype"] != "uint8"):
                raise ValueError("下载来源、切片形状或编码类型与固定清单不一致。")
            minutes = np.asarray(row["minute_offsets"].split(":"), dtype=float)[:self.steps]
            if (minutes.shape != (self.steps,) or not np.isfinite(minutes).all()
                    or not np.all(np.diff(minutes) == event_settings["frame_interval_minutes"])):
                raise ValueError("实际帧间隔与事件配置不同，不能直接解释为固定分钟数。")
            stamp = datetime.fromisoformat(row["time_utc"])
            stamp = stamp.replace(tzinfo=timezone.utc) if stamp.tzinfo is None else stamp.astimezone(timezone.utc)
            origin = stamp + timedelta(minutes=float(minutes[self.past-1]))
            self.timings[event_id] = (origin.isoformat(), minutes[self.past:] - minutes[self.past-1])

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        row = self.records[index]
        event_id = row["event_id"]
        payload = (self.root / "samples" / (event_id + ".npz")).read_bytes()
        if hashlib.sha256(payload).hexdigest() != self.metadata[event_id]["sha256"]:
            raise ValueError(f"{event_id} 的缓存校验值不符，停止读取。")
        with np.load(io.BytesIO(payload), allow_pickle=False) as sample:
            encoded = sample["vil"]
        if encoded.shape != self.shape or encoded.dtype != np.uint8:
            raise ValueError("缓存实际帧数、空间形状或 uint8 编码不符合约定。")
        values, valid = normalize_vil(encoded, self.scale)
        origin, leads = self.timings[event_id]
        return {"history": values[:self.past, None], "future": values[self.past:, None],
                "history_valid": valid[:self.past, None], "future_valid": valid[self.past:, None],
                "event_id": event_id, "source_file": row["file_name"], "source_index": row["file_index"],
                "window_name": row["cascast_window"], "forecast_origin_utc": origin,
                "future_lead_minutes": torch.tensor(leads, dtype=torch.float32)}
