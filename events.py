"""在固定格网上提取 event_v1 标签；不加载数据或模型，不根据未来选择区域。"""

import numpy as np
from scipy import ndimage


EVENT_NAMES = ("sustained_3obs", "trigger_by_30min", "connected_area_any")


def extract_events(history, futures, history_valid, futures_valid, settings):
    """返回事件值及有效性掩码，形状均为 [B,K,J]。

    history 为 [B,T_in,1,H,W]，futures 为 [B,K,T_out,1,H,W]；
    数值使用原始 VIL 编码对应的尺度，允许浮点生成值。有效性为同形 bool 数组。
    原始观测调用前用 value != 255 构造掩码；生成值不按 255 推断缺测。
    真值也走同一路径：添加 K=1 维度，返回后去掉该维度即可。

    settings 来自 config.yaml 的 events 节。格网按行优先排列，每格内按
    EVENT_NAMES 排列。未知及不适用值为 NaN，掩码为 False，不能当作阴性。
    本算子使用 NumPy；Torch 输入由调用方显式 detach 并转到 CPU。
    """
    history, futures = np.asarray(history), np.asarray(futures)
    history_valid, futures_valid = np.asarray(history_valid), np.asarray(futures_valid)
    if history.ndim != 5 or futures.ndim != 6:
        raise ValueError("需要 history [B,T,1,H,W] 和 futures [B,K,T,1,H,W]。")
    batch, past, channels, height, width = history.shape
    if (batch < 1 or futures.shape[1] < 1 or channels != 1 or futures.shape[3] != 1
            or futures.shape[0] != batch or futures.shape[-2:] != (height, width)):
        raise ValueError("历史和未来的批次、单通道及空间形状必须一致且非空。")
    for values, valid in [(history, history_valid), (futures, futures_valid)]:
        if valid.shape != values.shape or valid.dtype != np.bool_:
            raise ValueError("有效性掩码必须是与输入同形的 bool 数组。")
        if not np.issubdtype(values.dtype, np.number) or np.iscomplexobj(values):
            raise ValueError("雷达值必须是实数数组。")
    for key in ("history_frames", "future_frames", "tile_size", "area_pixels",
                "consecutive_frames", "onset_horizon_frames"):
        if isinstance(settings[key], bool) or not isinstance(settings[key], (int, np.integer)) or settings[key] < 1:
            raise ValueError(f"{key} 必须是正整数。")
    tile, area = settings["tile_size"], settings["area_pixels"]
    duration, horizon = settings["consecutive_frames"], settings["onset_horizon_frames"]
    steps, members = futures.shape[2], futures.shape[1]
    if (settings["version"] != "event_v1" or duration != 3 or horizon != 6
            or settings["frame_interval_minutes"] != 5):
        raise ValueError("当前事件名称固定为 event_v1 的 3 帧和 30 分钟；修改定义需同步版本与算子。")
    if past != settings["history_frames"] or steps != settings["future_frames"]:
        raise ValueError("历史及未来帧数必须与配置一致。")
    if height < tile or width < tile or height % tile or width % tile or area > tile * tile:
        raise ValueError("空间尺寸必须能整除格网大小，面积门槛不能超过格网面积。")
    if duration > steps or horizon > steps or not np.isfinite(settings["threshold"]):
        raise ValueError("持续帧数、起始时限或强度阈值无效。")

    rows, columns = height // tile, width // tile
    hv = history_valid[:, :, 0] & np.isfinite(history[:, :, 0])
    fv = futures_valid[:, :, :, 0] & np.isfinite(futures[:, :, :, 0])
    history_high = (history[:, :, 0] >= settings["threshold"]) & hv
    future_high = (futures[:, :, :, 0] >= settings["threshold"]) & fv
    history_area = history_high.reshape(batch, past, rows, tile, columns, tile).sum(axis=(3, 5))
    future_area = future_high.reshape(batch, members, steps, rows, tile, columns, tile).sum(axis=(4, 6))
    history_known = hv.reshape(batch, past, rows, tile, columns, tile).all(axis=(1, 3, 5))
    future_known = fv.reshape(batch, members, steps, rows, tile, columns, tile).all(axis=(2, 4, 6))
    known = history_known[:, None] & future_known
    active = future_area >= area
    sustained = np.lib.stride_tricks.sliding_window_view(active, duration, axis=2).all(axis=-1).any(axis=2)
    onset_eligible = history_area[:, -1, None] < area
    onset = active[:, :, :horizon].any(axis=2)

    connected = np.zeros_like(known)
    for b, k, r, c in np.argwhere(known):
        # 总面积不足时连通面积一定不足；达到条件即停止，避免重复标记。
        for t in np.flatnonzero(active[b, k, :, r, c]):
            mask = future_high[b, k, t, r*tile:(r+1)*tile, c*tile:(c+1)*tile]
            labels, _ = ndimage.label(mask, structure=np.ones((3, 3)))
            if np.bincount(labels.ravel())[1:].max(initial=0) >= area:
                connected[b, k, r, c] = True
                break
    values = np.stack([sustained, onset, connected], axis=-1)
    valid = np.stack([known, known & onset_eligible, known], axis=-1)
    values = np.where(valid, values, np.nan).astype(np.float32)
    return values.reshape(batch, members, -1), valid.reshape(batch, members, -1)
