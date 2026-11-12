# StormSet

**Event-Aligned Diffusion Forecasting via Ensemble Reinforcement Learning**

StormSet 的最小科研代码库。研究目标是：让扩散模型生成的多个未来雷达序列，正确表达强回波出现时间、持续活动和连通面积等事件的概率。

**当前状态：7 个核心模块均已实现，已跑通“固定数据缓存 → 20 步随机去噪 → 事件/场评分 → RL 更新 → checkpoint”，并完成真实 checkpoint 恢复后的下一次更新。64 项测试全部通过。`config.yaml` 默认只执行一次工程验证更新和单批诊断评估；尚无完整 baseline 复现、正式训练或测试集效果结论。**

下一批实验的交接说明见 [EXPERIMENT_TASKS.md](EXPERIMENT_TASKS.md)：先补齐验证集与评估输出，再做冻结模型、场评分 RL、StormSet 的单 seed 小规模对照；任务书包含固定协议、预算、验收及停止规则，尚未执行。

论文工程保存在 Mac 的 `/Users/mianliu/Github/StormSet-CVPR2027`；本代码库保存在 `/Users/mianliu/Github/StormSet`。论文中的数学验证和真实数据实验计划应分别看待，本仓库当前没有雷达训练结果。

## 1. 研究问题与最小方案

扩散模型能够生成多个可能的未来，但逐像素预测较好，并不意味着跨时间、跨空间的事件概率正确。例如，20 个预测成员中有 6 个出现持续强回波，对应约 30% 的事件概率；我们需要验证这种概率是否可靠，以及模型能否区分不同条件下的风险。

第一阶段聚焦 **SEVIR + CasCast**：冻结预训练表示与确定性预测分支，独立采样多个未来，对固定事件和像素分布进行集合评分，再更新扩散去噪器。事件评分使用已有的 fair Brier，像素分布评分使用已有的 fair CRPS。对于精确阈值、连通区域和持续时间判断，通过随机去噪轨迹的策略梯度优化。

已有的 fair score 和 diffusion RL 是方法基础。项目需要用实验检验：直接优化精确的多帧事件概率，是否比可微评分、软事件近似或简单校准更有效。

PreDiff 的小规模配置与 MeteoNet 的第二数据集验证在主流程成立后再接入；目前保持一个 backbone 和一个实验配置。

## 2. 每个文件负责什么

文件全部放在仓库根目录，便于直接阅读和修改。

| 文件 | 职责 | 职责边界 |
| --- | --- | --- |
| `README.md` | 说明研究目标、当前状态、全部文件职责、数据流和实施顺序。 | 随实际实现与实验结果更新。 |
| `EXPERIMENT_TASKS.md` | 第一批实验交接任务：前置代码修改、数据/事件协议、对照、预算、评价与交付。 | 是待执行的 pilot 计划，不代表已有训练或测试结果。 |
| `AGENTS.md` | 保存“科研最小库”的开发约定，供后续开发者和 AI 助手遵循。 | 只规定本仓库的开发方式。 |
| `config.yaml` | 集中保存数据、事件、源码/权重身份、采样、评分权重及一次更新的工程配置。 | 当前数值不是最终研究实验方案。 |
| `data.py` | 已实现固定 NPZ 缓存的来源、划分、校验值、帧间隔检查，以及历史/未来切片和归一化还原。 | 当前仅支持已下载的 part 0 缓存；不下载数据，不加载模型，不计算奖励。 |
| `model.py` | 已实现官方组件的严格加载、历史到条件 latent、CasFormer 噪声预测及解码；冻结 autoencoder 和 Earthformer。 | 当前仅支持无缺测的 SEVIR 384×384、float32；不组织完整去噪轨迹或训练循环。 |
| `events.py` | 已实现固定格网的持续活动、首次越阈、8 邻接连通面积及有效性掩码。 | 使用 NumPy，在 CPU 提取不可微标签；真值与预测使用同一算子。 |
| `scores.py` | 已实现 Torch fair Brier、fair CRPS、解析成员奖励贡献和掩码平均。 | CRPS 保留场数值梯度，成员贡献已 detach；不组织 rollout 或优化器。 |
| `rl.py` | 已实现随机 DDIM、显式正方差末步、完整轨迹记录与重算、成员策略梯度和按时间块累计梯度。 | 不包含 KL、PPO、优化器或 checkpoint；不重复使用已更新策略的旧轨迹。 |
| `train.py` | 直接组织新 rollout、评分、一次 AdamW 更新、日志、完整训练状态保存和严格恢复。 | 固定清单顺序读训练缓存；不构造通用 Trainer 或复杂 CLI。 |
| `evaluate.py` | 提供训练/评估共享的 `score_forecasts`，按相同采样和评分协议运行冻结或 checkpoint 模型。 | 默认仅单批训练数据诊断；真值不进入生成，也不挑选预测成员。 |
| `.gitignore` | 忽略数据、权重、运行输出、缓存和本地环境。 | 小型源码、配置与文档正常版本管理。 |
| `requirements.txt` | 记录当前模块及官方网络前向/反向实际通过验证的依赖版本。 | 不代表支持上游的整套 trainer、分布式设施或 sampler fork。 |
| `test_events_scores.py` | 用手算案例、有限分布精确枚举和本地训练样本验证具体研究风险。 | 使用标准库 `unittest`，不新增测试框架或 CI。 |
| `test_data.py` | 检查错位切片、跨划分重叠、缓存损坏、时间与缺测处理，并验证数据到评分的手算接口。 | 临时缓存仅为测试输入；不构造假模型或训练流程。 |
| `test_rl.py` | 核对高斯密度、动作/奖励停止梯度、成员缩放和两步轨迹的解析期望梯度。 | 使用有限枚举与确定性积分，不运行雷达训练。 |
| `test_model.py` | 检查权重校验、严格加载、冻结状态与模型输入边界。 | 有本地官方源码与权重时执行 3 项真实组件检查；不自动占用 GPU。 |
| `test_training.py` | 核对共享事件/场奖励、缺测掩码、checkpoint 的模型/优化器/数据游标及 RNG 恢复。 | 使用手算场和小型张量验证具体正确性，不冒充雷达训练结果。 |

## 3. 数据流与张量约定

`data.py` 提供历史与未来；`model.py` 提供预训练组件；`rl.py` 生成独立的未来集合；`events.py` 提取事件；`scores.py` 计算集合评分和成员贡献；`train.py` 完成更新，`evaluate.py` 在固定协议下报告结果。

以下是计划采用的统一外部张量约定，具体模型内部格式在对接上游时处理：

| 对象 | 形状 | 含义 |
| --- | --- | --- |
| 历史雷达 | `[B, T_in, 1, H, W]` | 每个样本的一段过去观测 |
| 真实未来 | `[B, T_out, 1, H, W]` | 每个历史对应的观测未来 |
| 预测集合 | `[B, K, T_out, 1, H, W]` | 每个历史独立采样的 K 个未来 |
| 预测事件 | `[B, K, J]` | 每个成员在 J 个固定二元事件上的结果 |
| 真实事件 | `[B, J]` | 真实未来的相同事件结果 |

`B` 为批大小，`K` 为集合成员数，`J` 为事件数，`H/W` 为空间尺寸。onset 可通过固定时间阈值展开为二元事件。事件阈值使用数据集明确声明的单位，不能直接将归一化数值当作物理量。

当前 `data.py` 返回 `/255` 后的 float32 历史和未来，以及独立的 bool 掩码；送入事件算子前必须调用 `restore_vil` 还原编码尺度。模型条件只使用历史和历史掩码，真实未来用于监督或评价。

后续实现必须保留以下方法条件：

- 给定历史后，各预测成员独立采样；同一成员内部保留时空依赖。
- 使用可计算转移密度、方差为正的随机去噪过程；基线与训练后模型采用相同采样设置。
- 策略梯度重算时，采样状态、动作和评分系数应 detach；每批新 rollout 对应一次更新。
- 事件定义、数据划分和测试协议提前固定，checkpoint 与校准器仅使用验证集选择。

## 4. 第一轮实验只回答一个问题

保持 backbone、数据划分、训练参数子集及采样条件一致，先比较：

| 对照 | 要回答的问题 |
| --- | --- |
| 原始冻结模型 | 现有模型是否确实存在目标事件概率误差？ |
| 可微概率评分后训练 | 直接改变概率损失是否已经足够？ |
| 软事件近似后训练 | 精确事件的 RL 更新是否比近似梯度更有价值？ |
| StormSet | 优化精确事件的集合评分是否带来可重复收益？ |

在这四组成立后，再加入匹配训练预算的继续去噪训练、事件概率校准，以及其他必要消融。评价同时关注事件概率质量、场预测质量与计算成本，不预设 StormSet 一定更好。

## 5. 后续实现顺序

1. **数据与冻结模型**：实现 `data.py`、`model.py` 和最小的 `evaluate.py`，先确认原始模型能复现合理输出，并检查数据划分与归一化。
2. **事件与评分**：实现 `events.py`、`scores.py`，用小型可手算例子核对事件边界、有限集合修正和成员贡献。
3. **最小 RL 闭环**：实现 `rl.py`、`train.py`，先让少量数据上的一次更新具有可检查的梯度、评分和 checkpoint。
4. **公平对照**：跑第一轮四组实验，记录配置、随机种子、代码 commit、权重来源、数据清单和实际计算成本。
5. **扩展证据**：根据真实结果决定是否开展多个训练种子、未奖励事件与 MeteoNet 验证。

当前最小代码链路已完成；数据审计、模块与运行用法见第 7–12 节。后续重点是正式数据/验证协议和公平对照，完整 baseline 复现未执行。

## 6. 仓库保持最小

当前可运行数据、模型组件、事件、评分和 RL 计算的验证：

```bash
python -m pip install -r requirements.txt
python -m unittest -v
```

依赖版本来自本机 Python 3.11、Torch 2.12.1+cu130 的 CPU 测试及 RTX 3090 实测，未验证上游整套 trainer。测试共 64 项：60 项不需要下载数据或权重，1 项需要本地 100 条训练数据，3 项需要官方源码与预训练权重；缺少对应文件时明确跳过。`unittest` 不自动运行 GPU 检查。实际训练、恢复和评估运行见第 12 节。

实验参数集中在一个 `config.yaml`；依赖在实际对接并验证后再记录。先写清晰的函数和直接训练循环，新增文件应由具体实验需要驱动。

本地数据放 `data/`，预训练权重放 `checkpoints/`，运行结果放 `outputs/`，均按需创建并由 Git 忽略。数据、权重、凭证和大规模预测数组不进入代码仓库。

本仓库不预建复杂 CLI、多层配置继承、注册器、训练框架、CI、Docker 或部署设施。相关约定已写入 `AGENTS.md`。

## 7. 已完成的事件数据审计（2026-11-11）

本轮只检查真实数据和事件标签，没有运行模型。服务器工作目录为 `/data1/mianliu/StormSet`；本地数据及产物由 Git 忽略，未自动同步到 Mac。

从官方 CasCast 训练清单按源文件名和 `file_index` 去重得到 11,906 个记录，用 NumPy `default_rng(0)` 无放回抽取 100 个，固定取每个记录的第一段 25 帧窗口。抽样清单在读取像素前保存，所选源记录不与验证或测试清单重叠；本次未读取验证或测试图像。

每个 384×384 窗口划分为 36 个固定的 64×64 格网，共 3,600 个格网窗口。本轮三个强度阈值（133、181、219）与三个面积门槛（16、64、256）均保留完整统计，没有根据结果替换记录或阈值。

初始实现采用 `event_v1`：VIL 编码值 ≥181、面积 ≥64 像素，保留以下三个事件。区域按左上角开始的行优先顺序固定，不能根据未来选择。

| 事件 | 定义 | 本轮阳性计数 |
| --- | --- | --- |
| 持续活动（主要事件） | 未来 12 帧内，至少连续 3 帧的高 VIL 总面积达到门槛。 | 266 / 3,600，涉及 45 个源记录 |
| 未来 30 分钟内首次越阈 | 最后历史帧未达到门槛时，未来第 1–6 帧至少一次达到门槛。 | 105 / 3,402 个适用窗口 |
| 连通面积 | 未来任意一帧，固定格网内的最大 8 邻接连通区域达到门槛。 | 289 / 3,600，涉及 45 个源记录 |

这些是训练数据的描述性计数，不能当作模型概率或独立风暴样本量。连续 3 帧观测的首尾间隔为 10 分钟；区域持续活动不等于同一个风暴的身份跟踪。105 个首次越阈窗口中，23 个在更早的历史帧已活跃，因此该事件不表示风暴物理生成。

原始观测的 `255` 表示缺测，应与数值分开保存有效性掩码。`event_v1` 对含缺测的 25 帧格网窗口保留“未知”，不能标成阴性；生成数值不能仅因等于 `255` 被误判为观测缺测。本轮 100 个样本没有缺测，真实缺测案例仍待验证。连通区域在格网边界内计算，边界截断属于定义的一部分。

完整文件位于本地 `data/sevir_event100/` 与 `outputs/event100/`：

- `data/sevir_event100/selection.json`：固定抽样清单、种子和上游清单校验值。
- `data/sevir_event100/acquisition.json`：100 个样本的来源、字节范围和 SHA-256。
- `outputs/event100/event_protocol_v1.json`：数据审计时保存的事件定义、有效性与标签顺序，现已由 `events.py` 实现。
- `outputs/event100/threshold_summary.csv`、`events_by_source.csv`：全部候选条件和按源记录汇总的计数。
- `outputs/event100/event_counts.png`：计数图。

该协议已通过 `SEVIRDataset → DataLoader → restore_vil → extract_events` 逐条复现全部 100 个记录的标签及掩码。完整模型训练链路只在固定清单的前两条记录上作工程检查，不将这些训练样本诊断作为测试集结果。

## 8. 事件与评分模块用法

`extract_events(history, futures, history_valid, futures_valid, settings)` 的 `settings` 来自 `config.yaml` 的 `events` 节。历史形状为 `[B,13,1,H,W]`，未来为 `[B,K,12,1,H,W]`；输入使用原始 VIL 编码尺度，并显式传入同形的 bool 有效性掩码。原始观测的掩码可由 `value != 255` 构造，生成值的有效性不能按观测缺测编码推断。

返回 `(values, valid)`，形状均为 `[B,K,J]`。384×384 输入有 36 个格网，每格依次排列 `sustained_3obs`、`trigger_by_30min`、`connected_area_any`，所以 `J=108`。真值添加 `K=1` 维度后使用同一路径，返回后去掉该维度。未知或不适用项为 `NaN`，掩码为 `False`；Torch 输入由调用方显式 detach 后转到 CPU。

评分的手算示例可直接运行：

```python
import torch
from scores import fair_brier

events = torch.tensor([[[0.], [0.], [1.], [1.]]])  # [B=1,K=4,J=1]
observed = torch.tensor([[1.]])
score, credit = fair_brier(events, observed)
print(score)   # 1/6；只用于核对公式，没有运行雷达模型
print(credit)  # 对应成员的奖励系数，不是评价指标
```

`fair_brier(samples, observed, valid=None)` 和 `fair_crps(samples, observed, valid=None)` 都返回 `(score, credit)`。输入为 Torch 张量：成员值 `[B,K,...]`、观测及共享掩码 `[B,...]`，要求 `K>=2`。二元 Brier 仅接受 0/1 事件；CRPS 接受实数场，保持对场数值的梯度。实现保留有限样本修正，采用 [Ferro 的 fair score](https://doi.org/10.1002/qj.2270)；[scoringrules 官方说明](https://scoringrules.readthedocs.io/en/latest/crps_estimators.html)给出了相应 CRPS 估计式。

`score` 形状为 `[B,...]`，越小越好；无效位置为 `NaN`。使用 `masked_mean(score, valid, dim=...)` 明确约简坐标，不能用补零后的普通均值。没有有效值时抛错，可评分位置出现非有限预测时也抛错，不通过删除成员或缩小集合评分。

`credit` 形状为 `[B,K,...]`，是最大化 `reward=-score` 的成员系数，已 detach。先用与评分相同的坐标权重聚合，保留 K 维；均匀掩码平均可调用 `masked_mean(credit, valid[:,None,...], dim=...)`。策略损失由 `ensemble_policy_loss(log_probability, credit)` 实现，成员求和后按批次平均。系数已经包含 `1/K`，不能再次对 K 平均；也不能把其他成员 credit 的均值直接当作独立基线。当前共享评分采用事件已知坐标的均匀平均及 /255 尺度的场 CRPS，二者权重均为 1，仅用于工程检查。

**论文同步事项：** 当前 Mac 论文草稿用连通面积定义持续活动及起始，而数据审计后固定的 `event_v1` 用总面积定义这两项，并将连通面积单列。当前代码遵循 `event_v1`；论文方法和实验协议需要随后同步，不能将两种定义的结果混合报告。

## 9. 固定缓存的数据接口

`SEVIRDataset(config["data"], config["events"])` 使用已保存的 `selection.json`、`acquisition.json` 和 `samples/<event_id>.npz`。当前缓存包含 100 个固定训练源记录，每个文件为原始 `uint8 [25,384,384]`，只支持 part 0。完整 SEVIR HDF5 和上游 NPY 尚未对接。

路径来自 `config.yaml`；相对路径按仓库根目录解释。读取时核对官方清单、源记录归属、抽样清单与下载元数据的关联，以及每份样本的 SHA-256。三个划分按源文件名和索引检查交集，不能将同一源记录的重叠窗口跨划分；这不构成完整的天气过程独立性审计。当前只有训练缓存，将其配置成 `val` 或 `test` 会明确报错。

每个样本返回以下字段，标准 Torch `DataLoader` 添加批次维度：

| 字段 | 单样本内容 |
| --- | --- |
| `history` / `future` | float32 `[13,1,H,W]` / `[12,1,H,W]`，原始观测除以固定的 255，缺测填零 |
| `history_valid` / `future_valid` | 同形 bool 掩码，区分真实零值和缺测填零 |
| `event_id`、`source_file`、`source_index`、`window_name` | 来源与固定窗口身份 |
| `forecast_origin_utc` | 根据目录时间和逐帧偏移计算的最后历史帧 UTC 时间 |
| `future_lead_minutes` | 相对最后历史帧的未来时效，本配置为 5、10、…、60 分钟 |

帧间隔由记录的 `minute_offsets` 核对，不用目录时间直接充当预测起点。固定 `/255` 不拟合训练或测试统计；`restore_vil(values, valid, scale)` 将有效值还原为编码尺度，未知值为 `NaN`，不裁剪或取整，并保留生成值的梯度。

在已下载缓存的服务器上可直接检查一个批次：

```python
import yaml
from torch.utils.data import DataLoader
from data import SEVIRDataset, restore_vil

with open("config.yaml") as handle:
    config = yaml.safe_load(handle)
dataset = SEVIRDataset(config["data"], config["events"])
batch = next(iter(DataLoader(dataset, batch_size=2, shuffle=False)))
history = restore_vil(batch["history"], batch["history_valid"], dataset.scale)
print(history.shape)  # [2,13,1,384,384]
print(batch["event_id"])
```

## 10. RL 转移与损失计算

`rl.py` 用直接函数实现转移密度与随机轨迹：

| 函数 | 输入与返回 |
| --- | --- |
| `sample_gaussian_transition(mean, std, generator=None)` | `mean:[B,K,...]`，固定标准差 `std` 严格为正；返回同形的独立高斯噪声动作，不建梯度图、不裁剪。 |
| `gaussian_transition_log_prob(action, mean, std)` | 在记录动作处重算联合对数概率，返回 `[B,K]`；动作停止梯度，均值保留参数梯度。 |
| `ensemble_policy_loss(trajectory_log_prob, credit)` | 两者均为 `[B,K]`、`K>=2`；返回标量 `-(credit.detach()*trajectory_log_prob).sum(dim=1).mean()`。 |
| `ddim_schedule(settings, device)` / `ddim_mean(state, epsilon, schedule, index)` | 构造固定线性 beta、leading 时间格和正方差末步，计算状态条件下的高斯均值。 |
| `sample_trajectories(model, condition, schedule, generator)` | 返回 `[S+1,B,K,...]` 状态、各步记录概率、固定条件与策略版本；不接收未来真值。 |
| `backward_trajectory(model, trajectory, credit, step_batch_size)` | 对记录状态/动作停止梯度，重算全部 S 步并累计策略梯度；不调用 optimizer.step。 |

`std` 表示标准差，不是方差；可以是标量，或与 `mean` 同维数并通过单例维广播的张量。当前不支持学习标准差。密度计算至少使用 float32，float64 保持原精度；半精度均值的采样动作也使用 float32。

联合密度在全部动作坐标求和，各随机转移的 log probability 再相加为轨迹对数概率；两者均不取平均。本机用 Torch `Independent(Normal(...), ...)` 核对了数值。这一点区别于 [DDPO 作者实现](https://github.com/kvablack/ddpo-pytorch/blob/main/ddpo_pytorch/diffusers_patch/ddim_with_logprob.py)中对动作坐标取均值的实现约定，本库使用完整联合密度，接入模型时需关注动作维数带来的梯度尺度。

完整轨迹遵循以下条件：

- 给定历史后独立采样 K 个成员；随机数发生器连续使用，不在每个成员处重置相同种子，均值计算也不能混合成员。
- rollout 不建图；更新时先 detach 每步记录状态，再用当前去噪器重算均值。仅在密度函数内 detach 动作，不能替代输入状态的 detach。
- 采样与重算使用同一参数版本、条件、转移标准差和运算约定。每批新轨迹只做一次 on-policy 更新，不能复用旧轨迹做多轮此类更新。
- 所计入的高斯转移标准差严格为正；不能把确定性末步的标准差夹到 epsilon，或直接省略仍依赖可训练参数的确定性末步后声称保持原目标。当前显式定义新的随机末步，冻结对照也须采用同一分布。
- 先按共享真值掩码聚合评分的成员系数，再计算策略损失。不按成员丢弃无效预测，不对 K 再次平均，不加入未经验证的奖励标准化。

RL 测试共 18 项，包括联合密度与梯度、随机数复现、半精度与非法输入、评分掩码、越阈事件期望梯度、两步 Gaussian 链解析梯度，以及完整轨迹的分块/整批梯度、策略版本和重复使用检查。解析测试不代表模型效果。当前没有加入 KL、PPO、奖励标准化或梯度裁剪。

## 11. 官方 CasCast 组件接入

`model.py` 使用 [OpenEarthLab/CasCast](https://github.com/OpenEarthLab/CasCast/tree/487a68b5ade9aa829fe7df2e8f6746b4d9acc233) 的原始网络定义，版本固定为 `487a68b5ade9aa829fe7df2e8f6746b4d9acc233`。官方 checkout 和权重保存在 Git 忽略的目录，不复制进本库源码；加载时检查 checkout 版本及已跟踪文件是否改动，并核对每个权重的 SHA-256，使用 `strict=True` 加载对应组件。

新机器先安装本仓库依赖，再准备官方源码：

```bash
git clone https://github.com/OpenEarthLab/CasCast.git data/reference/cascast/upstream
git -C data/reference/cascast/upstream checkout 487a68b5ade9aa829fe7df2e8f6746b4d9acc233
```

从官方 README 的链接下载以下权重，保存为配置中的文件名；三个原始下载文件合计约 1.82 GiB。文件路径和实际核对的 SHA-256 已写入 `config.yaml`，本机另保存 `checkpoints/cascast/manifest.json` 记录来源及下载时间。

| 组件 | 官方权重 | 本地文件 |
| --- | --- | --- |
| Earthformer | [earthformer_ckpt.pth](https://drive.google.com/file/d/1ZrUgJMbx4hRE4uA0Y43m4Q1OPwA_R_Ow/view) | `checkpoints/cascast/earthformer.pth` |
| Autoencoder | [autoencoder_ckpt.pth](https://drive.google.com/file/d/18HaCuLpiX15Os5XCOqoenj2r__2cRHm7/view) | `checkpoints/cascast/autoencoder.pth` |
| CasFormer | [casformer_ckpt.pth](https://drive.google.com/file/d/1CoUsHmBcTiv1vqj8I88c16RIlNOCSuZi/view) | `checkpoints/cascast/casformer.pth` |

最小接口为 `CasCast(config["model"], device=...)`：

| 调用 | 输入与结果 |
| --- | --- |
| `condition(history, history_valid, members, generator=None)` | 输入 `/255` 的 `[B,13,1,384,384]` 历史及同形 bool 掩码，返回粗预测 `[B,12,1,384,384]` 与条件 latent `[B,K,12,4,48,48]`。含缺测历史会报错。 |
| `predict_noise(latents, timesteps, condition)` | 状态/条件为 `[B,K,12,4,48,48]`，时间步为 `[B,K]`、0–999 的整数；返回同形 epsilon，保留 denoiser 和状态梯度，条件停止梯度。 |
| `decode(latents)` | 返回 `/255` 尺度的 `[B,K,12,1,384,384]`，不裁剪或取整；评分前仍需用 `restore_vil` 还原编码尺度。 |

当前只接受模型同设备的 float32 输入。Earthformer 与 autoencoder 冻结，CasFormer 保留 309,246,016 个可训练参数；固定位置编码继续冻结。调用 `.train()` 仍保留各网络的 eval 行为，参数梯度可正常计算。VAE 解码在代码中保留输入梯度，但完整窗口的 decoder 反向尚未验证；RL 事件评分路径使用 `no_grad`。

条件生成沿用官方的 VAE 后验采样，使用 SEVIR 推理代码中的 `latent_scale=0.6786020398139954`，不使用 Stable Diffusion 的通用缩放值。粗预测及后验参数只计算一次，每个成员独立抽取条件 latent；不能把同一个随机后验样本复制到 K 个成员后声称给定历史的成员独立。后续每条轨迹需保存自己的条件，重算密度时直接复用，不能重新抽取。条件生成不接收未来真值。

本机使用固定训练样本 `R17110515178083`、`B=1,K=2`，在 GPU 0（RTX 3090，UUID `GPU-ea17a175-dade-87b9-4163-d828a81dc8e4`）完成以下实际检查：

- 适配器的粗预测、条件后验采样、epsilon 预测和解码结果与直接调用官方网络逐值一致；交换成员后输出相应交换。
- 一次完整 CasFormer 前向/反向中，256 个可训练张量的梯度均有限；冻结组件、条件与固定位置编码没有梯度。该诊断使用 epsilon 平方均值，不是 StormSet 训练目标，optimizer updates 为 0。
- 普通批量编码 12 帧约 1.20 秒，两个成员的 24 帧解码约 3.47 秒，CasFormer 两成员前向/反向约 0.70 秒。最大 tensor 分配峰值约 14.23 GiB，缓存池峰值约 17.92 GiB，未使用 slicing、CPU offload 或激活重算。这是共享 GPU 上的接入耗时，不是吞吐基准或完整训练的最优 batch 结论。

完整记录保存在 `outputs/cascast_integration/forward_validation.json`、`forward.log` 和 `verify_forward.py`，包括源码/配置校验值、输入身份、运行前 GPU 占用、精度设置、逐阶段峰值与耗时。

**采样差异：** 官方默认推理使用 20 步 DDIM，`eta=0`；当前工程策略使用 `eta=1`，并显式给末步加入标准差 `0.01` 的高斯输出转移。20 步均值和前 19 步噪声系数已与固定版本的官方 scheduler 逐位核对；末步噪声属于本库的新策略定义，改变了预测分布。未来冻结对照和训练后模型必须使用相同设置，不能直接把文献中的原采样指标当作本策略的 baseline。

## 12. 最小训练、恢复与诊断评估

当前工程配置为 `seed=0,B=1,K=2`、13 帧历史预测 12 帧、384×384、float32、20 步随机采样、AdamW `lr=1e-7,weight_decay=0`。事件与场评分权重均为 1。只训练官方 CasFormer 的原有可训练参数，不缩小网络；默认一次更新，这些超参数尚未作为正式研究方案验证。

已具备前述数据、源码和权重时，修改 `config.yaml` 后直接运行，无 CLI 参数：

```bash
# 本次实测使用服务器第二张 3090；进程内 run.device=cuda:0。
export CUDA_VISIBLE_DEVICES=GPU-ce4ccdd7-d780-c027-d4ef-2699fafcb88c
python train.py
python evaluate.py
```

运行前按当前机器选择可用 GPU。本次两个默认入口已实际执行；重复运行需在配置中指定新的 `output_dir`，程序拒绝覆盖旧目录或 checkpoint。默认评估加载 `outputs/train_smoke/step_000001.pth`；评估冻结模型时将 `evaluation.checkpoint` 改为 `null`，并指定另一个输出目录。当前缓存只有训练划分，这些单批输出均为训练诊断，不是泛化结果。

训练按固定抽样清单顺序读取，超过清单长度时循环；没有随机 shuffle 或数据增广。每批只用历史生成条件和轨迹，随后才使用真值评分。事件在原始编码尺度提取，场 CRPS 在 /255 尺度计算，按共享真值掩码聚合成员奖励。全部去噪时间步都进入更新，分块仅改变执行方式，不减少采样预算或把时间步当作 ensemble 成员。

checkpoint 保存 denoiser、Adam 状态、步数/下一条记录位置、完整配置、源码/数据身份及 Python、NumPy、Torch CPU/CUDA、专用采样 generator 的 RNG。冻结组件依配置和 SHA-256 重载，不重复保存权重。恢复时设置 `training.resume`、新的输出目录和更大的 `max_updates`；除预算/输出路径/恢复路径外，源码、数据、精度和训练配置必须匹配。修改时间分块也会被拒绝，不能未经核对静默改变正在恢复的执行方式。

RTX 3090 的实际测试保持同一初始权重、优化器、RNG、样本和逻辑 batch，结果如下。耗时不含 checkpoint 写盘，仅为各设置一次更新的观测，不能证明长期吞吐最优。

| 重算时间步块大小 | 首次更新 | 恢复后第二次更新 | 实际选择 |
| --- | --- | --- | --- |
| 20 / 5 | 显存不足 | 未执行 | 不采用 |
| 4 | 约 8.60 秒，分配峰值约 20.40 GiB | Adam 状态已分配时显存不足 | 不作为默认 |
| 3 | 约 8.67 秒 | 约 8.66 秒，分配峰值约 18.34 GiB | 当前默认 |
| 2 | 约 8.77 秒 | 约 8.74 秒，分配峰值约 16.54 GiB | 可运行，本次略慢 |

因此，本配置的连续更新分配峰值约 19.7 GB，缓存池峰值约 21.7 GB，适用于实测这张 24 GB 3090；修改 B、K、精度或分辨率后需重新测量。未使用 activation checkpointing、CPU offload 或 VAE slicing。各设置均保存更新前后的状态；虽然手算测试中的分块梯度一致，真实 float32 计算在分块 4 与 3/2 之间产生了一步后参数最大约 `2.09e-7` 的差异，因此没有声称训练轨迹逐位相同。

真实数据检查使用清单前两条记录：第一条事件奖励为零，更新来自场 CRPS；从 checkpoint 恢复后，下一条记录产生了非零事件奖励并完成第二次更新。独立重放真实 checkpoint 时，全部已使用 RNG、下一条记录的评分及奖励系数逐位一致；单独使用事件奖励反传时，256 个可训练张量的梯度均有限且非零。两条记录属于训练数据，检查只验证代码和状态恢复，不说明训练改善了事件预测。

本次产物均由 Git 忽略：

- `outputs/train_smoke/step_000000.pth`、`step_000001.pth`、`metrics.jsonl`：默认入口的更新前/后状态和日志。
- `outputs/trajectory_profile/gpu1_resume_chunk_3/step_000002.pth`：实际恢复后完成第二次更新的状态。
- `outputs/trajectory_profile/summary.json`：首次更新的资源比较、初始状态一致性及一步数值差异；各 `gpu1_resume_trial_*.json` 记录恢复后的实测结果。
- `outputs/trajectory_profile/final_validation.json`：最终选择 3 步分块及验证汇总；`restore_event_validation.json`、`event_tensor_gradients.json`：真实 RNG/数据游标重放与事件梯度检查。
- `outputs/evaluation_smoke/metrics.json`：冻结模型的单批诊断；`outputs/evaluation_trained_smoke/metrics.json`：默认 checkpoint 的单批诊断。

正式训练数据覆盖、验证/测试缓存、超参数选择、公平对照、概率校准统计和多个种子的研究证据仍待开展；当前没有复现文献 baseline 指标。
