# 尚未实现：对接官方 CasCast 的预训练组件与权重。
# 冻结 autoencoder、条件表示和确定性预测分支，确定可训练 denoiser 参数。
# 提供条件信息、去噪操作与 latent 到雷达序列的解码。
# 随机 rollout 与 log probability 的组织由 rl.py 负责。
# 本文件不复制整个上游仓库，也不包含数据或训练循环。
