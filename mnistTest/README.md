# 用 NumPy 实现 MNIST 全连接神经网络

项目手写前向传播、Softmax、交叉熵、反向传播和小批量梯度下降。NumPy 用于数组与矩阵运算，PyYAML 用于读取配置；不使用深度学习或机器学习框架。

## 安装与运行

在项目根目录 `mnistTest` 下运行：

```bash
python3 -m pip install -r requirements.txt
python3 src/main.py
```

默认配置采用 `784 → 128 → 64 → 10` 网络，隐藏层使用 Sigmoid，输出层使用 Softmax；使用 Xavier 正态初始化权重、零初始化偏置，训练 15 个 epoch，batch size 为 128，学习率为 0.5。原始 60,000 个训练样本随机划分为 50,000 个训练样本和 10,000 个验证样本，随机种子为 42。MNIST 测试集只在训练结束后评估。

可以在 `config.yaml` 修改数据路径、隐藏层大小、随机种子、训练参数及输出目录。相对路径以该配置文件所在目录为准。指定其他配置文件：

```bash
python3 src/main.py --config /path/to/config.yaml
```

训练完成后可加载保存的模型，单独对配置中的测试集进行预测：

```bash
python3 src/main.py --model runs/某次训练目录/model.npz
```

该命令会新建运行目录并输出预测 CSV，不重新训练。目前预测入口面向 MNIST IDX 测试集，不读取单张外部图片。

也可以使用 [手写识别网页](web/README.md)，通过鼠标或触屏书写并调用已训练模型。

## 运行结果

每次运行在 `runs/日期_时间_随机后缀/` 下创建独立目录。运行过程中终端也会显示日志：

| 文件 | 内容 |
| --- | --- |
| `run.log` | 参数、每轮指标、最终测试指标和错误信息 |
| `config.yaml` | 本次配置快照，仅供记录；其中的相对路径仍以原配置文件目录为基准 |
| `training.csv` | 每轮训练损失、验证集损失、验证集准确率和耗时；仅训练时生成 |
| `model.npz` | 网络结构和权重、偏置；仅训练时生成 |
| `predictions.csv` | 测试集每个样本的原始索引、真实标签、预测标签、是否正确和 10 类概率 |

`training.csv` 的 `train_loss_during_updates` 是各 batch 在**各自参数更新前**的损失按样本数加权平均；因此它不是 epoch 结束后用同一组最终参数重新计算的训练集损失。验证损失和准确率则使用该轮结束时的参数计算。`sample_index` 从 0 开始，对应原始测试集顺序。

## 代码与公式

- `src/load_data.py`：读取 YAML 路径和 gzip 格式的 IDX 数据。
- `src/network.py`：网络初始化、前向传播、损失、梯度、参数更新和模型保存加载。
- `src/main.py`：数据准备、划分、训练循环、评估、日志和 CSV。

代码按样本为行组织矩阵。每层计算 $Z=A W+b$，隐藏层 $A=\sigma(Z)$；输出层计算 Softmax 和平均交叉熵。反向传播中输出层梯度为 $(P-Y)/B$，然后依次计算 $\nabla_W=A^T\Delta$、$\nabla_b=\sum\Delta$，并通过 $\Delta_{前层}=(\Delta W^T)\odot A(1-A)$ 向前传递。最后同时更新每层的 $W\leftarrow W-\eta\nabla_W$ 和 $b\leftarrow b-\eta\nabla_b$。
