"""使用 NumPy 实现卷积神经网络及全部反向传播。"""
from pathlib import Path
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view


class Conv2D:
    """步长 1、边缘补零的 3×3 卷积（实现中使用互相关）。

    Args:
        in_channels: 输入通道数。
        out_channels: 输出通道数。
        rng: NumPy 随机数生成器。
    """

    def __init__(self, in_channels: int, out_channels: int, rng: np.random.Generator):
        self.w = rng.normal(0, np.sqrt(2 / (in_channels * 9)),
                            (out_channels, in_channels, 3, 3)).astype(np.float32)
        self.b = np.zeros(out_channels, dtype=np.float32)

    def forward(self, x: np.ndarray) -> np.ndarray:
        """计算卷积特征图并保存输入列矩阵。

        Args:
            x: NCHW 排列的输入特征图。

        Returns:
            NCHW 排列的输出特征图。
        """
        self.shape = x.shape
        n, c, h, w = x.shape
        padded = np.pad(x, ((0, 0), (0, 0), (1, 1), (1, 1)))
        windows = sliding_window_view(padded, (3, 3), axis=(2, 3))
        self.cols = windows.transpose(0, 2, 3, 1, 4, 5).reshape(n * h * w, c * 9)
        out = self.cols @ self.w.reshape(len(self.b), -1).T + self.b
        return out.reshape(n, h, w, len(self.b)).transpose(0, 3, 1, 2)

    def backward(self, dout: np.ndarray) -> np.ndarray:
        """从输出梯度计算权重、偏置和输入梯度。

        Args:
            dout: 后一层传回的输出梯度，NCHW 排列。

        Returns:
            对输入的梯度；重叠窗口的梯度贡献已经累加。
        """
        n, c, h, w = self.shape
        d = dout.transpose(0, 2, 3, 1).reshape(n * h * w, -1)
        self.dw = (d.T @ self.cols).reshape(self.w.shape)
        self.db = d.sum(axis=0)
        dcols = (d @ self.w.reshape(len(self.b), -1)).reshape(n, h, w, c, 3, 3)
        dx_pad = np.zeros((n, c, h + 2, w + 2), dtype=dout.dtype)
        for ky in range(3):
            for kx in range(3):
                dx_pad[:, :, ky:ky+h, kx:kx+w] += dcols[:, :, :, :, ky, kx].transpose(0, 3, 1, 2)
        return dx_pad[:, :, 1:-1, 1:-1]


class MaxPool2D:
    """2×2 最大池化，步长为 2；反向传播只经过最大值位置。"""

    def forward(self, x: np.ndarray) -> np.ndarray:
        """缩小空间尺寸并记录每个池化窗口的最大值位置。

        Args:
            x: 高和宽均为偶数的 NCHW 特征图。

        Returns:
            空间尺寸减半的特征图。
        """
        self.shape = x.shape
        n, c, h, w = x.shape
        patches = x.reshape(n, c, h // 2, 2, w // 2, 2).transpose(0, 1, 2, 4, 3, 5).reshape(n, c, h // 2, w // 2, 4)
        self.indices = patches.argmax(axis=-1)
        return patches.max(axis=-1)

    def backward(self, dout: np.ndarray) -> np.ndarray:
        """将梯度散射到前向传播选中的最大值位置。

        Args:
            dout: 后一层传回的池化输出梯度。

        Returns:
            与池化输入形状一致的梯度数组。
        """
        n, c, h, w = self.shape
        dx = np.zeros((*dout.shape, 4), dtype=dout.dtype)
        np.put_along_axis(dx, self.indices[..., None], dout[..., None], axis=-1)
        return dx.reshape(n, c, h // 2, w // 2, 2, 2).transpose(0, 1, 2, 4, 3, 5).reshape(self.shape)


class Dense:
    """全连接层，输入与输出均为二维矩阵。"""

    def __init__(self, n_in: int, n_out: int, rng: np.random.Generator):
        self.w = rng.normal(0, np.sqrt(2 / n_in), (n_in, n_out)).astype(np.float32)
        self.b = np.zeros(n_out, dtype=np.float32)

    def forward(self, x: np.ndarray) -> np.ndarray:
        """计算 xW+b 并保存输入。

        Args:
            x: 形状为 (样本数, 输入特征数) 的数组。

        Returns:
            形状为 (样本数, 输出特征数) 的数组。
        """
        self.x = x
        return x @ self.w + self.b

    def backward(self, dout: np.ndarray) -> np.ndarray:
        """计算全连接层权重、偏置和输入的梯度。

        Args:
            dout: 后一层传回的输出梯度。

        Returns:
            对本层输入的梯度。
        """
        self.dw = self.x.T @ dout
        self.db = dout.sum(axis=0)
        return dout @ self.w.T


class CNN:
    """两个卷积池化模块与一个隐藏全连接层组成的分类网络。

    Args:
        channels: 两个卷积层的输出通道数。
        hidden_size: 全连接隐藏层神经元数量。
        seed: 参数初始化随机种子。
        momentum: SGD 动量系数，范围为 [0, 1)。
    """

    def __init__(self, channels=(8, 16), hidden_size=64, seed=42, momentum=0.9):
        rng = np.random.default_rng(seed)
        self.channels = list(channels)
        self.hidden_size = hidden_size
        self.momentum = momentum
        self.conv1 = Conv2D(1, channels[0], rng)
        self.conv2 = Conv2D(channels[0], channels[1], rng)
        self.pool1, self.pool2 = MaxPool2D(), MaxPool2D()
        self.fc1 = Dense(channels[1] * 7 * 7, hidden_size, rng)
        self.fc2 = Dense(hidden_size, 10, rng)
        self.layers = [self.conv1, self.conv2, self.fc1, self.fc2]
        self.velocities = [(np.zeros_like(layer.w), np.zeros_like(layer.b)) for layer in self.layers]

    def forward(self, x: np.ndarray) -> np.ndarray:
        """执行卷积、池化与全连接层的前向传播。

        Args:
            x: 归一化输入，形状为 (N, 784) 或 (N, 1, 28, 28)。

        Returns:
            形状为 (N, 10) 的类别分数 logits。
        """
        x = x.reshape(-1, 1, 28, 28)
        z1 = self.conv1.forward(x)
        self.mask1 = z1 > 0
        a1 = self.pool1.forward(np.maximum(z1, 0))
        z2 = self.conv2.forward(a1)
        self.mask2 = z2 > 0
        a2 = self.pool2.forward(np.maximum(z2, 0))
        self.pool_shape = a2.shape
        z3 = self.fc1.forward(a2.reshape(len(x), -1))
        self.mask3 = z3 > 0
        return self.fc2.forward(np.maximum(z3, 0))

    @staticmethod
    def loss_and_probabilities(logits: np.ndarray, labels: np.ndarray) -> tuple[float, np.ndarray]:
        """计算稳定的平均交叉熵和 Softmax 概率。

        Args:
            logits: 形状为 (N, 10) 的输出分数。
            labels: 形状为 (N,) 的整数标签。

        Returns:
            平均损失和形状为 (N, 10) 的概率数组。
        """
        shifted = logits - logits.max(axis=1, keepdims=True)
        exp = np.exp(shifted)
        sums = exp.sum(axis=1, keepdims=True)
        loss = -np.mean((shifted - np.log(sums))[np.arange(len(labels)), labels])
        return float(loss), exp / sums

    def backward(self, probabilities: np.ndarray, labels: np.ndarray) -> None:
        """反向传播计算各层参数梯度；不更新参数。

        Args:
            probabilities: 当前 batch 的 Softmax 概率。
            labels: 当前 batch 的整数标签。
        """
        delta = probabilities.copy()
        delta[np.arange(len(labels)), labels] -= 1
        delta /= len(labels)
        delta = self.fc2.backward(delta) * self.mask3
        delta = self.fc1.backward(delta).reshape(self.pool_shape)
        delta = self.pool2.backward(delta) * self.mask2
        delta = self.conv2.backward(delta)
        delta = self.pool1.backward(delta) * self.mask1
        self.conv1.backward(delta)

    def train_batch(self, x: np.ndarray, labels: np.ndarray, learning_rate: float) -> float:
        """训练一个 batch。

        Args:
            x: 已归一化的输入图片。
            labels: 当前 batch 的整数标签。
            learning_rate: 参数更新的学习率。

        Returns:
            参数更新之前的平均损失。
        """
        loss, probabilities = self.loss_and_probabilities(self.forward(x), labels)
        self.backward(probabilities, labels)
        for layer, (vw, vb) in zip(self.layers, self.velocities):
            vw *= self.momentum
            vw += layer.dw
            vb *= self.momentum
            vb += layer.db
            layer.w -= learning_rate * vw
            layer.b -= learning_rate * vb
        return loss

    def save(self, path: Path) -> None:
        """将结构和模型参数保存为不含 pickle 的 npz 文件。

        Args:
            path: 模型文件的输出路径。
        """
        params = {'architecture': np.array('numpy-cnn-v1'), 'channels': np.array(self.channels),
                  'hidden_size': np.array(self.hidden_size), 'momentum': np.array(self.momentum)}
        for i, layer in enumerate(self.layers):
            params[f'w{i}'], params[f'b{i}'] = layer.w, layer.b
        np.savez_compressed(path, **params)

    @classmethod
    def load(cls, path: Path) -> 'CNN':
        """加载保存的 CNN 模型，用于预测。

        Args:
            path: 已保存的 npz 文件路径。

        Returns:
            包含已恢复参数的 CNN 对象。
        """
        with np.load(path, allow_pickle=False) as data:
            if str(data['architecture']) != 'numpy-cnn-v1':
                raise ValueError('模型格式不匹配')
            model = cls(data['channels'].tolist(), int(data['hidden_size']), momentum=float(data['momentum']))
            for i, layer in enumerate(model.layers):
                w, b = data[f'w{i}'], data[f'b{i}']
                if w.shape != layer.w.shape or b.shape != layer.b.shape:
                    raise ValueError('模型参数的形状与结构不匹配')
                layer.w[:], layer.b[:] = w, b
        return model
