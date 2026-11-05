"""仅使用 NumPy 实现多层全连接神经网络及其反向传播。"""

from pathlib import Path

import numpy as np


class NeuralNetwork:
    """Sigmoid 隐藏层、Softmax 输出层的全连接分类器。

    Args:
        sizes: 每层的神经元数量，包括输入层和输出层。
        seed: 随机初始化权重使用的种子。
    """

    def __init__(self, sizes: list[int], seed: int = 42) -> None:
        if len(sizes) < 3 or any(size <= 0 for size in sizes):
            raise ValueError("网络至少需要输入层、一个隐藏层和输出层，且维度必须为正")
        self.sizes = list(sizes)
        rng = np.random.default_rng(seed)
        # Xavier 正态初始化；使用行向量 batch，权重形状为 (输入维度, 输出维度)。
        self.weights = [
            rng.normal(0, np.sqrt(2.0 / (n_in + n_out)), (n_in, n_out)).astype(np.float32)
            for n_in, n_out in zip(sizes[:-1], sizes[1:])
        ]
        self.biases = [np.zeros(n, dtype=np.float32) for n in sizes[1:]]

    def forward(self, x: np.ndarray) -> tuple[np.ndarray, list[np.ndarray]]:
        """计算输出层分数并保存各层激活值，供反向传播使用。

        Args:
            x: 已归一化的输入，形状为 (batch_size, 输入维度)。

        Returns:
            包含 logits 和激活值列表的元组；列表首项为输入。
        """
        activations = [x]
        a = x
        for weight, bias in zip(self.weights[:-1], self.biases[:-1]):
            z = a @ weight + bias
            # 截断避免指数溢出；在浮点精度内，极端值已饱和。
            a = 1.0 / (1.0 + np.exp(-np.clip(z, -80, 80)))
            activations.append(a)
        logits = a @ self.weights[-1] + self.biases[-1]
        return logits, activations

    @staticmethod
    def loss_and_probabilities(logits: np.ndarray, labels: np.ndarray) -> tuple[float, np.ndarray]:
        """计算数值稳定的平均交叉熵与 Softmax 概率。

        Args:
            logits: 输出层分数，形状为 (样本数, 类别数)。
            labels: 整数类别标签，形状为 (样本数,)。

        Returns:
            平均交叉熵以及与 logits 形状相同的概率数组。
        """
        shifted = logits - logits.max(axis=1, keepdims=True)
        exp_scores = np.exp(shifted)
        sums = exp_scores.sum(axis=1, keepdims=True)
        probabilities = exp_scores / sums
        log_probs = shifted - np.log(sums)
        loss = -log_probs[np.arange(len(labels)), labels].mean()
        return float(loss), probabilities

    def gradients(self, activations: list[np.ndarray], probabilities: np.ndarray,
                  labels: np.ndarray) -> tuple[list[np.ndarray], list[np.ndarray]]:
        """按链式法则计算各层权重和偏置的平均梯度。

        Args:
            activations: 前向传播得到的各层输入及隐藏层输出。
            probabilities: Softmax 输出的类别概率。
            labels: 当前 batch 的整数标签。

        Returns:
            权重梯度列表与偏置梯度列表，顺序均为从输入到输出。
        """
        # Softmax + 交叉熵：dJ/dlogits = (p - one_hot(y)) / batch_size。
        delta = probabilities.copy()
        delta[np.arange(len(labels)), labels] -= 1
        delta /= len(labels)
        weight_grads = [np.empty_like(w) for w in self.weights]
        bias_grads = [np.empty_like(b) for b in self.biases]
        for layer in range(len(self.weights) - 1, -1, -1):
            weight_grads[layer] = activations[layer].T @ delta
            bias_grads[layer] = delta.sum(axis=0)
            if layer > 0:
                a = activations[layer]
                delta = (delta @ self.weights[layer].T) * a * (1.0 - a)
        return weight_grads, bias_grads

    def train_batch(self, x: np.ndarray, labels: np.ndarray, learning_rate: float) -> float:
        """完成一个 batch 的前向、反向计算和梯度下降更新。

        Args:
            x: 当前 batch 的输入。
            labels: 当前 batch 的类别标签。
            learning_rate: 梯度下降步长。

        Returns:
            更新参数前的 batch 平均损失。
        """
        logits, activations = self.forward(x)
        loss, probabilities = self.loss_and_probabilities(logits, labels)
        weight_grads, bias_grads = self.gradients(activations, probabilities, labels)
        for i in range(len(self.weights)):
            self.weights[i] -= learning_rate * weight_grads[i]
            self.biases[i] -= learning_rate * bias_grads[i]
        return loss

    def save(self, path: Path) -> None:
        """将模型结构和全部参数保存为 NumPy 压缩文件。

        Args:
            path: 输出的 .npz 文件路径。
        """
        params = {"sizes": np.array(self.sizes, dtype=np.int64)}
        for i, (weight, bias) in enumerate(zip(self.weights, self.biases)):
            params[f"weight_{i}"] = weight
            params[f"bias_{i}"] = bias
        np.savez_compressed(path, **params)

    @classmethod
    def load(cls, path: Path) -> "NeuralNetwork":
        """加载由本类保存的网络。

        Args:
            path: 已保存的 .npz 文件路径。

        Returns:
            已恢复参数的网络对象。
        """
        with np.load(path, allow_pickle=False) as data:
            model = cls(data["sizes"].tolist())
            model.weights = [data[f"weight_{i}"].copy() for i in range(len(model.weights))]
            model.biases = [data[f"bias_{i}"].copy() for i in range(len(model.biases))]
        return model
