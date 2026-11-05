from pathlib import Path
import gzip
import struct

import numpy as np
import yaml


# 根据脚本位置找到配置文件，避免从不同目录运行时找不到配置。
CONFIG_PATH = Path(__file__).resolve().parent.parent / "config.yaml"


def load_data_paths(config_path: Path = CONFIG_PATH) -> dict[str, Path]:
    """读取配置中的数据路径；相对路径以配置文件所在目录为准。

    Args:
        config_path: YAML 配置文件路径。

    Returns:
        数据名称到对应文件路径的映射。
    """
    config_path = Path(config_path).resolve()
    with config_path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    return {
        name: config_path.parent / Path(path).expanduser()
        for name, path in config["data"].items()
    }


def read_images(path: Path) -> np.ndarray:
    """读取 MNIST 图片。

    Args:
        path: IDX 图片 gzip 文件路径。

    Returns:
        形状为 (图片数量, 高度, 宽度) 的无符号整数数组。
    """
    with gzip.open(path, "rb") as f:
        # 图片文件的前 16 字节是文件头。
        # > 表示大端字节序；4 个 I 分别表示 4 个无符号 32 位整数。
        magic, count, rows, cols = struct.unpack(">IIII", f.read(16))

        if magic != 2051:
            raise ValueError(f"不是 MNIST 图片文件：{path}")

        # 文件头之后，每个字节表示一个像素。
        pixels = np.frombuffer(f.read(), dtype=np.uint8)

    return pixels.reshape(count, rows, cols)


def read_labels(path: Path) -> np.ndarray:
    """读取 MNIST 标签。

    Args:
        path: IDX 标签 gzip 文件路径。

    Returns:
        形状为 (标签数量,) 的整数数组。
    """
    with gzip.open(path, "rb") as f:
        # 标签文件的前 8 字节是文件头。
        magic, count = struct.unpack(">II", f.read(8))

        if magic != 2049:
            raise ValueError(f"不是 MNIST 标签文件：{path}")

        labels = np.frombuffer(f.read(), dtype=np.uint8)

    return labels.reshape(count).astype(np.int64)


if __name__ == "__main__":
    paths = load_data_paths()
    train_images = read_images(paths["train_images"])
    train_labels = read_labels(paths["train_labels"])
    test_images = read_images(paths["test_images"])
    test_labels = read_labels(paths["test_labels"])

    # 普通全连接神经网络：把每张 28×28 图片展平成 784 个输入特征。
    # 转成浮点数，并将像素值从 [0, 255] 归一化到 [0, 1]。
    x_train = train_images.reshape(-1, 784).astype(np.float32) / 255.0
    x_test = test_images.reshape(-1, 784).astype(np.float32) / 255.0

    y_train = train_labels
    y_test = test_labels

    print("训练输入：", x_train.shape)   # (60000, 784)
    print("训练标签：", y_train.shape)   # (60000,)
    print("测试输入：", x_test.shape)    # (10000, 784)
    print("测试标签：", y_test.shape)    # (10000,)
    print("第一张训练图片的标签：", y_train[0])
