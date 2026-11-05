"""MNIST 全连接网络的训练与测试集预测命令行入口。"""

import argparse
import csv
from datetime import datetime
import logging
from pathlib import Path
import shutil
import sys
import time
import uuid

import numpy as np
import yaml

from load_data import CONFIG_PATH, load_data_paths, read_images, read_labels
from network import NeuralNetwork


def prepare_dataset(images_path: Path, labels_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """读取 IDX 数据，检查标签并将图片展平、归一化。

    Args:
        images_path: 图片压缩文件路径。
        labels_path: 标签压缩文件路径。

    Returns:
        形状分别为 (样本数, 784)、(样本数,) 的输入与整数标签。

    Raises:
        ValueError: 图片尺寸、标签范围或样本数量不符合 MNIST 要求。
    """
    images = read_images(images_path)
    labels = read_labels(labels_path)
    if images.shape[1:] != (28, 28) or len(images) != len(labels) or len(labels) == 0:
        raise ValueError("图片应为 28×28，且图片和标签的非零样本数量应一致")
    if np.any((labels < 0) | (labels > 9)):
        raise ValueError("MNIST 标签必须位于 0 到 9 之间")
    return images.reshape(-1, 784).astype(np.float32) / 255.0, labels


def evaluate(model: NeuralNetwork, x: np.ndarray, labels: np.ndarray,
             batch_size: int) -> tuple[float, float, np.ndarray]:
    """分批评估网络，计算平均损失、准确率和全部样本的概率。

    Args:
        model: 待评估网络。
        x: 输入样本数组。
        labels: 与输入对应的类别标签。
        batch_size: 每次前向计算的最大样本数。

    Returns:
        平均损失、准确率和形状为 (样本数, 10) 的概率数组。
    """
    total_loss = 0.0
    correct = 0
    all_probabilities = []
    for start in range(0, len(labels), batch_size):
        batch_x = x[start:start + batch_size]
        batch_y = labels[start:start + batch_size]
        logits, _ = model.forward(batch_x)
        loss, probabilities = model.loss_and_probabilities(logits, batch_y)
        total_loss += loss * len(batch_y)
        correct += int((probabilities.argmax(axis=1) == batch_y).sum())
        all_probabilities.append(probabilities)
    return total_loss / len(labels), correct / len(labels), np.concatenate(all_probabilities)


def write_predictions(path: Path, labels: np.ndarray, probabilities: np.ndarray) -> None:
    """将测试集标签、预测类别及各类概率按原始样本顺序写入 CSV。

    Args:
        path: 输出 CSV 的路径。
        labels: 测试集真实标签。
        probabilities: 每个样本对应 10 个类别的预测概率。
    """
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["sample_index", "true_label", "predicted_label", "correct"] +
                        [f"probability_{i}" for i in range(10)])
        for index, (label, probs) in enumerate(zip(labels, probabilities)):
            prediction = int(np.argmax(probs))
            writer.writerow([index, int(label), prediction, int(prediction == label)] +
                            [f"{float(p):.8f}" for p in probs])


def run(args: argparse.Namespace) -> None:
    """根据命令行参数训练或加载模型，并输出测试结果。

    Args:
        args: 已解析的命令行参数。
    """
    config_path = args.config.resolve()
    with config_path.open(encoding="utf-8") as file:
        config = yaml.safe_load(file)
    paths = load_data_paths(config_path)
    settings = config["training"]
    batch_size = settings["batch_size"]
    epochs = settings["epochs"]
    learning_rate = settings["learning_rate"]
    validation_size = settings["validation_size"]
    seed = config["model"]["seed"]
    hidden_sizes = config["model"]["hidden_sizes"]
    if (not isinstance(batch_size, int) or batch_size <= 0 or
            not isinstance(epochs, int) or epochs <= 0 or
            not isinstance(validation_size, int) or validation_size <= 0 or
            not isinstance(learning_rate, (float, int)) or not np.isfinite(learning_rate) or learning_rate <= 0 or
            not isinstance(hidden_sizes, list) or not hidden_sizes or
            any(not isinstance(size, int) or size <= 0 for size in hidden_sizes)):
        raise ValueError("请检查训练参数：轮数、批大小、验证集大小、学习率及隐藏层尺寸必须为正")

    output_root = config_path.parent / Path(config["output"]["directory"]).expanduser()
    run_dir = output_root / (datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:8])
    run_dir.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(config_path, run_dir / "config.yaml")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        handlers=[logging.FileHandler(run_dir / "run.log", encoding="utf-8"),
                                  logging.StreamHandler(sys.stdout)], force=True)
    logging.captureWarnings(True)
    logging.info("运行目录: %s", run_dir.resolve())
    logging.info("配置文件: %s", config_path)
    logging.info("模式: %s", "预测" if args.model else "训练并预测")
    start_time = time.monotonic()
    try:
        x_test, y_test = prepare_dataset(paths["test_images"], paths["test_labels"])
        if args.model:
            model = NeuralNetwork.load(args.model)
            if model.sizes[0] != 784 or model.sizes[-1] != 10:
                raise ValueError("模型的输入或输出维度不适合 MNIST")
            logging.info("加载模型: %s，结构: %s", args.model.resolve(), model.sizes)
        else:
            x_all, y_all = prepare_dataset(paths["train_images"], paths["train_labels"])
            if validation_size >= len(y_all):
                raise ValueError("验证集大小必须小于训练数据总量")
            rng = np.random.default_rng(seed)
            indices = rng.permutation(len(y_all))
            val_indices, train_indices = indices[:validation_size], indices[validation_size:]
            x_val, y_val = x_all[val_indices], y_all[val_indices]
            x_train, y_train = x_all[train_indices], y_all[train_indices]
            del x_all, y_all
            model = NeuralNetwork([784, *hidden_sizes, 10], seed)
            logging.info("网络: %s；训练: %d，验证: %d，测试: %d", model.sizes,
                         len(y_train), len(y_val), len(y_test))
            logging.info("epochs=%d batch_size=%d learning_rate=%s seed=%d", epochs, batch_size, learning_rate, seed)
            with (run_dir / "training.csv").open("w", encoding="utf-8", newline="") as file:
                writer = csv.writer(file)
                writer.writerow(["epoch", "train_loss_during_updates", "validation_loss", "validation_accuracy", "seconds"])
                for epoch in range(1, epochs + 1):
                    epoch_start = time.monotonic()
                    order = rng.permutation(len(y_train))
                    total_loss = 0.0
                    for start in range(0, len(order), batch_size):
                        batch_indices = order[start:start + batch_size]
                        loss = model.train_batch(x_train[batch_indices], y_train[batch_indices], learning_rate)
                        if not np.isfinite(loss):
                            raise FloatingPointError("训练损失不是有限数，请检查学习率和数据")
                        total_loss += loss * len(batch_indices)
                    train_loss = total_loss / len(y_train)
                    val_loss, val_accuracy, _ = evaluate(model, x_val, y_val, batch_size)
                    seconds = time.monotonic() - epoch_start
                    writer.writerow([epoch, f"{train_loss:.6f}", f"{val_loss:.6f}",
                                     f"{val_accuracy:.6f}", f"{seconds:.2f}"])
                    file.flush()
                    logging.info("epoch %d/%d train_loss=%.4f val_loss=%.4f val_accuracy=%.2f%% time=%.1fs",
                                 epoch, epochs, train_loss, val_loss, val_accuracy * 100, seconds)
            model.save(run_dir / "model.npz")
            logging.info("模型已保存: %s", run_dir / "model.npz")
        test_loss, test_accuracy, probabilities = evaluate(model, x_test, y_test, batch_size)
        write_predictions(run_dir / "predictions.csv", y_test, probabilities)
        logging.info("测试 loss=%.4f accuracy=%.2f%%，预测结果已写入 predictions.csv", test_loss, test_accuracy * 100)
        logging.info("运行完成，耗时 %.1f 秒", time.monotonic() - start_time)
    except Exception:
        logging.exception("运行失败")
        raise


def main() -> None:
    """解析命令行参数并启动运行。"""
    parser = argparse.ArgumentParser(description="使用 NumPy 手写神经网络训练和预测 MNIST")
    parser.add_argument("--config", type=Path, default=CONFIG_PATH, help="配置文件路径")
    parser.add_argument("--model", type=Path, help="加载已有 model.npz，仅运行测试集预测")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
