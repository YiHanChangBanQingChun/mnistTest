"""手写 NumPy CNN 的训练和测试集预测入口。"""
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

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT / 'src'))
from load_data import load_data_paths
from main import prepare_dataset, write_predictions
from cnn import CNN


def evaluate(model: CNN, images: np.ndarray, labels: np.ndarray, batch_size: int) -> tuple[float, float, np.ndarray]:
    """分批计算数据集平均损失、准确率及每个样本的类别概率。

    Args:
        model: 待评估的 CNN。
        images: 已归一化的输入图片。
        labels: 图片对应的整数标签。
        batch_size: 单次计算的最大样本数。

    Returns:
        平均损失、准确率和形状为 (样本数, 10) 的概率数组。
    """
    loss_sum, correct, probabilities = 0.0, 0, []
    for start in range(0, len(labels), batch_size):
        x, y = images[start:start + batch_size], labels[start:start + batch_size]
        loss, p = model.loss_and_probabilities(model.forward(x), y)
        loss_sum += loss * len(y)
        correct += int((p.argmax(axis=1) == y).sum())
        probabilities.append(p)
    return loss_sum / len(labels), correct / len(labels), np.concatenate(probabilities)


def run(config_path: Path, model_path: Path | None) -> None:
    """训练新模型或加载已有模型，生成日志、CSV 和模型文件。

    Args:
        config_path: YAML 配置文件路径，相对路径以配置文件目录为准。
        model_path: 需要加载的模型；为 None 时执行训练。
    """
    config_path = config_path.resolve()
    with config_path.open(encoding='utf-8') as f:
        config = yaml.safe_load(f)
    paths = load_data_paths(config_path)
    settings, model_config = config['training'], config['model']
    epochs, batch_size = settings['epochs'], settings['batch_size']
    rate, momentum = settings['learning_rate'], settings['momentum']
    val_size, seed = settings['validation_size'], model_config['seed']
    channels, hidden = model_config['channels'], model_config['hidden_size']
    positive_int = lambda x: isinstance(x, int) and not isinstance(x, bool) and x > 0
    if (not all(positive_int(v) for v in (epochs, batch_size, val_size, hidden))
            or not isinstance(channels, list) or len(channels) != 2
            or not all(positive_int(c) for c in channels)
            or not isinstance(rate, (int, float)) or not np.isfinite(rate) or rate <= 0
            or not isinstance(momentum, (int, float)) or not np.isfinite(momentum) or not 0 <= momentum < 1):
        raise ValueError('请检查 CNN 配置中的结构与训练参数')
    output_dir = config_path.parent / Path(config['output']['directory']).expanduser()
    run_dir = output_dir / (datetime.now().strftime('%Y%m%d_%H%M%S') + '_' + uuid.uuid4().hex[:8])
    run_dir.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(config_path, run_dir / 'config.yaml')
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s',
                        handlers=[logging.FileHandler(run_dir / 'run.log', encoding='utf-8'), logging.StreamHandler(sys.stdout)], force=True)
    logging.captureWarnings(True)
    logging.info('运行目录: %s；配置: %s；模式: %s', run_dir.resolve(), config_path, '预测' if model_path else '训练并预测')
    started = time.monotonic()
    try:
        x_test, y_test = prepare_dataset(paths['test_images'], paths['test_labels'])
        if model_path:
            model = CNN.load(model_path)
            logging.info('加载模型: %s；channels=%s hidden=%d', model_path.resolve(), model.channels, model.hidden_size)
        else:
            x_all, y_all = prepare_dataset(paths['train_images'], paths['train_labels'])
            if val_size >= len(y_all):
                raise ValueError('验证集大小必须小于训练样本数')
            rng = np.random.default_rng(seed)
            order = rng.permutation(len(y_all))
            val_indices, train_indices = order[:val_size], order[val_size:]
            x_val, y_val = x_all[val_indices], y_all[val_indices]
            x_train, y_train = x_all[train_indices], y_all[train_indices]
            del x_all, y_all
            model = CNN(channels, hidden, seed, momentum)
            logging.info('CNN channels=%s hidden=%d；训练=%d 验证=%d 测试=%d', channels, hidden, len(y_train), len(y_val), len(y_test))
            logging.info('epochs=%d batch_size=%d learning_rate=%s momentum=%s seed=%d', epochs, batch_size, rate, momentum, seed)
            with (run_dir / 'training.csv').open('w', encoding='utf-8', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['epoch', 'train_loss_during_updates', 'validation_loss', 'validation_accuracy', 'seconds'])
                for epoch in range(1, epochs + 1):
                    start = time.monotonic()
                    order = rng.permutation(len(y_train))
                    loss_sum = 0.0
                    for i in range(0, len(order), batch_size):
                        idx = order[i:i + batch_size]
                        loss = model.train_batch(x_train[idx], y_train[idx], rate)
                        if not np.isfinite(loss):
                            raise FloatingPointError('损失不是有限数，请检查数据和学习率')
                        loss_sum += loss * len(idx)
                    train_loss = loss_sum / len(y_train)
                    val_loss, val_accuracy, _ = evaluate(model, x_val, y_val, batch_size)
                    elapsed = time.monotonic() - start
                    writer.writerow([epoch, f'{train_loss:.6f}', f'{val_loss:.6f}', f'{val_accuracy:.6f}', f'{elapsed:.2f}'])
                    f.flush()
                    logging.info('epoch %d/%d train_loss=%.4f val_loss=%.4f val_accuracy=%.2f%% time=%.1fs', epoch, epochs, train_loss, val_loss, val_accuracy * 100, elapsed)
            model.save(run_dir / 'model.npz')
            logging.info('模型已保存: %s', run_dir / 'model.npz')
        test_loss, test_accuracy, probabilities = evaluate(model, x_test, y_test, batch_size)
        write_predictions(run_dir / 'predictions.csv', y_test, probabilities)
        logging.info('测试 loss=%.4f accuracy=%.2f%%，预测结果已写入 predictions.csv', test_loss, test_accuracy * 100)
        logging.info('运行完成，耗时 %.1f 秒', time.monotonic() - started)
    except Exception:
        logging.exception('运行失败')
        raise


def main() -> None:
    """解析配置与模型路径参数并运行。"""
    parser = argparse.ArgumentParser(description='使用 NumPy 手写 CNN 识别 MNIST')
    parser.add_argument('--config', type=Path, default=Path(__file__).with_name('config.yaml'))
    parser.add_argument('--model', type=Path, help='加载已保存的 CNN 模型，只进行测试集预测')
    args = parser.parse_args()
    run(args.config, args.model)


if __name__ == '__main__':
    main()
