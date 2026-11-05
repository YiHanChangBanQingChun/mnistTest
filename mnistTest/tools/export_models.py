"""导出最新 NumPy 模型为浏览器可直接读取的 JSON。"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'src-cnn'))
from network import NeuralNetwork
from cnn import CNN


def latest(directory: Path) -> Path:
    """返回训练目录中最近修改的 model.npz。

    Args:
        directory: 包含各次训练子目录的目录。

    Returns:
        最近修改的模型路径。
    """
    paths = list(directory.glob('*/model.npz'))
    if not paths:
        raise FileNotFoundError(f'未找到模型: {directory}')
    return max(paths, key=lambda path: (path.stat().st_mtime_ns, str(path)))


def export_mlp(path: Path) -> dict:
    """读取全连接模型，导出网络结构、权重和偏置。"""
    model = NeuralNetwork.load(path)
    return {'type': 'mlp', 'sizes': model.sizes,
            'weights': [w.tolist() for w in model.weights],
            'biases': [b.tolist() for b in model.biases]}


def export_cnn(path: Path) -> dict:
    """读取 CNN 模型，导出卷积与全连接层的权重和偏置。"""
    model = CNN.load(path)
    return {'type': 'cnn', 'channels': model.channels, 'hidden_size': model.hidden_size,
            'weights': [layer.w.ravel().tolist() for layer in model.layers],
            'shapes': [list(layer.w.shape) for layer in model.layers],
            'biases': [layer.b.tolist() for layer in model.layers]}


def main() -> None:
    """导出指定或最新的两种模型，写入静态网页的模型目录。"""
    parser = argparse.ArgumentParser(description='导出 MNIST 浏览器模型')
    parser.add_argument('--mlp', type=Path, help='指定全连接模型，默认取 runs 中最新模型')
    parser.add_argument('--cnn', type=Path, help='指定 CNN 模型，默认取 runs-cnn 中最新模型')
    parser.add_argument('--output', type=Path, default=ROOT / 'web' / 'models')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = {'methods': []}
    for name, label, source, exporter in [
        ('mlp', '全连接神经网络', args.mlp or latest(ROOT / 'runs'), export_mlp),
        ('cnn', '卷积神经网络 CNN', args.cnn or latest(ROOT / 'runs-cnn'), export_cnn),
    ]:
        model = exporter(source)
        target = args.output / f'{name}.json'
        target.write_text(json.dumps(model, ensure_ascii=False, separators=(',', ':'), allow_nan=False), encoding='utf-8')
        manifest['methods'].append({'id': name, 'label': label, 'file': f'models/{name}.json',
                                    'model_name': f'{source.parent.name}/{source.name}'})
        print(f'{name}: {source} → {target} ({target.stat().st_size:,} bytes)')
    (args.output / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
