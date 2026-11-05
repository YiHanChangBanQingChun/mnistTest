"""本地手写识别服务：按所选方法自动加载最新模型。"""
import argparse
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import sys
import threading

import numpy as np

WEB_DIR = Path(__file__).resolve().parent
PROJECT_DIR = WEB_DIR.parent
sys.path.insert(0, str(PROJECT_DIR / 'src'))
sys.path.insert(0, str(PROJECT_DIR / 'src-cnn'))
from network import NeuralNetwork
from cnn import CNN


class ModelRegistry:
    """发现模型并缓存当前版本；串行执行预测以保护模型的前向缓存。

    Args:
        project_dir: 包含 runs、runs-cnn 和模型文件的项目目录。
        mlp_override: 可选的全连接模型固定路径，兼容原有 --model 参数。
    """

    def __init__(self, project_dir: Path, mlp_override: Path | None = None):
        self.project_dir = project_dir
        self.mlp_override = mlp_override
        self.lock = threading.RLock()
        self.cache = {}
        self.methods = {
            'mlp': ('全连接神经网络', 'runs', NeuralNetwork),
            'cnn': ('卷积神经网络 CNN', 'runs-cnn', CNN),
        }

    def latest_path(self, method: str) -> Path | None:
        """返回某个方法最新修改的模型路径，没有文件则返回 None。"""
        if method == 'mlp' and self.mlp_override:
            return self.mlp_override if self.mlp_override.is_file() else None
        directory = self.methods[method][1]
        candidates = [p for p in (self.project_dir / directory).glob('*/model.npz') if p.is_file()]
        return max(candidates, key=lambda p: (p.stat().st_mtime_ns, str(p))) if candidates else None

    def list_methods(self) -> list[dict]:
        """列出每个方法、最新模型名称和是否可用。"""
        result = []
        for method, (label, _, _) in self.methods.items():
            path = self.latest_path(method)
            result.append({'id': method, 'label': label, 'available': path is not None,
                           'model_name': f'{path.parent.name}/{path.name}' if path else None,
                           'pinned': method == 'mlp' and self.mlp_override is not None})
        return result

    def predict(self, method: str, pixels: np.ndarray) -> dict:
        """加载所选方法的最新模型并执行预测，返回类别、概率和模型名称。"""
        if method not in self.methods:
            raise ValueError('未知的模型方法')
        with self.lock:
            path = self.latest_path(method)
            if path is None:
                raise ValueError('该方法尚无可用模型，请先训练模型')
            stat = path.stat()
            key = (path.resolve(), stat.st_mtime_ns, stat.st_size)
            cached = self.cache.get(method)
            if cached is None or cached[0] != key:
                model = self.methods[method][2].load(path)
                if method == 'mlp' and (model.sizes[0] != 784 or model.sizes[-1] != 10):
                    raise ValueError('模型的输入或输出维度不匹配 MNIST')
                self.cache[method] = (key, model)
                print(f'加载 {self.methods[method][0]}: {path.resolve()}', flush=True)
            model = self.cache[method][1]
            output = model.forward(pixels.reshape(1, 784))
            logits = output[0] if method == 'mlp' else output
            scores = logits[0] - logits[0].max()
            exp = np.exp(scores)
            probabilities = exp / exp.sum()
            if not np.isfinite(probabilities).all():
                raise ValueError('模型输出无效，请检查模型文件')
            return {'method': method, 'model_name': f'{path.parent.name}/{path.name}',
                    'digit': int(probabilities.argmax()), 'probabilities': probabilities.tolist()}


class Handler(SimpleHTTPRequestHandler):
    """提供静态网页、可用模型列表和预测接口。"""

    def __init__(self, *args, registry: ModelRegistry, **kwargs):
        self.registry = registry
        super().__init__(*args, directory=str(WEB_DIR), **kwargs)

    def do_GET(self):
        """返回模型列表，其他路径由静态文件处理器提供。"""
        if self.path == '/api/models':
            self.respond(200, {'methods': self.registry.list_methods()})
        else:
            super().do_GET()

    def do_POST(self):
        """校验像素和模型方法，返回预测结果。"""
        if self.path != '/api/predict':
            self.send_error(404)
            return
        try:
            length = int(self.headers.get('Content-Length', '0'))
            if not 0 < length <= 20000:
                raise ValueError('请求数据大小不正确')
            data = json.loads(self.rfile.read(length))
            pixels = np.asarray(data['pixels'], dtype=np.float32)
            if pixels.shape != (784,) or not np.isfinite(pixels).all() or np.any((pixels < 0) | (pixels > 1)):
                raise ValueError('需要 784 个 0 到 1 之间的像素值')
            if not np.any(pixels > 0):
                raise ValueError('请先写一个数字')
            result = self.registry.predict(data.get('method', 'mlp'), pixels)
            self.respond(200, result)
        except (ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
            self.respond(400, {'error': str(exc)})
        except (OSError, EOFError) as exc:
            print(f'模型读取失败: {exc}', file=sys.stderr, flush=True)
            self.respond(500, {'error': '模型文件读取失败，请检查模型文件'})

    def respond(self, status: int, payload: dict) -> None:
        """以 UTF-8 JSON 返回数据，避免缓存过期模型信息。"""
        body = json.dumps(payload, ensure_ascii=False).encode('utf-8')
        self.send_response(status)
        self.send_header('Content-Type', 'application/json; charset=utf-8')
        self.send_header('Cache-Control', 'no-store')
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def main() -> None:
    """启动本地服务；模型在预测时自动查找和加载。"""
    parser = argparse.ArgumentParser(description='MNIST 手写识别本地服务')
    parser.add_argument('--model', type=Path, help='固定使用指定的全连接模型（不影响 CNN）')
    parser.add_argument('--host', default='127.0.0.1', help='监听地址；局域网访问可指定电脑的局域网 IP')
    parser.add_argument('--port', type=int, default=8000)
    args = parser.parse_args()
    if args.model and not args.model.is_file():
        parser.error('指定的全连接模型文件不存在')
    registry = ModelRegistry(PROJECT_DIR, args.model.resolve() if args.model else None)
    server = ThreadingHTTPServer((args.host, args.port), partial(Handler, registry=registry))
    print(f'监听地址: {args.host}:{args.port}  （按 Ctrl+C 停止）', flush=True)
    if args.host == '0.0.0.0':
        print(f'在手机浏览器中访问 http://电脑的局域网IP:{args.port}', flush=True)
    else:
        print(f'打开 http://{args.host}:{args.port}', flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == '__main__':
    main()
