# MNIST 手写数字识别

使用 NumPy 手写全连接神经网络和卷积神经网络，完成 MNIST 手写数字训练与预测。网页使用导出的模型在浏览器中识别鼠标或触屏书写的数字，支持选择模型及深浅色模式。

```text
mnistTest/
  data/                MNIST 原始 gzip 数据
  src/                 全连接神经网络训练与预测
  src-cnn/             CNN 训练、预测及详细方法介绍
  config.yaml          全连接网络参数与数据路径
  tools/               导出浏览器模型的工具
  web/                 静态网页、浏览器推理及模型权重
  requirements.txt     Python 依赖
```

在 `mnistTest` 目录安装依赖并训练：

```bash
python3 -m pip install -r requirements.txt
python3 src/main.py       # 全连接网络
python3 src-cnn/main.py   # CNN
python3 tools/export_models.py  # 导出最新模型供网页使用
```

网页可通过 `python3 -m http.server 8000 --directory web` 在本地预览；更多信息见 [网页说明](mnistTest/web/README.md) 和 [CNN 方法说明](mnistTest/src-cnn/README.md)。

GitHub Pages 的自动部署配置位于 `.github/workflows/pages.yml`，发布 `mnistTest/web`。更新训练模型后，须重新导出、提交并发布，网页才会使用新模型。
