/** 浏览器端手写推理；权重顺序与 NumPy 的 C 顺序一致。 */
function dense(input, weights, biases, relu = false) {
  const output = new Float64Array(biases.length);
  for (let j = 0; j < output.length; j++) {
    let sum = biases[j];
    for (let i = 0; i < input.length; i++) sum += input[i] * weights[i * output.length + j];
    output[j] = relu ? Math.max(0, sum) : sum;
  }
  return output;
}

function convolution(input, inChannels, size, weights, biases) {
  const outChannels = biases.length;
  const out = new Float64Array(outChannels * size * size);
  for (let oc = 0; oc < outChannels; oc++) for (let y = 0; y < size; y++) for (let x = 0; x < size; x++) {
    let value = biases[oc];
    for (let ic = 0; ic < inChannels; ic++) for (let ky = 0; ky < 3; ky++) for (let kx = 0; kx < 3; kx++) {
      const iy = y + ky - 1, ix = x + kx - 1;
      if (iy >= 0 && iy < size && ix >= 0 && ix < size) {
        value += input[(ic * size + iy) * size + ix] * weights[((oc * inChannels + ic) * 3 + ky) * 3 + kx];
      }
    }
    out[(oc * size + y) * size + x] = Math.max(0, value);
  }
  return out;
}

function maxPool(input, channels, size) {
  const outputSize = size / 2;
  const out = new Float64Array(channels * outputSize * outputSize);
  for (let c = 0; c < channels; c++) for (let y = 0; y < outputSize; y++) for (let x = 0; x < outputSize; x++) {
    const offset = (c * size + y * 2) * size + x * 2;
    out[(c * outputSize + y) * outputSize + x] = Math.max(input[offset], input[offset + 1], input[offset + size], input[offset + size + 1]);
  }
  return out;
}

/** 使用导出模型计算 10 个类别概率。
 * @param {object} model 导出的模型结构和权重。
 * @param {number[]} pixels 784 个归一化灰度值。
 * @returns {number[]} Softmax 概率。
 */
function predictProbabilities(model, pixels) {
  if (pixels.length !== 784) throw new Error('输入图像必须为 28×28');
  let value = Float64Array.from(pixels);
  if (model.type === 'mlp') {
    for (let l = 0; l < model.weights.length; l++) {
      const matrix = model.weights[l];
      const flat = matrix.flat();
      value = dense(value, flat, model.biases[l]);
      if (l < model.weights.length - 1) value = value.map(v => 1 / (1 + Math.exp(-Math.max(-80, Math.min(80, v)))));
    }
  } else if (model.type === 'cnn') {
    value = maxPool(convolution(value, 1, 28, model.weights[0], model.biases[0]), model.channels[0], 28);
    value = maxPool(convolution(value, model.channels[0], 14, model.weights[1], model.biases[1]), model.channels[1], 14);
    value = dense(value, model.weights[2], model.biases[2], true);
    value = dense(value, model.weights[3], model.biases[3]);
  } else throw new Error('不支持的模型类型');
  const max = Math.max(...value);
  const exp = Array.from(value, v => Math.exp(v - max));
  const sum = exp.reduce((a, b) => a + b, 0);
  return exp.map(v => v / sum);
}
if (typeof module !== 'undefined') module.exports = { predictProbabilities };
