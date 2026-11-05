/**
 * 将画布透明度转换为 MNIST 风格的 28×28 灰度图。
 * 使用面积平均保留缩小时的笔画信息，并按灰度重心平移到中心。
 * @param {Uint8ClampedArray|Uint8Array} rgba 原画布的 RGBA 像素。
 * @param {number} width 原画布宽度。
 * @param {number} height 原画布高度。
 * @returns {number[]} 行优先排列的 784 个 0 到 1 的灰度值。
 */
function preprocessPixels(rgba, width, height) {
  if (rgba.length !== width * height * 4) throw new Error('画布数据尺寸不正确');
  let minX = width, minY = height, maxX = -1, maxY = -1;
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      if (rgba[(y * width + x) * 4 + 3] > 10) {
        minX = Math.min(minX, x); maxX = Math.max(maxX, x);
        minY = Math.min(minY, y); maxY = Math.max(maxY, y);
      }
    }
  }
  if (maxX < 0) throw new Error('请先写一个数字');
  const w = maxX - minX + 1, h = maxY - minY + 1;
  const scale = 20 / Math.max(w, h);
  const left = (28 - w * scale) / 2, top = (28 - h * scale) / 2;
  const pixels = new Float64Array(784);
  // 将每个原图像素映射到目标图，按覆盖面积分配灰度。
  for (let y = minY; y <= maxY; y++) {
    for (let x = minX; x <= maxX; x++) {
      const alpha = rgba[(y * width + x) * 4 + 3] / 255;
      if (!alpha) continue;
      const x0 = left + (x - minX) * scale, x1 = x0 + scale;
      const y0 = top + (y - minY) * scale, y1 = y0 + scale;
      for (let ty = Math.floor(y0); ty < Math.ceil(y1); ty++) {
        for (let tx = Math.floor(x0); tx < Math.ceil(x1); tx++) {
          if (tx < 0 || tx >= 28 || ty < 0 || ty >= 28) continue;
          const area = Math.max(0, Math.min(x1, tx + 1) - Math.max(x0, tx)) *
            Math.max(0, Math.min(y1, ty + 1) - Math.max(y0, ty));
          pixels[ty * 28 + tx] += alpha * area;
        }
      }
    }
  }
  let mass = 0, sumX = 0, sumY = 0;
  let inkMinX = 28, inkMaxX = -1, inkMinY = 28, inkMaxY = -1;
  for (let y = 0; y < 28; y++) for (let x = 0; x < 28; x++) {
    const value = pixels[y * 28 + x];
    mass += value; sumX += x * value; sumY += y * value;
    if (value > 0) {
      inkMinX = Math.min(inkMinX, x); inkMaxX = Math.max(inkMaxX, x);
      inkMinY = Math.min(inkMinY, y); inkMaxY = Math.max(inkMaxY, y);
    }
  }
  if (!mass) throw new Error('请先写一个数字');
  // 限制平移量，避免把笔画移出边界。
  const dx = Math.max(-inkMinX, Math.min(27 - inkMaxX, Math.round(13.5 - sumX / mass)));
  const dy = Math.max(-inkMinY, Math.min(27 - inkMaxY, Math.round(13.5 - sumY / mass)));
  const centered = new Array(784).fill(0);
  for (let y = 0; y < 28; y++) for (let x = 0; x < 28; x++) {
    const nx = x + dx, ny = y + dy;
    if (nx >= 0 && nx < 28 && ny >= 0 && ny < 28) {
      centered[ny * 28 + nx] = Math.min(1, Math.max(0, pixels[y * 28 + x]));
    }
  }
  return centered;
}
if (typeof module !== 'undefined') module.exports = { preprocessPixels };
