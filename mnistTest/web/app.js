const canvas = document.querySelector('#drawing');
const ctx = canvas.getContext('2d', { willReadFrequently: true });
const hint = document.querySelector('#canvasHint');
const message = document.querySelector('#message');
const button = document.querySelector('#predict');
const methodSelect = document.querySelector('#method');
const modelInfo = document.querySelector('#modelInfo');
let methods = [];
// 按模型路径与版本缓存已解析的模型及正在执行的请求。
const modelCache = new Map();
let predicting = false;
const retryButton = document.querySelector('#retryModel');
function cacheKey(method) { return `${method.file}::${method.model_name}`; }
function currentMethod() { return methods.find(item => item.id === methodSelect.value); }
function updateModelStatus() {
  const method = currentMethod();
  const entry = method && modelCache.get(cacheKey(method));
  const status = entry ? entry.status : 'loading';
  button.disabled = predicting || !method || status !== 'ready';
  retryButton.hidden = status !== 'error' || !method;
  modelInfo.dataset.state = status;
  if (!method) modelInfo.textContent = '正在读取可用模型…';
  else if (status === 'loading') modelInfo.textContent = `模型加载中：${method.label}…`;
  else if (status === 'ready') modelInfo.textContent = `● 模型已就绪：${method.model_name}`;
  else modelInfo.textContent = '模型加载失败，请检查网络后重试。';
}
function loadModel(method) {
  const key = cacheKey(method);
  const cached = modelCache.get(key);
  if (cached && cached.status !== 'error') return cached.promise;
  const entry = { status: 'loading', model: null, promise: null };
  modelCache.set(key, entry);
  entry.promise = (async () => {
    try {
      const response = await fetch(method.file, { cache: 'no-cache' });
      if (!response.ok) throw new Error(`模型请求失败 (${response.status})`);
      const model = await response.json();
      if (model.type !== method.id) throw new Error('模型类型与清单不匹配');
      // 提前展平全连接权重，识别时无需重复分配数组。
      if (model.type === 'mlp') model.flatWeights = model.weights.map(matrix => matrix.flat());
      entry.model = model;
      entry.status = 'ready';
    } catch (error) {
      entry.status = 'error';
      console.error('模型加载失败:', error);
    }
    updateModelStatus();
    return entry.model;
  })();
  updateModelStatus();
  return entry.promise;
}
retryButton.addEventListener('click', () => {
  const method = currentMethod();
  if (method) loadModel(method);
});
let drawing = false;
let hasInk = false;
let revision = 0;
ctx.strokeStyle = '#fff';
ctx.lineWidth = 20;
ctx.lineCap = 'round';
ctx.lineJoin = 'round';

async function refreshModels() {
  try {
    const response = await fetch('models/manifest.json', { cache: 'no-store' });
    if (!response.ok) throw new Error('无法读取模型列表');
    const data = await response.json();
    if (!Array.isArray(data.methods) || !data.methods.length) throw new Error('模型列表为空');
    methods = data.methods;
    methodSelect.replaceChildren();
    methods.forEach(method => {
      const option = document.createElement('option');
      option.value = method.id;
      option.textContent = method.label;
      methodSelect.append(option);
    });
    methodSelect.value = methods[0].id;
    methodSelect.disabled = false;
    updateModelStatus();
    // 先加载当前选择；在它完成后预加载其他模型，减少首屏网络竞争。
    await loadModel(methods[0]);
    methods.slice(1).forEach(loadModel);
  } catch (error) {
    methodSelect.disabled = true;
    button.disabled = true;
    modelInfo.dataset.state = 'error';
    modelInfo.textContent = '无法读取模型列表，请检查网络后刷新页面。';
  }
}

methodSelect.addEventListener('change', () => {
  revision++;
  message.textContent = '';
  document.querySelector('#empty').hidden = false;
  document.querySelector('#result').hidden = true;
  updateModelStatus();
  const method = currentMethod();
  if (method) loadModel(method);
});
refreshModels();

function point(event) {
  const rect = canvas.getBoundingClientRect();
  return [(event.clientX - rect.left) * canvas.width / rect.width,
    (event.clientY - rect.top) * canvas.height / rect.height];
}
canvas.addEventListener('pointerdown', event => {
  if (event.pointerType === 'mouse' && event.button !== 0) return;
  canvas.setPointerCapture(event.pointerId);
  drawing = true;
  hasInk = true;
  revision++;
  hint.hidden = true;
  message.textContent = '';
  document.querySelector('#empty').hidden = false;
  document.querySelector('#result').hidden = true;
  const [x, y] = point(event);
  ctx.beginPath(); ctx.moveTo(x, y); ctx.lineTo(x + 0.01, y); ctx.stroke();
});
canvas.addEventListener('pointermove', event => {
  if (!drawing) return;
  const [x, y] = point(event);
  ctx.lineTo(x, y); ctx.stroke();
});
for (const name of ['pointerup', 'pointercancel', 'lostpointercapture']) {
  canvas.addEventListener(name, () => { drawing = false; });
}
document.querySelector('#clear').addEventListener('click', () => {
  revision++;
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  hasInk = false; hint.hidden = false; message.textContent = '';
  document.querySelector('#empty').hidden = false;
  document.querySelector('#result').hidden = true;
});

// 从原画布读取透明度；preprocessPixels 返回模型实际接收的灰度值。
function preprocess() {
  const rgba = ctx.getImageData(0, 0, canvas.width, canvas.height).data;
  return preprocessPixels(rgba, canvas.width, canvas.height);
}

function showPreview(pixels) {
  const preview = document.querySelector('#preview');
  const previewCtx = preview.getContext('2d');
  const image = previewCtx.createImageData(28, 28);
  pixels.forEach((value, i) => {
    const gray = Math.round(value * 255);
    image.data[i * 4] = gray;
    image.data[i * 4 + 1] = gray;
    image.data[i * 4 + 2] = gray;
    image.data[i * 4 + 3] = 255;
  });
  previewCtx.putImageData(image, 0, 0);
}
button.addEventListener('click', async () => {
  if (!hasInk) { message.textContent = '请先在画布上写一个数字。'; return; }
  const entry = currentMethod();
  const cached = entry && modelCache.get(cacheKey(entry));
  if (!cached || cached.status !== 'ready') { message.textContent = '模型尚未加载完成。'; return; }
  const currentRevision = revision;
  predicting = true; updateModelStatus(); message.textContent = '';
  try {
    const pixels = preprocess();
    showPreview(pixels);
    const probabilities = predictProbabilities(cached.model, pixels);
    const digit = probabilities.indexOf(Math.max(...probabilities));
    const data = { digit, probabilities, model_name: entry.model_name };
    if (currentRevision !== revision) return;
    document.querySelector('#empty').hidden = true;
    document.querySelector('#result').hidden = false;
    document.querySelector('#digit').textContent = data.digit;
    document.querySelector('#confidence').textContent = (data.probabilities[data.digit] * 100).toFixed(1) + '%';
    const bars = document.querySelector('#bars'); bars.replaceChildren();
    data.probabilities.forEach((probability, digit) => {
      const row = document.createElement('div'); row.className = 'bar-row' + (digit === data.digit ? ' active' : '');
      const label = document.createElement('span'); label.textContent = digit;
      const track = document.createElement('div'); track.className = 'track';
      const fill = document.createElement('div'); fill.className = 'fill'; fill.style.width = `${probability * 100}%`; track.append(fill);
      const value = document.createElement('span'); value.textContent = (probability * 100).toFixed(1) + '%';
      row.append(label, track, value); bars.append(row);
    });
  } catch (error) { if (currentRevision === revision) message.textContent = error.message || '识别失败，请刷新页面后重试。'; }
  finally { predicting = false; updateModelStatus(); }
});
