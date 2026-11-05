const canvas = document.querySelector('#drawing');
const ctx = canvas.getContext('2d', { willReadFrequently: true });
const hint = document.querySelector('#canvasHint');
const message = document.querySelector('#message');
const button = document.querySelector('#predict');
const methodSelect = document.querySelector('#method');
const modelInfo = document.querySelector('#modelInfo');
let methods = [];
let drawing = false;
let hasInk = false;
let revision = 0;
ctx.strokeStyle = '#fff';
ctx.lineWidth = 20;
ctx.lineCap = 'round';
ctx.lineJoin = 'round';

function showModelInfo() {
  const method = methods.find(item => item.id === methodSelect.value);
  modelInfo.textContent = method && method.available
    ? `已发布模型：${method.model_name}`
    : '该方法尚无模型，请先训练。';
}

async function refreshModels() {
  try {
    const response = await fetch('models/manifest.json', { cache: 'no-store' });
    if (!response.ok) throw new Error('无法读取模型列表');
    const data = await response.json();
    const previous = methodSelect.value;
    methods = data.methods.map(item => ({ ...item, available: true }));
    methodSelect.replaceChildren();
    methods.forEach(method => {
      const option = document.createElement('option');
      option.value = method.id;
      option.textContent = method.label + (method.available ? '' : '（尚未训练）');
      option.disabled = !method.available;
      methodSelect.append(option);
    });
    const selected = methods.find(item => item.id === previous && item.available)
      || methods.find(item => item.available);
    methodSelect.disabled = !selected;
    if (selected) methodSelect.value = selected.id;
    showModelInfo();
  } catch (error) {
    methodSelect.disabled = true;
    modelInfo.textContent = '无法读取模型列表，请检查网页部署或刷新页面。';
  }
}

methodSelect.addEventListener('change', () => {
  revision++;
  message.textContent = '';
  document.querySelector('#empty').hidden = false;
  document.querySelector('#result').hidden = true;
  showModelInfo();
});
// 页面重新获得焦点时检查是否有新的训练结果。
window.addEventListener('focus', refreshModels);
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
  if (methodSelect.disabled || !methodSelect.value) { message.textContent = '请先选择可用的模型方法。'; return; }
  const currentRevision = revision;
  const method = methodSelect.value;
  button.disabled = true; message.textContent = '';
  try {
    const pixels = preprocess();
    showPreview(pixels);
    const entry = methods.find(item => item.id === method);
    const response = await fetch(entry.file);
    if (!response.ok) throw new Error('模型加载失败，请刷新页面重试');
    const model = await response.json();
    const probabilities = predictProbabilities(model, pixels);
    const digit = probabilities.indexOf(Math.max(...probabilities));
    const data = { digit, probabilities, model_name: entry.model_name };
    if (currentRevision !== revision) return;
    modelInfo.textContent = `本次使用：${data.model_name}`;
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
  finally { button.disabled = false; }
});
