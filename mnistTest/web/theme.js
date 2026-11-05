const themePicker = document.querySelector('#theme');
const darkPreference = window.matchMedia('(prefers-color-scheme: dark)');
let selectedTheme = 'auto';
try {
  const saved = localStorage.getItem('mnist-theme');
  if (['auto', 'light', 'dark'].includes(saved)) selectedTheme = saved;
} catch (_) { /* 浏览器禁用本地存储时仍可切换外观。 */ }
function applyTheme() {
  const dark = selectedTheme === 'dark' || (selectedTheme === 'auto' && darkPreference.matches);
  document.documentElement.dataset.theme = dark ? 'dark' : 'light';
  document.documentElement.style.colorScheme = dark ? 'dark' : 'light';
  themePicker.value = selectedTheme;
}
themePicker.addEventListener('change', () => {
  selectedTheme = themePicker.value;
  try { localStorage.setItem('mnist-theme', selectedTheme); } catch (_) { /* 本次会话仍可使用。 */ }
  applyTheme();
});
darkPreference.addEventListener('change', applyTheme);
applyTheme();
