import { rng, clamp } from './math';

// ---------- Value noise / fBm ----------
function hash(x, y, seed) {
  let h = (x * 374761393 + y * 668265263 + seed * 2147483647) | 0;
  h = Math.imul(h ^ (h >>> 13), 1274126177);
  return ((h ^ (h >>> 16)) >>> 0) / 4294967296;
}
function smooth(t) { return t * t * (3 - 2 * t); }
export function valueNoise(x, y, seed) {
  const xi = Math.floor(x), yi = Math.floor(y);
  const xf = smooth(x - xi), yf = smooth(y - yi);
  const a = hash(xi, yi, seed), b = hash(xi + 1, yi, seed), c = hash(xi, yi + 1, seed), d = hash(xi + 1, yi + 1, seed);
  return a + (b - a) * xf + (c - a) * yf + (a - b - c + d) * xf * yf;
}
export function fbm(x, y, seed, oct = 4) {
  let s = 0, amp = 0.5, f = 1;
  for (let i = 0; i < oct; i++) { s += amp * valueNoise(x * f, y * f, seed + i * 17); amp *= 0.5; f *= 2; }
  return s / (1 - 0.5 ** oct);
}

// Спектральные сигнатуры (коэффициенты отражения) [blue, green, red, nir]
export const SIG = {
  healthy: [0.04, 0.11, 0.04, 0.52],
  stressed: [0.06, 0.15, 0.13, 0.30],
  soil: [0.14, 0.18, 0.22, 0.28],
  water: [0.07, 0.06, 0.03, 0.02],
};

/**
 * Синтетический мультиспектральный снимок поля (рисовые чеки + канал + дорога + зоны стресса).
 * Возвращает каналы red/nir (отражение 0..1), маски и RGBA для отображения.
 */
export function generateField(w, h, seed = 7, { stressLevel = 1, nStress = 5, parcels = 4 } = {}) {
  const r = rng(seed);
  const N = w * h;
  const red = new Float32Array(N), nir = new Float32Array(N), green = new Float32Array(N), blue = new Float32Array(N);
  const stressMask = new Float32Array(N), cover = new Float32Array(N);
  const kind = new Uint8Array(N); // 0 поле, 1 межа, 2 вода, 3 дорога
  const blobs = Array.from({ length: nStress }, () => ({
    x: r.range(0.08, 0.92) * w, y: r.range(0.12, 0.88) * h, rx: r.range(0.05, 0.12) * w, ry: r.range(0.07, 0.17) * h, k: r.range(0.6, 1),
  }));
  const pw = w / parcels, ph = h / 2;
  const parcelVigor = Array.from({ length: parcels * 2 }, () => r.range(0.62, 0.9));
  const canalY = h * 0.5 + 0.03 * h;
  const roadX = w * 0.74;
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const i = y * w + x;
      const px = Math.floor(x / pw), py = Math.floor(y / ph);
      const lx = x - px * pw, ly = y - py * ph;
      let k = 0;
      const wob = (fbm(x / 40, y / 40, seed + 3, 2) - 0.5) * 6;
      if (Math.abs(y - canalY + wob) < Math.max(2, h * 0.012)) k = 2;
      else if (Math.abs(x - roadX - wob * 0.5) < Math.max(2, w * 0.008)) k = 3;
      else if (lx < 2 || ly < 2 || lx > pw - 2 || ly > ph - 2) k = 1;
      kind[i] = k;
      let sig;
      if (k === 2) sig = SIG.water;
      else if (k === 3) sig = [0.2, 0.2, 0.21, 0.24];
      else {
        const n = fbm(x / 28, y / 28, seed, 4);
        const rows = 0.06 * Math.sin((x + y * 0.15) * 0.9);
        let f = k === 1 ? 0.08 : clamp(parcelVigor[px + parcels * py] * (0.8 + 0.4 * n) + rows, 0.05, 0.97);
        let s = 0;
        for (const b of blobs) {
          const d = ((x - b.x) / b.rx) ** 2 + ((y - b.y) / b.ry) ** 2;
          const nn = fbm(x / 15, y / 15, seed + 9, 3);
          s = Math.max(s, b.k * Math.exp(-d * (1.3 + 0.8 * nn)));
        }
        s = clamp(s * stressLevel, 0, 1);
        if (k === 1) s = 0;
        f = f * (1 - 0.45 * s);
        stressMask[i] = s;
        cover[i] = f;
        sig = SIG.healthy.map((hv, j) => f * (hv + (SIG.stressed[j] - hv) * s) + (1 - f) * SIG.soil[j]);
      }
      const jitter = 1 + (hash(x, y, seed + 99) - 0.5) * 0.06;
      blue[i] = sig[0] * jitter; green[i] = sig[1] * jitter; red[i] = sig[2] * jitter; nir[i] = sig[3] * jitter;
    }
  }
  const rgba = new Uint8ClampedArray(N * 4);
  for (let i = 0; i < N; i++) {
    rgba[i * 4] = clamp(Math.pow(red[i] * 3.0, 0.8) * 255, 0, 255);
    rgba[i * 4 + 1] = clamp(Math.pow(green[i] * 3.6, 0.8) * 255, 0, 255);
    rgba[i * 4 + 2] = clamp(Math.pow(blue[i] * 3.0, 0.8) * 255, 0, 255);
    rgba[i * 4 + 3] = 255;
  }
  return { w, h, red, nir, green, blue, rgba, stressMask, cover, kind, blobs };
}

export function putRGBA(canvas, rgba, w, h) {
  if (!canvas) return;
  canvas.width = w; canvas.height = h;
  const ctx = canvas.getContext('2d');
  ctx.putImageData(new ImageData(new Uint8ClampedArray(rgba), w, h), 0, 0);
}

// Простая сцена «вид сверху» в оттенках серого для обработки изображений
export function grayScene(w, h, seed = 3) {
  const g = new Float32Array(w * h);
  for (let y = 0; y < h; y++)
    for (let x = 0; x < w; x++) {
      let v = 0.35 + 0.35 * fbm(x / 30, y / 30, seed, 4);
      if (Math.abs(y - h * 0.55 - Math.sin(x / 40) * 8) < 4) v = 0.85; // дорога
      if (x > w * 0.12 && x < w * 0.3 && y > h * 0.15 && y < h * 0.38) v = 0.15; // здание
      if ((x - w * 0.7) ** 2 + (y - h * 0.25) ** 2 < 400) v = 0.62; // круглый объект
      for (let k = 0; k < 4; k++) if (x > w * (0.45 + k * 0.1) && x < w * (0.45 + k * 0.1) + 10 && y > h * 0.72 && y < h * 0.82) v = 0.95; // автомобили
      g[y * w + x] = v;
    }
  return g;
}
