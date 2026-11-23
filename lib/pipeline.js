import { rng, EKF, mean, ndvi } from './math';

// ======================= detect(): поиск объектов по цвету + связные компоненты =======================
// Аналог cv2.inRange + cv2.connectedComponentsWithStats
export function detectBlobs(rgba, w, h, minArea = 14) {
  const mask = new Uint8Array(w * h);
  for (let i = 0; i < w * h; i++) {
    const r = rgba[i * 4], g = rgba[i * 4 + 1], b = rgba[i * 4 + 2];
    const mx = Math.max(r, g, b), mn = Math.min(r, g, b);
    const sat = mx ? (mx - mn) / mx : 0;
    // искусственные объекты: насыщенный красный/синий/жёлтый, очень светлые или очень тёмные пиксели
    if ((r > 150 && g < 100 && b < 100) || (b > 140 && r < 110 && b > g + 20) || (r > 200 && g > 200 && b > 200) || (mx < 65) || (r > 180 && g > 150 && b < 90 && sat > 0.5)) mask[i] = 1;
  }
  const lab = new Int32Array(w * h).fill(-1);
  const blobs = [];
  const stack = [];
  for (let s = 0; s < w * h; s++) {
    if (!mask[s] || lab[s] >= 0) continue;
    const id = blobs.length;
    let x0 = w, y0 = h, x1 = 0, y1 = 0, area = 0;
    stack.push(s); lab[s] = id;
    while (stack.length) {
      const p = stack.pop();
      const x = p % w, y = (p / w) | 0;
      area++;
      if (x < x0) x0 = x; if (x > x1) x1 = x; if (y < y0) y0 = y; if (y > y1) y1 = y;
      const nb = [p - 1, p + 1, p - w, p + w];
      for (let k = 0; k < 4; k++) {
        const q = nb[k];
        if (q < 0 || q >= w * h) continue;
        if ((k === 0 && x === 0) || (k === 1 && x === w - 1)) continue;
        if (mask[q] && lab[q] < 0) { lab[q] = id; stack.push(q); }
      }
    }
    blobs.push({ x: x0, y: y0, w: x1 - x0 + 1, h: y1 - y0 + 1, area });
  }
  return blobs.filter((b) => b.area >= minArea && b.w < w * 0.5 && b.h < h * 0.5);
}

// ======================= Проверка аномалий текстуры (NDVI + локальная дисперсия) =======================
export function textureAnomalies(red, nir, w, h, { block = 20, thr = 0.38 } = {}) {
  const out = [];
  for (let by = 0; by + block <= h; by += block)
    for (let bx = 0; bx + block <= w; bx += block) {
      const all = [];
      for (let y = by; y < by + block; y += 2) for (let x = bx; x < bx + block; x += 2) all.push(ndvi(nir[y * w + x], red[y * w + x]));
      // блоки с водой/межами/дорогой (NDVI < 0.15) не являются аномалиями растительности
      const vals = all.filter((v) => v >= 0.15);
      if (vals.length < all.length * 0.85) continue;
      const m = mean(vals);
      const sd = Math.sqrt(mean(vals.map((v) => (v - m) ** 2)));
      if (m < thr) out.push({ x: bx, y: by, w: block, h: block, ndvi: m, sd });
    }
  return out;
}

// ======================= Траектории для обучения/теста =======================
export function genTrajectory(n, seed, dt = 0.5) {
  const r = rng(seed);
  let x = 0, y = 0, psi = r.range(0, 6.28), v = r.range(6, 9);
  const a1 = r.range(0.08, 0.2), a2 = r.range(0.04, 0.12), f1 = r.range(0.02, 0.05), f2 = r.range(0.06, 0.12), ph = r.range(0, 6);
  const pts = [];
  let gx = 0, gy = 0;
  for (let k = 0; k < n; k++) {
    const t = k * dt;
    const w = a1 * Math.sin(2 * Math.PI * f1 * t + ph) + a2 * Math.sin(2 * Math.PI * f2 * t);
    v = Math.max(4, Math.min(11, v + r.gauss(0, 0.08)));
    psi += w * dt;
    gx = 0.95 * gx + r.gauss(0, 0.05); gy = 0.95 * gy + r.gauss(0, 0.05);
    x += (v * Math.cos(psi) + gx) * dt; y += (v * Math.sin(psi) + gy) * dt;
    pts.push([x, y]);
  }
  return pts;
}

// ======================= Упрощённая CNN–LSTM =======================
// Conv1D (фиксированные ядра: сглаживание, 1-я и 2-я разность, тождество) над окном приращений →
// рекуррентный слой (эхо-сеть, tanh) → обучаемый выходной слой (Adam, MSE)
const KERNELS = [[1 / 3, 1 / 3, 1 / 3], [-0.5, 0, 0.5], [1, -2, 1], [0, 0, 1]];
export class CnnLstm {
  constructor({ win = 10, units = 64, seed = 3 } = {}) {
    const r = rng(seed);
    this.win = win; this.units = units;
    this.nIn = 2 * KERNELS.length * 2; // каналы × ядра × (последний отсчёт + среднее)
    this.Win = Array.from({ length: units }, () => Array.from({ length: this.nIn }, () => r.gauss(0, 0.4)));
    this.Wr = Array.from({ length: units }, () => Array.from({ length: units }, () => (r() < 0.1 ? r.gauss(0, 1) : 0)));
    // нормировка спектрального радиуса ≈ 0.9 (оценка степенным методом)
    let v = Array.from({ length: units }, () => r());
    let lam = 1;
    for (let it = 0; it < 30; it++) {
      const u = this.Wr.map((row) => row.reduce((s, a, j) => s + a * v[j], 0));
      lam = Math.sqrt(u.reduce((s, a) => s + a * a, 0)) || 1;
      v = u.map((a) => a / lam);
    }
    this.Wr = this.Wr.map((row) => row.map((a) => (a * 0.9) / lam));
    this.leak = 0.5;
    this.scale = 1;
  }
  features(win) {
    // win: массив [dx, dy] длины this.win
    const f = [];
    for (let c = 0; c < 2; c++) {
      const s = win.map((p) => p[c] / this.scale);
      for (const k of KERNELS) {
        const conv = [];
        for (let i = 0; i + 2 < s.length; i++) conv.push(k[0] * s[i] + k[1] * s[i + 1] + k[2] * s[i + 2]);
        f.push(conv[conv.length - 1], mean(conv));
      }
    }
    return f;
  }
  // прогон по последовательности приращений; возвращает матрицу состояний [h; u; 1]
  states(deltas) {
    let h = new Array(this.units).fill(0);
    const S = [];
    for (let t = this.win; t <= deltas.length; t++) {
      const u = this.features(deltas.slice(t - this.win, t));
      const pre = this.Win.map((row, i) => row.reduce((s, a, j) => s + a * u[j], 0) + this.Wr[i].reduce((s, a, j) => s + a * h[j], 0));
      h = h.map((hi, i) => (1 - this.leak) * hi + this.leak * Math.tanh(pre[i]));
      S.push([...h, ...u, 1]);
    }
    return S;
  }
  fit(trainSeqs, valSeq, { epochs = 150, lr = 0.01, onEpoch, inputNoise = 0.3, seed = 4 } = {}) {
    const rn = rng(seed);
    const allD = trainSeqs.flatMap((s) => s.slice(1).map((p, i) => [p[0] - s[i][0], p[1] - s[i][1]]));
    this.scale = Math.sqrt(mean(allD.map((d) => d[0] ** 2 + d[1] ** 2))) || 1;
    const build = (seq) => {
      const d = seq.slice(1).map((p, i) => [p[0] - seq[i][0], p[1] - seq[i][1]]);
      // аугментация: на вход подаются зашумлённые приращения (как от оценок фильтра), цель — истинные
      const S = this.states(d.map((q) => [q[0] + rn.gauss(0, inputNoise), q[1] + rn.gauss(0, inputNoise)]));
      const X = [], Y = [];
      for (let k = 0; k < S.length - 1; k++) { X.push(S[k]); Y.push([d[k + this.win][0] / this.scale, d[k + this.win][1] / this.scale]); }
      return { X, Y };
    };
    const tr = trainSeqs.map(build);
    const X = tr.flatMap((t) => t.X), Y = tr.flatMap((t) => t.Y);
    const va = build(valSeq);
    const D = X[0].length;
    this.W = [new Array(D).fill(0), new Array(D).fill(0)];
    const m = [new Array(D).fill(0), new Array(D).fill(0)], v = [new Array(D).fill(0), new Array(D).fill(0)];
    const hist = [];
    const mse = (XX, YY) => mean(XX.map((x, i) => [0, 1].reduce((s, o) => s + (this.W[o].reduce((a, w, j) => a + w * x[j], 0) - YY[i][o]) ** 2, 0) / 2));
    for (let ep = 1; ep <= epochs; ep++) {
      for (let o = 0; o < 2; o++) {
        const g = new Array(D).fill(0);
        for (let i = 0; i < X.length; i++) {
          const e = this.W[o].reduce((a, w, j) => a + w * X[i][j], 0) - Y[i][o];
          for (let j = 0; j < D; j++) g[j] += (2 * e * X[i][j]) / X.length;
        }
        for (let j = 0; j < D; j++) {
          g[j] += 1e-4 * this.W[o][j];
          m[o][j] = 0.9 * m[o][j] + 0.1 * g[j];
          v[o][j] = 0.999 * v[o][j] + 0.001 * g[j] * g[j];
          const mh = m[o][j] / (1 - 0.9 ** ep), vh = v[o][j] / (1 - 0.999 ** ep);
          this.W[o][j] -= (lr * mh) / (Math.sqrt(vh) + 1e-8);
        }
      }
      if (ep % 5 === 0 || ep === 1) {
        const rec = { ep, train: mse(X, Y), val: mse(va.X, va.Y) };
        hist.push(rec);
        if (onEpoch) onEpoch(rec);
      }
    }
    this.history = hist;
    return hist;
  }
  predictNext(deltas, hState) {
    // онлайн-шаг: обновляет внутреннее состояние и возвращает прогноз следующего приращения (м)
    const u = this.features(deltas.slice(-this.win));
    const pre = this.Win.map((row, i) => row.reduce((s, a, j) => s + a * u[j], 0) + this.Wr[i].reduce((s, a, j) => s + a * hState[j], 0));
    const h = hState.map((hi, i) => (1 - this.leak) * hi + this.leak * Math.tanh(pre[i]));
    const z = [...h, ...u, 1];
    const out = [0, 1].map((o) => this.W[o].reduce((a, w, j) => a + w * z[j], 0) * this.scale);
    return { d: out, h };
  }
}

// ======================= Оценка: GPS / EKF / CNN–LSTM / гибрид =======================
export function evaluateHybrid(model, truth, { sigma = 3, outages = [[120, 170], [260, 300]], seed = 12, dt = 0.5 } = {}) {
  const r = rng(seed);
  const meas = truth.map((p, k) => (outages.some(([a, b]) => k >= a && k < b) ? null : [p[0] + r.gauss(0, sigma), p[1] + r.gauss(0, sigma)]));
  const init = (() => {
    const [a, b] = [truth[0], truth[1]];
    return [a[0], a[1], Math.hypot(b[0] - a[0], b[1] - a[1]) / dt, Math.atan2(b[1] - a[1], b[0] - a[0]), 0];
  })();
  const ekf = new EKF(init, { dt });
  const hyb = new EKF(init, { dt });
  const outE = [], outH = [], outN = [];
  let hN = new Array(model.units).fill(0), hH = new Array(model.units).fill(0);
  const histN = [truth[0]], histH = [truth[0]];
  let posN = truth[0].slice();
  for (let k = 0; k < truth.length; k++) {
    const z = meas[k];
    // --- только EKF ---
    ekf.predict();
    if (z) ekf.update(z, sigma * sigma);
    outE.push([ekf.s[0], ekf.s[1]]);
    // --- только нейросеть (авторегрессия по собственным оценкам, с привязкой к GPS при наличии) ---
    const dN = histN.slice(1).map((p, i) => [p[0] - histN[i][0], p[1] - histN[i][1]]);
    if (dN.length >= model.win) {
      const pr = model.predictNext(dN, hN);
      hN = pr.h;
      posN = [posN[0] + pr.d[0], posN[1] + pr.d[1]];
    } else if (z) posN = z.slice();
    if (z && dN.length >= model.win) posN = [0.8 * posN[0] + 0.2 * z[0], 0.8 * posN[1] + 0.2 * z[1]];
    histN.push(posN.slice());
    outN.push(posN.slice());
    // --- гибрид CNN–LSTM + EKF: при потере GPS прогноз сети подаётся в EKF как псевдоизмерение ---
    hyb.predict();
    const dH = histH.slice(1).map((p, i) => [p[0] - histH[i][0], p[1] - histH[i][1]]);
    if (dH.length >= model.win) {
      const pr = model.predictNext(dH, hH);
      hH = pr.h;
      if (!z) {
        const prev = histH[histH.length - 1];
        hyb.update([prev[0] + pr.d[0], prev[1] + pr.d[1]], 0.5);
      }
    }
    if (z) {
      if (k > 0 && !meas[k - 1]) hyb.P = hyb.P.map((row, i) => row.map((v, j) => v + (i === j ? (i < 2 ? 25 : 1) : 0))); // повторный захват GPS
      hyb.update(z, sigma * sigma);
    }
    histH.push([hyb.s[0], hyb.s[1]]);
    outH.push([hyb.s[0], hyb.s[1]]);
  }
  const err = (a) => a.map((p, k) => Math.hypot(p[0] - truth[k][0], p[1] - truth[k][1]));
  const inOut = (k) => outages.some(([a, b]) => k >= a && k < b);
  const rm = (e, f) => Math.sqrt(mean(e.filter((_, k) => f(k)).map((x) => x * x)));
  const eG = meas.map((m, k) => (m ? Math.hypot(m[0] - truth[k][0], m[1] - truth[k][1]) : NaN));
  const res = { meas, ekf: outE, nn: outN, hyb: outH, errs: { gps: eG, ekf: err(outE), nn: err(outN), hyb: err(outH) } };
  res.rmse = {};
  for (const k of ['ekf', 'nn', 'hyb']) res.rmse[k] = { all: rm(res.errs[k], () => true), gps: rm(res.errs[k], (i) => !inOut(i)), outage: rm(res.errs[k], inOut) };
  res.rmse.gps = { all: rm(eG.filter(Number.isFinite), () => true), gps: rm(eG.filter(Number.isFinite), () => true), outage: NaN };
  res.outages = outages;
  return res;
}
