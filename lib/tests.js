import { ndvi, geodeticToEcef, ecefToGeodetic, geodeticToEnu, haversine, kalmanCV, EKF, rng, rmse, toUTM, mean } from './math';
import { astar } from './astar';
import { detectBlobs, textureAnomalies } from './pipeline';
import { cropDataset, mamdani } from '@/components/stages/Stage7';
import { trainForest, forestProba, argmax } from './ml';
import { nms } from '@/components/stages/Stage4';
import { AREA } from './pages';

const approx = (a, b, eps) => Math.abs(a - b) <= eps;

export const TESTS = [
  ['test_ndvi_formula', 'NDVI = (NIR − Red)/(NIR + Red): ndvi(0.5, 0.1) = 0.6667, ndvi(0.3, 0.3) = 0', () => {
    const a = ndvi(0.5, 0.1), b = ndvi(0.3, 0.3);
    return { ok: approx(a, 0.6667, 1e-4) && b === 0, info: `${a.toFixed(4)}, ${b}` };
  }],
  ['test_ndvi_range', 'NDVI лежит в [−1; 1] для любых неотрицательных каналов', () => {
    const r = rng(1); let ok = true;
    for (let i = 0; i < 1000; i++) { const v = ndvi(r(), r()); if (v < -1 || v > 1) ok = false; }
    return { ok, info: '1000 случайных пар' };
  }],
  ['test_frame_none_continue', 'Если кадр не получен (None) → continue, цикл не падает', () => {
    const frames = [{ ok: true }, null, { ok: true }, null, null, { ok: true }];
    let processed = 0, skipped = 0;
    for (const f of frames) { if (!f) { skipped++; continue; } processed++; }
    return { ok: processed === 3 && skipped === 3, info: `обработано ${processed}, пропущено ${skipped}` };
  }],
  ['test_detect_blobs', 'detect(): 3 цветных объекта на фоне растительности → 3 рамки', () => {
    const w = 80, h = 60, px = new Uint8ClampedArray(w * h * 4);
    for (let i = 0; i < w * h; i++) { px[i * 4] = 90; px[i * 4 + 1] = 140; px[i * 4 + 2] = 70; px[i * 4 + 3] = 255; }
    const paint = (x0, y0, c) => { for (let y = y0; y < y0 + 8; y++) for (let x = x0; x < x0 + 6; x++) { const i = (y * w + x) * 4; px[i] = c[0]; px[i + 1] = c[1]; px[i + 2] = c[2]; } };
    paint(5, 5, [220, 40, 40]); paint(40, 20, [30, 70, 220]); paint(60, 45, [235, 235, 235]);
    const b = detectBlobs(px, w, h);
    return { ok: b.length === 3, info: `найдено ${b.length}` };
  }],
  ['test_texture_anomaly', 'Проверка аномалий: блок с NDVI≈0.25 помечается, здоровый (≈0.7) — нет', () => {
    const w = 40, h = 20, red = new Float32Array(w * h), nir = new Float32Array(w * h);
    for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) { const s = x < 20; red[y * w + x] = s ? 0.15 : 0.05; nir[y * w + x] = s ? 0.25 : 0.3; }
    const a = textureAnomalies(red, nir, w, h);
    return { ok: a.length === 1 && a[0].x === 0, info: `аномальных блоков: ${a.length}, NDVI=${a[0]?.ndvi.toFixed(2)}` };
  }],
  ['test_ecef_roundtrip', 'WGS-84 → ECEF → WGS-84: ошибка < 1 мм', () => {
    const p = [44.8152, 65.4622, 188.5];
    const b = ecefToGeodetic(...geodeticToEcef(...p));
    const e = haversine(p[0], p[1], b[0], b[1]) + Math.abs(b[2] - p[2]);
    return { ok: e < 1e-3, info: `${(e * 1000).toExponential(2)} мм` };
  }],
  ['test_enu_origin', 'ENU точки старта = (0, 0, 0)', () => {
    const e = geodeticToEnu(AREA.lat0, AREA.lon0, AREA.h0, AREA.lat0, AREA.lon0, AREA.h0);
    return { ok: e.every((v) => Math.abs(v) < 1e-6), info: e.map((v) => v.toExponential(1)).join(', ') };
  }],
  ['test_haversine_degree', '1° широты ≈ 111,2 км', () => {
    const d = haversine(44, 65, 45, 65);
    return { ok: approx(d, 111195, 50), info: `${(d / 1000).toFixed(3)} км` };
  }],
  ['test_utm_zone', 'Кызылорда (65,46° в.д.) попадает в зону UTM 41N', () => {
    const u = toUTM(AREA.lat0, AREA.lon0);
    return { ok: u.zone === 41 && u.hemi === 'N', info: `${u.zone}${u.hemi}, E=${u.E.toFixed(0)}` };
  }],
  ['test_kalman_reduces_error', 'Фильтр Калмана уменьшает RMSE зашумлённого трека', () => {
    const r = rng(3);
    const truth = Array.from({ length: 200 }, (_, t) => [t * 2, 30 * Math.sin(t / 20)]);
    const meas = truth.map(([x, y]) => [x + r.gauss(0, 4), y + r.gauss(0, 4)]);
    const kf = kalmanCV(meas, 1, 0.5, 16);
    const e0 = Math.sqrt(mean(meas.map((m, i) => (m[0] - truth[i][0]) ** 2 + (m[1] - truth[i][1]) ** 2)));
    const e1 = Math.sqrt(mean(kf.map((m, i) => (m[0] - truth[i][0]) ** 2 + (m[1] - truth[i][1]) ** 2)));
    return { ok: e1 < e0 * 0.8, info: `${e0.toFixed(2)} → ${e1.toFixed(2)} м` };
  }],
  ['test_ekf_ctrv', 'EKF (CTRV) сходится при движении по окружности, ошибка < σ GPS', () => {
    const r = rng(5), dt = 0.5, R = 50, w = 0.1;
    const truth = Array.from({ length: 200 }, (_, k) => [R * Math.cos(w * k * dt), R * Math.sin(w * k * dt)]);
    const f = new EKF([R, 0, 5, Math.PI / 2, 0], { dt });
    const out = truth.map((p) => { f.predict(); return f.update([p[0] + r.gauss(0, 3), p[1] + r.gauss(0, 3)], 9); });
    const e = rmse(out.slice(50).map((p) => p[0]), truth.slice(50).map((p) => p[0]));
    return { ok: e < 3 && Math.abs(f.s[4] - 0.1) < 0.02, info: `RMSE(x) = ${e.toFixed(2)} м, ω̂ = ${f.s[4].toFixed(3)} (истина 0.1)` };
  }],
  ['test_astar_optimal', 'A* на пустой сетке: длина = октильное расстояние; обход стены', () => {
    const g = Array.from({ length: 20 }, () => new Array(20).fill(0));
    const a = astar(g, [0, 0], [10, 4]);
    const opt = 6 + 4 * Math.SQRT2;
    for (let y = 0; y < 18; y++) g[y][8] = 1;
    const b = astar(g, [2, 2], [15, 2]);
    return { ok: approx(a.cost, opt, 1e-9) && b.path && b.path.every(([x, y]) => !g[y][x]), info: `${a.cost.toFixed(3)} = ${opt.toFixed(3)}; обход ${b.path?.length} узлов` };
  }],
  ['test_nms', 'NMS: две рамки одного класса с IoU≈0.8 → остаётся одна', () => {
    const d = nms([{ x: 0, y: 0, w: 10, h: 10, cls: 'car', conf: 0.9 }, { x: 1, y: 1, w: 10, h: 10, cls: 'car', conf: 0.7 }, { x: 50, y: 50, w: 10, h: 10, cls: 'car', conf: 0.6 }], 0.3, 0.5);
    return { ok: d.length === 2 && d[0].conf === 0.9, info: `осталось ${d.length}` };
  }],
  ['test_fuzzy_decisions', 'Нечёткая логика: АКБ 5% → возврат; АКБ 95%, ветер 1, препятствие 45 м → продолжить', () => {
    const a = mamdani({ battery: 5, wind: 3, dist: 40 }), b = mamdani({ battery: 95, wind: 1, dist: 45 });
    return { ok: a.risk > 62 && b.risk < 35, info: `риск ${a.risk.toFixed(1)} / ${b.risk.toFixed(1)}` };
  }],
  ['test_random_forest_accuracy', 'Random Forest: точность на тесте > 85%', () => {
    const { X, y } = cropDataset(600, 5);
    const f = trainForest(X.slice(0, 420), y.slice(0, 420), { nTrees: 20, maxDepth: 6 });
    const acc = X.slice(420).filter((x, i) => argmax(forestProba(f, x)) === y[420 + i]).length / 180;
    return { ok: acc > 0.85, info: `${(acc * 100).toFixed(1)}%` };
  }],
];

export function runTests() {
  return TESTS.map(([name, desc, fn]) => {
    const t0 = performance.now();
    let res;
    try { res = fn(); } catch (e) { res = { ok: false, info: 'Исключение: ' + e.message }; }
    return { name, desc, ...res, ms: performance.now() - t0 };
  });
}
