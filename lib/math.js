// ---------- Случайные числа (детерминированные, чтобы SSR и клиент совпадали) ----------
export function rng(seed = 1) {
  let a = seed >>> 0;
  const next = () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
  next.gauss = (mu = 0, sigma = 1) => {
    let u = 0, v = 0;
    while (u === 0) u = next();
    while (v === 0) v = next();
    return mu + sigma * Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
  };
  next.range = (lo, hi) => lo + (hi - lo) * next();
  next.int = (lo, hi) => Math.floor(lo + (hi - lo + 1) * next());
  return next;
}

// ---------- Статистика ----------
export const clamp = (x, lo, hi) => Math.max(lo, Math.min(hi, x));
export const mean = (a) => a.reduce((s, x) => s + x, 0) / (a.length || 1);
export const std = (a) => {
  const m = mean(a);
  return Math.sqrt(a.reduce((s, x) => s + (x - m) ** 2, 0) / (a.length || 1));
};
export const rmse = (a, b) => Math.sqrt(mean(a.map((x, i) => (x - b[i]) ** 2)));
export const fmt = (x, d = 2) => (Number.isFinite(x) ? x.toFixed(d) : '—');

// ---------- Матрицы (массивы массивов) ----------
export const zeros = (r, c) => Array.from({ length: r }, () => new Array(c).fill(0));
export const eye = (n, s = 1) => {
  const m = zeros(n, n);
  for (let i = 0; i < n; i++) m[i][i] = s;
  return m;
};
export const T = (A) => A[0].map((_, j) => A.map((r) => r[j]));
export function mul(A, B) {
  const r = A.length, c = B[0].length, k = B.length;
  const C = zeros(r, c);
  for (let i = 0; i < r; i++)
    for (let p = 0; p < k; p++) {
      const a = A[i][p];
      if (a === 0) continue;
      for (let j = 0; j < c; j++) C[i][j] += a * B[p][j];
    }
  return C;
}
export const add = (A, B) => A.map((r, i) => r.map((x, j) => x + B[i][j]));
export const sub = (A, B) => A.map((r, i) => r.map((x, j) => x - B[i][j]));
export function inv(M) {
  const n = M.length;
  const A = M.map((r, i) => [...r, ...eye(n)[i]]);
  for (let c = 0; c < n; c++) {
    let p = c;
    for (let r = c + 1; r < n; r++) if (Math.abs(A[r][c]) > Math.abs(A[p][c])) p = r;
    [A[c], A[p]] = [A[p], A[c]];
    const d = A[c][c] || 1e-12;
    for (let j = 0; j < 2 * n; j++) A[c][j] /= d;
    for (let r = 0; r < n; r++) {
      if (r === c) continue;
      const f = A[r][c];
      if (f === 0) continue;
      for (let j = 0; j < 2 * n; j++) A[r][j] -= f * A[c][j];
    }
  }
  return A.map((r) => r.slice(n));
}
export const col = (v) => v.map((x) => [x]);
export const flat = (M) => M.map((r) => r[0]);

// ---------- Фильтр Калмана (модель постоянной скорости, 2D) ----------
export function kalmanCV(meas, dt = 1, q = 0.5, r = 9) {
  // состояние [x, y, vx, vy]
  const F = [[1, 0, dt, 0], [0, 1, 0, dt], [0, 0, 1, 0], [0, 0, 0, 1]];
  const H = [[1, 0, 0, 0], [0, 1, 0, 0]];
  const g = [[dt * dt / 2, 0], [0, dt * dt / 2], [dt, 0], [0, dt]];
  const Q = mul(mul(g, eye(2, q)), T(g));
  const R = eye(2, r);
  let x = col([meas[0][0], meas[0][1], 0, 0]);
  let P = eye(4, 100);
  const out = [];
  for (const z of meas) {
    x = mul(F, x);
    P = add(mul(mul(F, P), T(F)), Q);
    if (z && Number.isFinite(z[0])) {
      const y = sub(col(z), mul(H, x));
      const S = add(mul(mul(H, P), T(H)), R);
      const K = mul(mul(P, T(H)), inv(S));
      x = add(x, mul(K, y));
      P = mul(sub(eye(4), mul(K, H)), P);
    }
    out.push([x[0][0], x[1][0]]);
  }
  return out;
}

// ---------- Расширенный фильтр Калмана (CTRV: x, y, v, ψ, ω) с численным якобианом ----------
export function ctrv(s, dt) {
  const [x, y, v, psi, w] = s;
  if (Math.abs(w) < 1e-7) return [x + v * Math.cos(psi) * dt, y + v * Math.sin(psi) * dt, v, psi, w];
  return [
    x + (v / w) * (Math.sin(psi + w * dt) - Math.sin(psi)),
    y + (v / w) * (-Math.cos(psi + w * dt) + Math.cos(psi)),
    v,
    psi + w * dt,
    w,
  ];
}
export function jacobian(f, s, eps = 1e-4) {
  const f0 = f(s);
  const J = zeros(f0.length, s.length);
  for (let j = 0; j < s.length; j++) {
    const s2 = s.slice();
    s2[j] += eps;
    const f1 = f(s2);
    for (let i = 0; i < f0.length; i++) J[i][j] = (f1[i] - f0[i]) / eps;
  }
  return J;
}
export class EKF {
  constructor(s0, { dt = 1, qPos = 0.05, qV = 0.3, qPsi = 0.01, qW = 0.02 } = {}) {
    this.s = s0.slice();
    this.P = eye(5, 10);
    this.dt = dt;
    this.Q = [[qPos, 0, 0, 0, 0], [0, qPos, 0, 0, 0], [0, 0, qV, 0, 0], [0, 0, 0, qPsi, 0], [0, 0, 0, 0, qW]];
  }
  predict() {
    const F = jacobian((s) => ctrv(s, this.dt), this.s);
    this.s = ctrv(this.s, this.dt);
    this.P = add(mul(mul(F, this.P), T(F)), this.Q);
    return [this.s[0], this.s[1]];
  }
  update(z, r) {
    const H = [[1, 0, 0, 0, 0], [0, 1, 0, 0, 0]];
    const y = [[z[0] - this.s[0]], [z[1] - this.s[1]]];
    const S = add(mul(mul(H, this.P), T(H)), eye(2, r));
    const K = mul(mul(this.P, T(H)), inv(S));
    const dx = mul(K, y);
    this.s = this.s.map((v, i) => v + dx[i][0]);
    this.P = mul(sub(eye(5), mul(K, H)), this.P);
    return [this.s[0], this.s[1]];
  }
}

// ---------- Полиномиальная регрессия (МНК) ----------
export function polyfit(xs, ys, deg) {
  const X = xs.map((x) => Array.from({ length: deg + 1 }, (_, k) => x ** k));
  const Xt = T(X);
  const A = add(mul(Xt, X), eye(deg + 1, 1e-9));
  return flat(mul(inv(A), mul(Xt, col(ys))));
}
export const polyval = (c, x) => c.reduce((s, ck, k) => s + ck * x ** k, 0);

// ---------- Геодезия (WGS-84) ----------
export const WGS84 = { a: 6378137.0, f: 1 / 298.257223563 };
WGS84.e2 = WGS84.f * (2 - WGS84.f);
WGS84.b = WGS84.a * (1 - WGS84.f);
const rad = (d) => (d * Math.PI) / 180;
const deg = (r) => (r * 180) / Math.PI;

export function geodeticToEcef(lat, lon, h) {
  const { a, e2 } = WGS84;
  const φ = rad(lat), λ = rad(lon);
  const N = a / Math.sqrt(1 - e2 * Math.sin(φ) ** 2);
  return [(N + h) * Math.cos(φ) * Math.cos(λ), (N + h) * Math.cos(φ) * Math.sin(λ), (N * (1 - e2) + h) * Math.sin(φ)];
}
export function ecefToGeodetic(X, Y, Z) {
  const { a, b, e2 } = WGS84;
  const ep2 = (a * a - b * b) / (b * b);
  const p = Math.hypot(X, Y);
  const θ = Math.atan2(Z * a, p * b);
  const λ = Math.atan2(Y, X);
  const φ = Math.atan2(Z + ep2 * b * Math.sin(θ) ** 3, p - e2 * a * Math.cos(θ) ** 3);
  const N = a / Math.sqrt(1 - e2 * Math.sin(φ) ** 2);
  const h = p / Math.cos(φ) - N;
  return [deg(φ), deg(λ), h];
}
export function ecefToEnu(X, Y, Z, lat0, lon0, h0) {
  const [X0, Y0, Z0] = geodeticToEcef(lat0, lon0, h0);
  const dx = X - X0, dy = Y - Y0, dz = Z - Z0;
  const φ = rad(lat0), λ = rad(lon0);
  const e = -Math.sin(λ) * dx + Math.cos(λ) * dy;
  const n = -Math.sin(φ) * Math.cos(λ) * dx - Math.sin(φ) * Math.sin(λ) * dy + Math.cos(φ) * dz;
  const u = Math.cos(φ) * Math.cos(λ) * dx + Math.cos(φ) * Math.sin(λ) * dy + Math.sin(φ) * dz;
  return [e, n, u];
}
export function enuToEcef(e, n, u, lat0, lon0, h0) {
  const [X0, Y0, Z0] = geodeticToEcef(lat0, lon0, h0);
  const φ = rad(lat0), λ = rad(lon0);
  const dx = -Math.sin(λ) * e - Math.sin(φ) * Math.cos(λ) * n + Math.cos(φ) * Math.cos(λ) * u;
  const dy = Math.cos(λ) * e - Math.sin(φ) * Math.sin(λ) * n + Math.cos(φ) * Math.sin(λ) * u;
  const dz = Math.cos(φ) * n + Math.sin(φ) * u;
  return [X0 + dx, Y0 + dy, Z0 + dz];
}
export const geodeticToEnu = (lat, lon, h, lat0, lon0, h0) => ecefToEnu(...geodeticToEcef(lat, lon, h), lat0, lon0, h0);
export const enuToGeodetic = (e, n, u, lat0, lon0, h0) => ecefToGeodetic(...enuToEcef(e, n, u, lat0, lon0, h0));

export function haversine(lat1, lon1, lat2, lon2) {
  const R = 6371008.8;
  const dφ = rad(lat2 - lat1), dλ = rad(lon2 - lon1);
  const h = Math.sin(dφ / 2) ** 2 + Math.cos(rad(lat1)) * Math.cos(rad(lat2)) * Math.sin(dλ / 2) ** 2;
  return 2 * R * Math.asin(Math.sqrt(h));
}

// Поперечная проекция Меркатора (UTM), ряды Крюгера
export function toUTM(lat, lon) {
  const { a, f } = WGS84;
  const zone = Math.floor((lon + 180) / 6) + 1;
  const lon0 = rad((zone - 1) * 6 - 180 + 3);
  const k0 = 0.9996;
  const n = f / (2 - f);
  const A = (a / (1 + n)) * (1 + n * n / 4 + n ** 4 / 64);
  const α = [n / 2 - (2 / 3) * n * n + (5 / 16) * n ** 3, (13 / 48) * n * n - (3 / 5) * n ** 3, (61 / 240) * n ** 3];
  const φ = rad(lat), λ = rad(lon) - lon0;
  const c = (2 * Math.sqrt(n)) / (1 + n);
  const t = Math.sinh(Math.atanh(Math.sin(φ)) - c * Math.atanh(c * Math.sin(φ)));
  const ξp = Math.atan2(t, Math.cos(λ));
  const ηp = Math.atanh(Math.sin(λ) / Math.sqrt(1 + t * t));
  let ξ = ξp, η = ηp;
  for (let j = 1; j <= 3; j++) {
    ξ += α[j - 1] * Math.sin(2 * j * ξp) * Math.cosh(2 * j * ηp);
    η += α[j - 1] * Math.cos(2 * j * ξp) * Math.sinh(2 * j * ηp);
  }
  const E = 500000 + k0 * A * η;
  const N = (lat < 0 ? 10000000 : 0) + k0 * A * ξ;
  return { zone, hemi: lat >= 0 ? 'N' : 'S', E, N };
}

// ---------- NDVI ----------
export const ndvi = (nir, red) => (nir + red === 0 ? 0 : (nir - red) / (nir + red));
export function ndviClass(v) {
  if (v < 0.15) return { key: 'soil', name: 'Почва / вода', color: '#8b5a2b' };
  if (v < 0.3) return { key: 'severe', name: 'Сильный стресс', color: '#ef4444' };
  if (v < 0.45) return { key: 'moderate', name: 'Умеренный стресс', color: '#f59e0b' };
  return { key: 'healthy', name: 'Здоровая растительность', color: '#22c55e' };
}
// Цветовая шкала RdYlGn для NDVI в диапазоне [lo, hi]
export function ndviColor(v, lo = 0.1, hi = 0.6) {
  const t = clamp((v - lo) / (hi - lo), 0, 1);
  const stops = [[165, 0, 38], [244, 109, 67], [254, 224, 139], [166, 217, 106], [26, 152, 80]];
  const s = t * (stops.length - 1);
  const i = Math.min(stops.length - 2, Math.floor(s));
  const k = s - i;
  return stops[i].map((c, j) => Math.round(c + (stops[i + 1][j] - c) * k));
}
