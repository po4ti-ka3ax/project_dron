'use client';
import { useEffect, useMemo, useRef, useState } from 'react';
import { StageHeader, Tasks, SectionTitle, Pager, Slider, Seg, Metric, Code } from '@/components/UI';
import { LineChart, BarChart } from '@/components/Charts';
import { rng, kalmanCV, polyfit, polyval, rmse, mean, std, fmt } from '@/lib/math';
import { grayScene } from '@/lib/scene';

const W = 300, H = 190;

// ---------- Фильтры изображений (аналоги cv2.blur / GaussianBlur / medianBlur / equalizeHist) ----------
function convolve(src, k, size) {
  const r = size >> 1, out = new Float32Array(src.length);
  for (let y = 0; y < H; y++)
    for (let x = 0; x < W; x++) {
      let s = 0;
      for (let j = -r; j <= r; j++)
        for (let i = -r; i <= r; i++) {
          const xx = Math.min(W - 1, Math.max(0, x + i)), yy = Math.min(H - 1, Math.max(0, y + j));
          s += src[yy * W + xx] * k[(j + r) * size + (i + r)];
        }
      out[y * W + x] = s;
    }
  return out;
}
const meanK = new Array(9).fill(1 / 9);
const gaussK = (() => {
  const k = [], s = 1.1;
  let sum = 0;
  for (let j = -2; j <= 2; j++) for (let i = -2; i <= 2; i++) { const v = Math.exp(-(i * i + j * j) / (2 * s * s)); k.push(v); sum += v; }
  return k.map((v) => v / sum);
})();
function median(src) {
  const out = new Float32Array(src.length), win = new Array(9);
  for (let y = 0; y < H; y++)
    for (let x = 0; x < W; x++) {
      let n = 0;
      for (let j = -1; j <= 1; j++) for (let i = -1; i <= 1; i++) win[n++] = src[Math.min(H - 1, Math.max(0, y + j)) * W + Math.min(W - 1, Math.max(0, x + i))];
      win.sort((a, b) => a - b);
      out[y * W + x] = win[4];
    }
  return out;
}
function minmax(src) {
  let lo = Infinity, hi = -Infinity;
  for (const v of src) { if (v < lo) lo = v; if (v > hi) hi = v; }
  return src.map((v) => (v - lo) / (hi - lo || 1));
}
function equalize(src) {
  const bins = new Array(256).fill(0);
  const q = src.map((v) => Math.max(0, Math.min(255, Math.round(v * 255))));
  q.forEach((v) => bins[v]++);
  let c = 0;
  const cdf = bins.map((b) => (c += b));
  const cmin = cdf.find((v) => v > 0);
  return q.map((v) => (cdf[v] - cmin) / (src.length - cmin));
}
function psnr(a, b) {
  let s = 0;
  for (let i = 0; i < a.length; i++) s += (Math.min(1, Math.max(0, a[i])) - Math.min(1, Math.max(0, b[i]))) ** 2;
  const mse = s / a.length;
  return 10 * Math.log10(1 / (mse || 1e-12));
}
function drawGray(canvas, g) {
  if (!canvas) return;
  canvas.width = W; canvas.height = H;
  const ctx = canvas.getContext('2d');
  const img = ctx.createImageData(W, H);
  for (let i = 0; i < g.length; i++) {
    const v = Math.max(0, Math.min(255, g[i] * 255));
    img.data[i * 4] = v; img.data[i * 4 + 1] = v; img.data[i * 4 + 2] = v; img.data[i * 4 + 3] = 255;
  }
  ctx.putImageData(img, 0, 0);
}

function ImageDemo() {
  const [sigma, setSigma] = useState(0.08);
  const [sp, setSp] = useState(0.04);
  const [filter, setFilter] = useState('median');
  const [norm, setNorm] = useState('minmax');
  const c1 = useRef(null), c2 = useRef(null), c3 = useRef(null);
  const [stats, setStats] = useState(null);
  const clean = useMemo(() => grayScene(W, H, 5), []);
  useEffect(() => {
    const r = rng(42);
    const noisy = clean.map((v) => {
      const u = r();
      if (u < sp / 2) return 0;
      if (u < sp) return 1;
      return v + r.gauss(0, sigma);
    });
    const t0 = performance.now();
    let f = noisy;
    if (filter === 'mean') f = convolve(noisy, meanK, 3);
    if (filter === 'gauss') f = convolve(noisy, gaussK, 5);
    if (filter === 'median') f = median(noisy);
    if (filter === 'median+gauss') f = convolve(median(noisy), gaussK, 5);
    const filtered = f;
    if (norm === 'minmax') f = minmax(f);
    if (norm === 'hist') f = equalize(f);
    const ms = performance.now() - t0;
    drawGray(c1.current, clean); drawGray(c2.current, noisy); drawGray(c3.current, f);
    setStats({ pNoisy: psnr(clean, noisy), pFilt: psnr(clean, filtered), ms, m: mean(Array.from(f)), s: std(Array.from(f)) });
  }, [sigma, sp, filter, norm, clean]);
  return (
    <div className="grid g-side">
      <div>
        <div className="grid g3">
          <div><canvas ref={c1} style={{ width: '100%' }} /><div className="caption">Исходный кадр</div></div>
          <div><canvas ref={c2} style={{ width: '100%' }} /><div className="caption">С шумом (гаусс + «соль-перец»)</div></div>
          <div><canvas ref={c3} style={{ width: '100%' }} /><div className="caption">После фильтрации и нормализации</div></div>
        </div>
        {stats ? (
          <div className="metrics" style={{ marginTop: 12 }}>
            <Metric k="PSNR зашумлённого, дБ" v={fmt(stats.pNoisy, 1)} color="#f87171" />
            <Metric k="PSNR после фильтра, дБ" v={fmt(stats.pFilt, 1)} color="#4ade80" />
            <Metric k="Выигрыш, дБ" v={'+' + fmt(stats.pFilt - stats.pNoisy, 1)} />
            <Metric k="Время обработки, мс" v={fmt(stats.ms, 1)} />
            <Metric k="μ / σ после нормализации" v={`${fmt(stats.m)} / ${fmt(stats.s)}`} />
          </div>
        ) : null}
      </div>
      <div className="card tight controls">
        <Slider label="Гауссов шум σ" value={sigma} min={0} max={0.25} step={0.01} onChange={setSigma} />
        <Slider label="Импульсный шум, доля" value={sp} min={0} max={0.15} step={0.01} onChange={setSp} />
        <div className="small muted">Фильтр (удаление шума)</div>
        <Seg value={filter} onChange={setFilter} options={[['none', 'Нет'], ['mean', 'Среднее'], ['gauss', 'Гаусс'], ['median', 'Медиана'], ['median+gauss', 'Мед.+Гаусс']]} />
        <div className="small muted">Нормализация</div>
        <Seg value={norm} onChange={setNorm} options={[['none', 'Нет'], ['minmax', 'Min-Max'], ['hist', 'Эквализация гистограммы']]} />
        <p className="small muted" style={{ margin: 0 }}>Медианный фильтр лучше всего удаляет импульсный шум, гауссов — флуктуационный; их комбинация даёт наибольший PSNR.</p>
      </div>
    </div>
  );
}

// ---------- Коррекция GPS ----------
export function makeTrack(n, sigma, pOut, seed) {
  const r = rng(seed);
  const truth = [], meas = [];
  for (let t = 0; t < n; t++) {
    const x = 120 * Math.sin(t / 38) + t * 0.9, y = 80 * Math.sin(t / 19) * Math.cos(t / 57);
    truth.push([x, y]);
    let mx = x + r.gauss(0, sigma), my = y + r.gauss(0, sigma);
    if (r() < pOut) { mx += r.gauss(0, 30); my += r.gauss(0, 30); }
    meas.push([mx, my]);
  }
  return { truth, meas };
}
export function rejectOutliers(meas, win = 3, k = 3.5) {
  // остаток относительно скользящей медианы, порог — k·MAD по всей серии
  const med = (arr) => { const q = [...arr].sort((x, y) => x - y); return q[q.length >> 1]; };
  const res = meas.map((p, i) => {
    const w = meas.slice(Math.max(0, i - win), i + win + 1);
    return [p[0] - med(w.map((q) => q[0])), p[1] - med(w.map((q) => q[1]))];
  });
  const mad = [0, 1].map((d) => med(res.map((r) => Math.abs(r[d]))) * 1.4826 + 1e-6);
  const bad = res.map((r) => Math.abs(r[0]) > k * mad[0] || Math.abs(r[1]) > k * mad[1]);
  const out = meas.map((p) => p.slice());
  const flags = [];
  for (let i = 0; i < meas.length; i++) {
    if (!bad[i]) continue;
    flags.push(i);
    let a = i - 1, b = i + 1;
    while (a >= 0 && bad[a]) a--;
    while (b < meas.length && bad[b]) b++;
    const A = meas[Math.max(0, a)], B = meas[Math.min(meas.length - 1, b)];
    const t = a < 0 || b >= meas.length ? 0.5 : (i - a) / (b - a);
    out[i] = [A[0] + (B[0] - A[0]) * t, A[1] + (B[1] - A[1]) * t];
  }
  return { out, flags };
}
export function movingAvg(m, w = 5) {
  return m.map((_, i) => {
    const s = m.slice(Math.max(0, i - w), i + w + 1);
    return [mean(s.map((p) => p[0])), mean(s.map((p) => p[1]))];
  });
}
export function polySmooth(m, w = 10, deg = 2) {
  return m.map((_, i) => {
    const lo = Math.max(0, i - w), hi = Math.min(m.length, i + w + 1);
    const ts = [];
    for (let t = lo; t < hi; t++) ts.push(t - i);
    const cx = polyfit(ts, m.slice(lo, hi).map((p) => p[0]), deg);
    const cy = polyfit(ts, m.slice(lo, hi).map((p) => p[1]), deg);
    return [polyval(cx, 0), polyval(cy, 0)];
  });
}
const err = (a, b) => a.map((p, i) => Math.hypot(p[0] - b[i][0], p[1] - b[i][1]));

function GpsDemo() {
  const [sigma, setSigma] = useState(4);
  const [pOut, setPOut] = useState(0.05);
  const res = useMemo(() => {
    const { truth, meas } = makeTrack(240, sigma, pOut, 11);
    const { out: clean, flags } = rejectOutliers(meas);
    const ma = movingAvg(clean);
    const pr = polySmooth(clean);
    const kf = kalmanCV(clean, 1, 0.5, sigma * sigma);
    // двухпроходный (прямой + обратный) фильтр Калмана — сглаживание при постобработке
    const kb = kalmanCV(clean.slice().reverse(), 1, 0.5, sigma * sigma).reverse();
    const kf2 = kf.map((p, i) => [(p[0] + kb[i][0]) / 2, (p[1] + kb[i][1]) / 2]);
    const e = (a) => Math.sqrt(mean(err(a, truth).map((v) => v * v)));
    return { truth, meas, kf, pr, flags, rm: { raw: e(meas), clean: e(clean), ma: e(ma), pr: e(pr), kf: e(kf), kf2: e(kf2) }, errs: { raw: err(meas, truth), kf: err(kf, truth), pr: err(pr, truth) } };
  }, [sigma, pOut]);
  return (
    <div className="grid g-side">
      <div className="grid g2">
        <div className="card tight">
          <h3>Траектория (м, локальная СК)</h3>
          <LineChart
            height={260}
            series={[
              { name: 'Сырые GPS', data: res.meas, color: '#f87171', scatter: true, r: 1.8, opacity: 0.55 },
              { name: 'Истинная', data: res.truth, color: '#e6ecf7', width: 1.5, dash: '4 3' },
              { name: 'Калман', data: res.kf, color: '#4ade80', width: 2 },
            ]}
          />
        </div>
        <div className="card tight">
          <h3>Ошибка положения во времени, м</h3>
          <LineChart
            height={260}
            xLabel="t, с"
            series={[
              { name: 'Сырые', data: res.errs.raw.map((v, i) => [i, v]), color: '#f87171', width: 1 },
              { name: 'Полином. регрессия', data: res.errs.pr.map((v, i) => [i, v]), color: '#a78bfa', width: 1.5 },
              { name: 'Калман', data: res.errs.kf.map((v, i) => [i, v]), color: '#4ade80', width: 1.5 },
            ]}
          />
        </div>
      </div>
      <div className="card tight controls">
        <Slider label="СКО шума GPS, м" value={sigma} min={1} max={10} step={0.5} onChange={setSigma} digits={1} />
        <Slider label="Доля выбросов (многолучёвость)" value={pOut} min={0} max={0.2} step={0.01} onChange={setPOut} />
        <div className="small muted">RMSE положения, м</div>
        <BarChart
          width={330}
          height={200}
          items={[
            { label: 'Сырые данные', value: res.rm.raw, color: '#f87171' },
            { label: 'Отсев выбросов', value: res.rm.clean, color: '#fb923c' },
            { label: 'Скольз. среднее', value: res.rm.ma, color: '#fbbf24' },
            { label: 'Полином. регрессия', value: res.rm.pr, color: '#a78bfa' },
            { label: 'Калман (реальн. время)', value: res.rm.kf, color: '#4ade80' },
            { label: 'Калман двухпроходный', value: res.rm.kf2, color: '#22d3ee' },
          ]}
        />
        <div className="small">Найдено и исправлено выбросов: <b>{res.flags.length}</b>. Точность повышена в <b>{fmt(res.rm.raw / res.rm.kf, 1)}×</b> в реальном времени и в <b>{fmt(res.rm.raw / Math.min(res.rm.pr, res.rm.kf2), 1)}×</b> при постобработке. Калман в реальном времени использует только прошлые измерения, поэтому уступает сглаживателям, которые «видят» и будущие точки.</div>
      </div>
    </div>
  );
}

const NORM_ROWS = [
  ['Высота, м', [42.1, 55.3, 61.8, 48.0, 50.2]],
  ['Скорость, м/с', [6.2, 8.9, 11.4, 7.5, 9.1]],
  ['Напряжение АКБ, В', [16.4, 15.9, 15.2, 16.1, 15.6]],
  ['Температура, °C', [24.5, 27.1, 31.0, 25.8, 28.4]],
];

const CODE = `
import cv2
import numpy as np
import pandas as pd

# ---------- 1. Сбор данных ----------
cap = cv2.VideoCapture(0)                 # поток с камеры БПЛА (или 'rtsp://...')
gps = pd.read_csv('flight_log.csv')       # t, lat, lon, alt, hdop, sats

def preprocess_frame(frame):
    # 2. Удаление шума: медианный фильтр (импульсный шум) + Гаусс (флуктуационный)
    f = cv2.medianBlur(frame, 3)
    f = cv2.GaussianBlur(f, (5, 5), 1.1)
    # 3. Нормализация яркости: CLAHE по каналу L
    lab = cv2.cvtColor(f, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(l)
    f = cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)
    # 4. Масштаб под вход нейросети и нормализация [0, 1]
    x = cv2.resize(f, (640, 640)).astype(np.float32) / 255.0
    return x

# ---------- Коррекция GPS ----------
def reject_outliers(s, win=7, k=3.0):
    med = s.rolling(2 * win + 1, center=True, min_periods=1).median()
    mad = (s - med).abs().rolling(2 * win + 1, center=True, min_periods=1).median() * 1.4826
    bad = (s - med).abs() > k * mad.clip(lower=1e-6)
    return s.mask(bad).interpolate(), bad.sum()

gps['lat'], n1 = reject_outliers(gps['lat'])
gps['lon'], n2 = reject_outliers(gps['lon'])
gps = gps[gps['hdop'] < 2.5]              # отбрасываем эпохи с плохой геометрией

# z-нормализация телеметрии перед обучением моделей
num = ['alt', 'speed', 'voltage', 'temp']
gps[num] = (gps[num] - gps[num].mean()) / gps[num].std()
`;

export default function Stage1() {
  const norm = NORM_ROWS.map(([name, v]) => {
    const lo = Math.min(...v), hi = Math.max(...v), m = mean(v), s = std(v);
    return { name, v, mm: v.map((x) => (x - lo) / (hi - lo)), z: v.map((x) => (x - m) / s) };
  });
  return (
    <>
      <StageHeader
        num={1}
        title="Подготовка данных и программное обеспечение"
        subtitle="Сбор входных данных с камеры, карты и GPS, описание используемых датасетов, удаление шума, нормализация, обработка изображений (OpenCV) и коррекция ошибок GPS."
        tags={['OpenCV', 'Медианный/гауссов фильтр', 'Min-Max / Z-score', 'Полиномиальная регрессия', 'Фильтр Калмана']}
      />
      <div className="grid g2">
        <div className="card"><h3>Подготовка</h3><Tasks items={['Сбор данных с камер и датчиков', 'Применение датасетов', 'Удаление шума (фильтрация)', 'Нормализация данных']} /></div>
        <div className="card"><h3>Выполняемые задачи</h3><Tasks items={['Обработка изображений (OpenCV)', 'Коррекция данных GPS', 'Коррекция ошибок (отсев выбросов)']} /></div>
      </div>

      <SectionTitle n="1.1">Процесс сбора входных данных</SectionTitle>
      <div className="grid g3">
        {[
          ['📷 Поток данных с камеры', 'RGB 1920×1080 @ 30 к/с + мультиспектральная камера (Green, Red, Red Edge, NIR) 1,2 Мп', 'H.264 / RTSP, TIFF по каналам', 'Детекция объектов (YOLOv8), SLAM, NDVI'],
          ['🗺 Карта, положение, цель', 'Векторная карта (OSM), ЦМР SRTM 30 м, границы полей, точки цели миссии', 'GeoJSON, GeoTIFF', 'Генерация траектории, учёт препятствий, бесполётные зоны'],
          ['🛰 Данные GPS', 'GNSS-приёмник 10 Гц: широта, долгота, высота, HDOP, число спутников; ИНС 200 Гц', 'NMEA-0183 / MAVLink', 'Навигация, преобразование координат, привязка снимков'],
        ].map(([t, d, f, u]) => (
          <div className="card" key={t}>
            <h3>{t}</h3>
            <p className="small">{d}</p>
            <dl className="kv small"><dt>Формат</dt><dd>{f}</dd><dt>Применение</dt><dd style={{ fontFamily: 'inherit' }}>{u}</dd></dl>
          </div>
        ))}
      </div>

      <SectionTitle n="1.2">Датасеты и особенности их применения</SectionTitle>
      <div className="card table-wrap">
        <table>
          <thead><tr><th>Датасет</th><th>Тип данных</th><th>Объём</th><th>Где используется</th><th>Особенности применения</th></tr></thead>
          <tbody>
            <tr><td><b>VisDrone2019-DET</b></td><td>Снимки с БПЛА, bbox, 10 классов</td><td>≈10 тыс. изображений</td><td>Этап 4 — дообучение YOLOv8</td><td>Мелкие объекты, разная высота съёмки → вход 1280 px, аугментация масштаба</td></tr>
            <tr><td><b>UAVid</b></td><td>Видео с БПЛА, семантическая сегментация</td><td>300 кадров 4K, 8 классов</td><td>Этап 5 — классы карты</td><td>Наклонная съёмка, нужна коррекция перспективы</td></tr>
            <tr><td><b>EuroSAT (Sentinel-2)</b></td><td>13 спектральных каналов, 10 классов земель</td><td>27 тыс. патчей 64×64</td><td>Этап 7, NDVI — классификация ДЗЗ (Random Forest)</td><td>Каналы B4 (Red) и B8 (NIR) → NDVI, разное разрешение каналов</td></tr>
            <tr><td><b>Sentinel-2 L2A</b> (район Кызылорды)</td><td>Отражение поверхности, 10 м</td><td>Временной ряд за сезон</td><td>Прогноз NDVI (CNN–LSTM)</td><td>Маскирование облаков по слою SCL</td></tr>
            <tr><td><b>Лог полёта БПЛА</b></td><td>GPS/ИНС/телеметрия, MAVLink</td><td>10–200 Гц</td><td>Этапы 1, 3, 6, программа (EKF)</td><td>Выбросы из-за многолучёвости, пропуски сигнала</td></tr>
            <tr><td><b>OpenStreetMap + SRTM</b></td><td>Вектор + рельеф</td><td>—</td><td>Этап 2 — препятствия, этап 5</td><td>Требуется перевод в локальную СК (этап 6)</td></tr>
          </tbody>
        </table>
      </div>

      <SectionTitle n="1.3">Обработка изображений: удаление шума и нормализация</SectionTitle>
      <ImageDemo />

      <SectionTitle n="1.4">Коррекция данных GPS и коррекция ошибок</SectionTitle>
      <p className="muted">Выбросы (многолучёвость) выявляются по медиане и MAD и заменяются интерполяцией; затем трек сглаживается. Сравниваются скользящее среднее, полиномиальная регрессия (ML, МНК в скользящем окне, степень 2) и фильтр Калмана.</p>
      <GpsDemo />

      <SectionTitle n="1.5">Нормализация телеметрии</SectionTitle>
      <div className="card table-wrap">
        <table>
          <thead><tr><th>Признак</th><th>Исходные значения</th><th>Min-Max [0..1]</th><th>Z-score</th></tr></thead>
          <tbody>
            {norm.map((r) => (
              <tr key={r.name}>
                <td>{r.name}</td>
                <td className="num">{r.v.join('  ')}</td>
                <td className="num">{r.mm.map((x) => x.toFixed(2)).join('  ')}</td>
                <td className="num">{r.z.map((x) => x.toFixed(2)).join('  ')}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <SectionTitle n="1.6">Программная реализация (Python + OpenCV)</SectionTitle>
      <Code code={CODE} title="preprocessing.py" />
      <div className="note section">
        <b>Вывод.</b> Комбинация медианного и гауссова фильтров повышает PSNR кадра на 8–12 дБ, а отсев выбросов + фильтр Калмана снижают ошибку GPS примерно в 2 раза в реальном времени и в 3 раза при постобработке (полиномиальная регрессия, двухпроходный Калман). Нормализованные данные подаются на вход моделей следующих этапов.
      </div>
      <Pager href="/etap-1" />
    </>
  );
}
