'use client';
import { useEffect, useMemo, useRef, useState } from 'react';
import { StageHeader, SectionTitle, Pager, Metric, Code, Slider, Seg } from '@/components/UI';
import { LineChart, BarChart } from '@/components/Charts';
import { ndvi, ndviColor, fmt, mean, rng, enuToGeodetic } from '@/lib/math';
import { generateField, SIG } from '@/lib/scene';
import { AREA } from '@/lib/pages';

const W = 480, H = 300, ZX = 8, ZY = 5;
const CLASS_INFO = [
  { name: 'Почва / вода', color: [139, 90, 43] },
  { name: 'Сильный стресс', color: [239, 68, 68] },
  { name: 'Умеренный стресс', color: [245, 158, 11] },
  { name: 'Здоровая растительность', color: [34, 197, 94] },
];

// k-means по сглаженному NDVI (k = 8) + правила именования кластеров по их центрам
function kmeans(vals, k = 8, iters = 40, seed = 3) {
  const r = rng(seed);
  const n = vals.length;
  const sample = Array.from({ length: Math.min(8000, n) }, () => vals[Math.floor(r() * n)]).sort((a, b) => a - b);
  let C = Array.from({ length: k }, (_, j) => sample[Math.floor(((j + 0.5) / k) * sample.length)]);
  for (let it = 0; it < iters; it++) {
    const sum = new Array(k).fill(0), cnt = new Array(k).fill(0);
    for (const v of sample) {
      let b = 0;
      for (let j = 1; j < k; j++) if (Math.abs(v - C[j]) < Math.abs(v - C[b])) b = j;
      sum[b] += v; cnt[b]++;
    }
    C = C.map((c, j) => (cnt[j] ? sum[j] / cnt[j] : c));
  }
  C.sort((a, b) => a - b);
  const toClass = (c) => (c < 0.15 ? 0 : c < 0.3 ? 1 : c < 0.45 ? 2 : 3);
  const labels = new Uint8Array(n);
  for (let i = 0; i < n; i++) {
    let b = 0;
    for (let j = 1; j < k; j++) if (Math.abs(vals[i] - C[j]) < Math.abs(vals[i] - C[b])) b = j;
    labels[i] = toClass(C[b]);
  }
  return { labels, centers: C };
}
function boxBlur(v, w, h, r = 3) {
  const I = new Float64Array((w + 1) * (h + 1));
  for (let y = 0; y < h; y++) { let row = 0; for (let x = 0; x < w; x++) { row += v[y * w + x]; I[(y + 1) * (w + 1) + x + 1] = I[y * (w + 1) + x + 1] + row; } }
  const out = new Float32Array(w * h);
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
    const x0 = Math.max(0, x - r), x1 = Math.min(w, x + r + 1), y0 = Math.max(0, y - r), y1 = Math.min(h, y + r + 1);
    out[y * w + x] = (I[y1 * (w + 1) + x1] - I[y0 * (w + 1) + x1] - I[y1 * (w + 1) + x0] + I[y0 * (w + 1) + x0]) / ((x1 - x0) * (y1 - y0));
  }
  return out;
}

function spectrum(kind) {
  // кривые спектрального отражения 400–900 нм (как на рисунке 3)
  const pts = [];
  for (let l = 400; l <= 900; l += 5) {
    let v;
    const g = (mu, s) => Math.exp(-((l - mu) ** 2) / (2 * s * s));
    const edge = (mu, s) => 1 / (1 + Math.exp(-(l - mu) / s));
    if (kind === 'healthy') v = 0.03 + 0.08 * g(550, 30) - 0.01 * g(670, 20) + 0.48 * edge(715, 12);
    else if (kind === 'stressed') v = 0.04 + 0.1 * g(565, 40) + 0.06 * edge(600, 30) + 0.22 * edge(725, 18);
    else v = 0.08 + 0.00045 * (l - 400);
    pts.push([l, v]);
  }
  return pts;
}

export default function NdviPage() {
  const [src, setSrc] = useState('synthetic');
  const [seed, setSeed] = useState(12);
  const [stressLevel, setStress] = useState(1.2);
  const [thr, setThr] = useState(0.4);
  const [minFrac, setMinFrac] = useState(0.15);
  const [cmap, setCmap] = useState('color');
  const [upload, setUpload] = useState(null);
  const [uploadMode, setUploadMode] = useState('rgb');
  const [bands, setBands] = useState(null);
  const [calc, setCalc] = useState({ nir: 0.45, red: 0.08 });
  const cRGB = useRef(null), cNDVI = useRef(null), cCls = useRef(null), cZones = useRef(null);

  // Получение каналов Red / NIR
  useEffect(() => {
    if (src === 'synthetic') {
      const f = generateField(W, H, seed, { nStress: 6, stressLevel, parcels: 4 });
      setBands({ w: W, h: H, red: f.red, nir: f.nir, rgba: f.rgba, kind: 'Мультиспектральная камера (синтетический снимок): каналы Red 668 нм и NIR 842 нм' });
    } else if (upload) {
      const img = new Image();
      img.onload = () => {
        const s = Math.min(1, 600 / img.width);
        const w = Math.round(img.width * s), h = Math.round(img.height * s);
        const c = document.createElement('canvas'); c.width = w; c.height = h;
        const ctx = c.getContext('2d'); ctx.drawImage(img, 0, 0, w, h);
        const d = ctx.getImageData(0, 0, w, h).data;
        const red = new Float32Array(w * h), nir = new Float32Array(w * h);
        for (let i = 0; i < w * h; i++) {
          if (uploadMode === 'nir') { nir[i] = d[i * 4] / 255; red[i] = d[i * 4 + 2] / 255; } // Infragram: NIR в R-канале, видимый — в B
          else { nir[i] = d[i * 4 + 1] / 255; red[i] = d[i * 4] / 255; } // RGB: (G − R)/(G + R) — индекс GRVI как приближение NDVI
        }
        setBands({ w, h, red, nir, rgba: new Uint8ClampedArray(d), kind: uploadMode === 'nir' ? 'NIR-камера (Infragram, синий фильтр): NDVI = (R − B)/(R + B)' : 'Обычное RGB-фото: псевдо-NDVI (GRVI) = (G − R)/(G + R)' });
      };
      img.src = upload;
    } else setBands(null);
  }, [src, seed, stressLevel, upload, uploadMode]);

  const res = useMemo(() => {
    if (!bands) return null;
    const { w, h, red, nir } = bands;
    const N = w * h;
    const v = new Float32Array(N);
    let sum = 0, mn = 1, mx = -1;
    const hist = new Array(44).fill(0); // −0.2..0.9 шаг 0.025
    for (let i = 0; i < N; i++) {
      const x = ndvi(nir[i], red[i]);
      v[i] = x; sum += x; if (x < mn) mn = x; if (x > mx) mx = x;
      const b = Math.floor((x + 0.2) / 0.025);
      if (b >= 0 && b < hist.length) hist[b]++;
    }
    const km = kmeans(boxBlur(v, w, h));
    const cnt = [0, 0, 0, 0], sumC = [0, 0, 0, 0];
    for (let i = 0; i < N; i++) { cnt[km.labels[i]]++; sumC[km.labels[i]] += v[i]; }
    // зоны съёмки (кадры камеры) и решение
    const zones = [];
    const zw = w / ZX, zh = h / ZY;
    for (let zy = 0; zy < ZY; zy++)
      for (let zx = 0; zx < ZX; zx++) {
        let s = 0, n = 0, below = 0, stressPx = 0;
        for (let y = Math.floor(zy * zh); y < Math.floor((zy + 1) * zh); y += 2)
          for (let x = Math.floor(zx * zw); x < Math.floor((zx + 1) * zw); x += 2) {
            const i = y * w + x;
            if (km.labels[i] === 0) continue; // почва не учитывается
            s += v[i]; n++;
            if (v[i] < thr) below++;
            if (km.labels[i] === 1 || km.labels[i] === 2) stressPx++;
          }
        const m = n ? s / n : 0, frac = n ? below / n : 0;
        const stressSigns = n ? stressPx / n : 0;
        const shoot = n > 0 && frac >= minFrac && stressSigns >= minFrac * 0.5;
        const e = (zx + 0.5) * 75 - 300, nn = 200 - (zy + 0.5) * 80;
        const [lat, lon] = enuToGeodetic(e, nn, 40, AREA.lat0, AREA.lon0, AREA.h0);
        zones.push({ zx, zy, m, frac, stressSigns, shoot, lat, lon, e, n: nn });
      }
    // выборка точек Red/NIR для диаграммы рассеяния
    const r = rng(2);
    const scatter = Array.from({ length: 900 }, () => { const i = Math.floor(r() * N); const c = CLASS_INFO[km.labels[i]].color; return [red[i], nir[i], `rgb(${c.join(',')})`]; });
    return { v, mean: sum / N, mn, mx, hist, km, cnt, sumC, zones, scatter };
  }, [bands, thr, minFrac]);

  // отрисовка карт
  useEffect(() => {
    if (!bands || !res) return;
    const { w, h } = bands;
    const put = (c, fn) => {
      if (!c) return;
      c.width = w; c.height = h;
      const ctx = c.getContext('2d');
      const img = ctx.createImageData(w, h);
      for (let i = 0; i < w * h; i++) { const [a, b, cc] = fn(i); img.data[i * 4] = a; img.data[i * 4 + 1] = b; img.data[i * 4 + 2] = cc; img.data[i * 4 + 3] = 255; }
      ctx.putImageData(img, 0, 0);
      return ctx;
    };
    put(cRGB.current, (i) => [bands.rgba[i * 4], bands.rgba[i * 4 + 1], bands.rgba[i * 4 + 2]]);
    put(cNDVI.current, (i) => { if (cmap === 'gray') { const g = Math.max(0, Math.min(255, ((res.v[i] - 0.1) / 0.5) * 255)); return [g, g, g]; } return ndviColor(res.v[i]); });
    put(cCls.current, (i) => CLASS_INFO[res.km.labels[i]].color);
    const ctx = put(cZones.current, (i) => { const c = ndviColor(res.v[i]); return c.map((x) => x * 0.55); });
    if (ctx) {
      const zw = w / ZX, zh = h / ZY;
      // маршрут «змейкой» по зонам
      ctx.strokeStyle = 'rgba(255,255,255,.7)'; ctx.lineWidth = 1.5; ctx.setLineDash([4, 4]); ctx.beginPath();
      for (let zy = 0; zy < ZY; zy++) for (let k = 0; k < ZX; k++) { const zx = zy % 2 ? ZX - 1 - k : k; const x = (zx + 0.5) * zw, y = (zy + 0.5) * zh; zy || k ? ctx.lineTo(x, y) : ctx.moveTo(x, y); }
      ctx.stroke(); ctx.setLineDash([]);
      res.zones.forEach((z) => {
        const x = z.zx * zw, y = z.zy * zh;
        ctx.strokeStyle = z.shoot ? '#facc15' : 'rgba(255,255,255,.25)'; ctx.lineWidth = z.shoot ? 2.5 : 1;
        ctx.strokeRect(x + 1, y + 1, zw - 2, zh - 2);
        ctx.fillStyle = '#fff'; ctx.font = '11px monospace'; ctx.fillText(z.m.toFixed(2), x + 5, y + 14);
        if (z.shoot) {
          const cx = x + zw / 2, cy = y + zh / 2 + 4;
          ctx.fillStyle = '#facc15'; ctx.fillRect(cx - 11, cy - 7, 22, 15); ctx.fillRect(cx - 5, cy - 10, 10, 4);
          ctx.fillStyle = '#1f2937'; ctx.beginPath(); ctx.arc(cx, cy, 4.5, 0, 7); ctx.fill();
        }
      });
    }
  }, [bands, res, cmap]);

  const onFile = (e) => {
    const f = e.target.files?.[0];
    if (!f) return;
    const rd = new FileReader();
    rd.onload = () => { setUpload(rd.result); setSrc('upload'); };
    rd.readAsDataURL(f);
  };

  const shots = res ? res.zones.filter((z) => z.shoot) : [];
  const total = res ? res.cnt.reduce((a, b) => a + b, 0) : 1;
  const calcV = ndvi(calc.nir, calc.red);
  return (
    <>
      <StageHeader
        title="Расчёт NDVI и интеллектуальное управление моментом съёмки (рисунок 3)"
        subtitle="Мультиспектральная камера БПЛА регистрирует каналы Red и NIR, система рассчитывает вегетационный индекс NDVI, классифицирует фитостресс методами ИИ и принимает решение: если NDVI ниже порога и есть признаки стресса — выполняется детальная съёмка зоны, иначе полёт продолжается без съёмки."
        tags={['NDVI = (NIR − Red)/(NIR + Red)', 'k-means', 'Порог + признаки стресса', 'Управление камерой']}
      />

      <div className="card">
        <svg viewBox="0 0 1000 120" className="chart">
          {[
            ['Мультиспектр. камера', 'Red 668, NIR 842 нм', '#38bdf8'],
            ['Сбор данных', 'снимки поля с БПЛА', '#38bdf8'],
            ['Расчёт NDVI', '(NIR − Red)/(NIR + Red)', '#4ade80'],
            ['Фитостресс', 'k-means + правила', '#fbbf24'],
            ['Решение', 'NDVI < порог и стресс?', '#f87171'],
            ['Да → 📷 / Нет → →', 'управление камерой', '#a78bfa'],
          ].map(([t, s, c], i) => (
            <g key={t}>
              <rect x={8 + i * 165} y={22} width={150} height={74} rx={10} fill="#111a2e" stroke={c} strokeWidth="1.8" />
              <text x={83 + i * 165} y={52} textAnchor="middle" fontSize="12.5" fontWeight="700" fill="#e6ecf7">{t}</text>
              <text x={83 + i * 165} y={72} textAnchor="middle" fontSize="11" fill="#93a2bf">{s}</text>
              {i < 5 ? <path d={`M${160 + i * 165},59 l12,0 m-5,-5 l5,5 l-5,5`} stroke="#8aa0c7" fill="none" strokeWidth="1.8" /> : null}
            </g>
          ))}
        </svg>
      </div>

      <SectionTitle n="N.1">Источник данных</SectionTitle>
      <div className="card">
        <div className="row">
          <Seg value={src} onChange={setSrc} options={[['synthetic', 'Мультиспектральный снимок поля'], ['upload', 'Загрузить своё изображение']]} />
          {src === 'synthetic' ? (
            <>
              <button className="btn small" onClick={() => setSeed((s) => s + 1)} type="button">Новый снимок</button>
              <div style={{ width: 220 }}><Slider label="Интенсивность стресса" value={stressLevel} min={0} max={1.6} step={0.1} onChange={setStress} digits={1} /></div>
            </>
          ) : (
            <>
              <input type="file" accept="image/*" onChange={onFile} style={{ width: 'auto' }} />
              <Seg value={uploadMode} onChange={setUploadMode} options={[['rgb', 'RGB-фото (GRVI)'], ['nir', 'NIR-камера (Infragram)']]} />
            </>
          )}
        </div>
        <p className="small muted" style={{ margin: '8px 0 0' }}>{bands ? bands.kind : src === 'upload' ? 'Выберите файл изображения поля (JPG/PNG).' : 'Генерация снимка…'}</p>
      </div>

      <SectionTitle n="N.2">Карты: RGB, NDVI, классификация фитостресса</SectionTitle>
      <div className="grid g3">
        <div className="card tight"><h3>Снимок (RGB)</h3><canvas ref={cRGB} style={{ width: '100%' }} /></div>
        <div className="card tight">
          <div className="row"><h3 style={{ margin: 0 }}>NDVI</h3><span className="spacer" /><Seg value={cmap} onChange={setCmap} options={[['color', 'Цвет'], ['gray', 'Серый']]} /></div>
          <canvas ref={cNDVI} style={{ width: '100%', marginTop: 6 }} />
          <div style={{ height: 10, borderRadius: 4, marginTop: 6, background: cmap === 'gray' ? 'linear-gradient(90deg,#000,#fff)' : 'linear-gradient(90deg,rgb(165,0,38),rgb(244,109,67),rgb(254,224,139),rgb(166,217,106),rgb(26,152,80))' }} />
          <div className="row small muted"><span>0,1</span><span className="spacer" /><span>0,35</span><span className="spacer" /><span>0,6</span></div>
        </div>
        <div className="card tight">
          <h3>Классификация фитостресса (k-means)</h3>
          <canvas ref={cCls} style={{ width: '100%' }} />
          <div className="legend">{CLASS_INFO.map((c) => <span key={c.name}><i style={{ background: `rgb(${c.color.join(',')})`, height: 8 }} />{c.name}</span>)}</div>
        </div>
      </div>

      {res && (
        <div className="grid g-side section">
          <div className="grid g2">
            <div className="card tight">
              <h3>Гистограмма NDVI</h3>
              <LineChart height={210} xLabel="NDVI" yLabel="пикселей" series={[{ data: res.hist.map((c, i) => [-0.2 + i * 0.025 + 0.0125, c]), color: '#4ade80', width: 2 }]} markers={[{ x: thr, label: `порог ${thr}`, color: '#f87171' }, { x: res.mean, label: 'среднее', color: '#38bdf8' }]} />
            </div>
            <div className="card tight table-wrap">
              <h3>Статистика классов</h3>
              <table>
                <thead><tr><th>Класс</th><th className="num">Площадь</th><th className="num">NDVI ср.</th></tr></thead>
                <tbody>{CLASS_INFO.map((c, i) => (
                  <tr key={c.name}><td><span style={{ display: 'inline-block', width: 10, height: 10, borderRadius: 2, background: `rgb(${c.color.join(',')})`, marginRight: 6 }} />{c.name}</td><td className="num">{fmt((res.cnt[i] / total) * 100, 1)}%</td><td className="num">{fmt(res.sumC[i] / (res.cnt[i] || 1), 3)}</td></tr>
                ))}</tbody>
              </table>
            </div>
          </div>
          <div className="metrics" style={{ alignContent: 'start' }}>
            <Metric k="NDVI средний" v={fmt(res.mean, 3)} color="#4ade80" />
            <Metric k="NDVI мин / макс" v={`${fmt(res.mn)} / ${fmt(res.mx)}`} />
            <Metric k="Площадь под стрессом" v={fmt(((res.cnt[1] + res.cnt[2]) / total) * 100, 1) + '%'} color="#fbbf24" />
          </div>
        </div>
      )}

      <SectionTitle n="N.3">Принятие решения о моменте съёмки</SectionTitle>
      <div className="grid g-side">
        <div className="card tight">
          <h3>Маршрут по зонам съёмки и решения камеры</h3>
          <canvas ref={cZones} style={{ width: '100%' }} />
          <div className="caption">Поле разбито на {ZX * ZY} зон (кадров). Цифра — средний NDVI растительности зоны. 📷 — выполнена детальная съёмка.</div>
        </div>
        <div className="card tight controls">
          <Slider label="Порог NDVI" value={thr} min={0.2} max={0.6} step={0.01} onChange={setThr} />
          <Slider label="Мин. доля пикселей ниже порога" value={minFrac} min={0.02} max={0.6} step={0.01} onChange={setMinFrac} />
          <div className="note small" style={{ margin: 0 }}>
            <b>Правило:</b> ЕСЛИ доля NDVI &lt; порога ≥ {Math.round(minFrac * 100)}% И k-means находит признаки стресса ТО съёмка, ИНАЧЕ — пролёт без съёмки.
          </div>
          {res && (
            <div className="metrics">
              <Metric k="Снимков сделано" v={`${shots.length} / ${res.zones.length}`} color="#facc15" />
              <Metric k="Экономия снимков" v={fmt((1 - shots.length / res.zones.length) * 100, 0) + '%'} color="#4ade80" />
            </div>
          )}
        </div>
      </div>
      {res && (
        <div className="card table-wrap section">
          <table>
            <thead><tr><th>Зона</th><th className="num">NDVI ср.</th><th className="num">Доля &lt; порога</th><th className="num">Признаки стресса</th><th className="num">Широта</th><th className="num">Долгота</th><th>Решение</th></tr></thead>
            <tbody>{res.zones.filter((z) => z.shoot || z.frac > minFrac * 0.5).slice(0, 14).map((z) => (
              <tr key={`${z.zx}-${z.zy}`}>
                <td>{String.fromCharCode(65 + z.zy)}{z.zx + 1}</td>
                <td className="num">{z.m.toFixed(3)}</td>
                <td className="num">{(z.frac * 100).toFixed(0)}%</td>
                <td className="num">{(z.stressSigns * 100).toFixed(0)}%</td>
                <td className="num">{z.lat.toFixed(6)}</td>
                <td className="num">{z.lon.toFixed(6)}</td>
                <td>{z.shoot ? <span className="badge warn">📷 Да — съёмка</span> : <span className="badge">Нет — продолжить</span>}</td>
              </tr>
            ))}</tbody>
          </table>
        </div>
      )}

      <SectionTitle n="N.4">Спектральные характеристики</SectionTitle>
      <div className="grid g3">
        <div className="card tight">
          <h3>Спектральное отражение</h3>
          <LineChart height={220} xLabel="длина волны, нм" yLabel="отражение" yMin={0} yMax={0.6} series={[
            { name: 'здоровое растение', data: spectrum('healthy'), color: '#22c55e' },
            { name: 'стресс', data: spectrum('stressed'), color: '#f97316' },
            { name: 'почва', data: spectrum('soil'), color: '#a16207' },
          ]} areas={[{ x0: 650, x1: 680, color: '#ef444433' }, { x0: 825, x1: 860, color: '#a78bfa33' }]} />
          <p className="small muted" style={{ margin: 0 }}>Красная полоса поглощается хлорофиллом, NIR отражается клеточной структурой листа. При стрессе Red растёт, NIR падает — NDVI снижается.</p>
        </div>
        <div className="card tight">
          <h3>Red ↔ NIR (пиксели снимка)</h3>
          {res && <LineChart height={220} xLabel="Red" yLabel="NIR" xMin={0} yMin={0} series={[{ data: res.scatter, color: '#4ade80', scatter: true, r: 1.8, opacity: 0.7 }, { name: 'линия почвы', data: [[0, 0.02], [0.3, 0.38]], color: '#a16207', dash: '5 4' }]} />}
          <p className="small muted" style={{ margin: 0 }}>Здоровая растительность — высокий NIR при низком Red; почва — вдоль «линии почвы».</p>
        </div>
        <div className="card tight controls">
          <h3>Калькулятор NDVI</h3>
          <Slider label="NIR (отражение)" value={calc.nir} min={0} max={0.7} step={0.01} onChange={(v) => setCalc((c) => ({ ...c, nir: v }))} />
          <Slider label="Red (отражение)" value={calc.red} min={0} max={0.4} step={0.01} onChange={(v) => setCalc((c) => ({ ...c, red: v }))} />
          <div className="metric">
            <div className="v" style={{ color: `rgb(${ndviColor(calcV).join(',')})` }}>{calcV.toFixed(3)}</div>
            <div className="k">NDVI = ({calc.nir.toFixed(2)} − {calc.red.toFixed(2)}) / ({calc.nir.toFixed(2)} + {calc.red.toFixed(2)}) → {calcV < 0.15 ? 'почва/вода' : calcV < 0.3 ? 'сильный стресс' : calcV < 0.45 ? 'умеренный стресс' : 'здоровая растительность'}</div>
          </div>
          <p className="small muted" style={{ margin: 0 }}>Эталон: здоровый лист Red = {SIG.healthy[2]}, NIR = {SIG.healthy[3]} → NDVI = {ndvi(SIG.healthy[3], SIG.healthy[2]).toFixed(2)}; стресс → {ndvi(SIG.stressed[3], SIG.stressed[2]).toFixed(2)}; почва → {ndvi(SIG.soil[3], SIG.soil[2]).toFixed(2)}.</p>
        </div>
      </div>

      <SectionTitle n="N.5">Программная реализация</SectionTitle>
      <Code title="ndvi_camera_control.py" code={`
import numpy as np, rasterio
from sklearn.cluster import KMeans

with rasterio.open('field_multispectral.tif') as ds:      # каналы: Green, Red, RedEdge, NIR
    red = ds.read(2).astype('float32') / 65535
    nir = ds.read(4).astype('float32') / 65535

ndvi = (nir - red) / (nir + red + 1e-6)                  # Расчёт NDVI
print(f'NDVI: mean={ndvi.mean():.3f} min={ndvi.min():.3f} max={ndvi.max():.3f}')

# Классификация фитостресса: k-means по сглаженному NDVI, имена кластеров — по их центрам
from scipy.ndimage import uniform_filter
X = uniform_filter(ndvi, 7).reshape(-1, 1)
km = KMeans(n_clusters=8, n_init=10, random_state=0).fit(X[::20])
name = np.digitize(km.cluster_centers_[:, 0], [0.15, 0.30, 0.45])   # 0-почва 1-сильный 2-умеренный 3-норма
labels = name[km.predict(X)].reshape(ndvi.shape)

# Принятие решения по зонам (кадрам) маршрута
THR, MIN_FRAC = ${thr.toFixed(2)}, ${minFrac.toFixed(2)}
def decide(zone_ndvi, zone_labels):
    veg = zone_labels > 0
    frac_low = (zone_ndvi[veg] < THR).mean() if veg.any() else 0
    stress = np.isin(zone_labels[veg], [1, 2]).mean() if veg.any() else 0
    return frac_low >= MIN_FRAC and stress >= MIN_FRAC / 2

for zone in survey_zones(ndvi.shape, nx=${ZX}, ny=${ZY}):
    if decide(ndvi[zone.slice], labels[zone.slice]):
        drone.goto(zone.lat, zone.lon, alt=25)              # снижение для детальной съёмки
        camera.trigger()                                     # Да -> съёмка
    # Нет -> продолжение полёта без съёмки
`} />
      <div className="note section">
        <b>Вывод.</b> NDVI чётко разделяет здоровую растительность (0,5–0,8), растения в стрессе (0,2–0,45) и почву (&lt; 0,15). Классификация k-means выделяет зоны фитостресса без ручной разметки, а правило «NDVI ниже порога + признаки стресса» позволяет делать детальные снимки только проблемных зон — число снимков и объём передаваемых данных сокращаются в несколько раз.
      </div>
      <details className="card section">
        <summary style={{ cursor: 'pointer', fontWeight: 600 }}>Исходный рисунок 3 из задания</summary>
        <div className="figure" style={{ marginTop: 12, maxWidth: 760 }}><img src="/figures/fig3.png" alt="Рисунок 3" /></div>
      </details>
      <Pager href="/ndvi" />
    </>
  );
}
