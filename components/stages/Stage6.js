'use client';
import { useMemo, useState } from 'react';
import { StageHeader, Tasks, SectionTitle, Pager, Metric, Code, Seg, Slider } from '@/components/UI';
import { LineChart, BarChart } from '@/components/Charts';
import { rng, fmt, geodeticToEcef, ecefToGeodetic, ecefToEnu, enuToGeodetic, geodeticToEnu, toUTM, haversine, WGS84 } from '@/lib/math';
import { AREA } from '@/lib/pages';

function quantile(a, q) { const s = [...a].sort((x, y) => x - y); return s[Math.min(s.length - 1, Math.floor(q * s.length))]; }

function gnssExperiment(n, sigma, seed) {
  const r = rng(seed);
  let ce = 0, cn = 0; // общая (атмосферная) ошибка — процесс Гаусса–Маркова
  const raw = [], dgps = [], avg = [], kf = [];
  let se = 0, sn = 0;
  let xe = 0, xn = 0, P = 100;
  for (let k = 0; k < n; k++) {
    ce = 0.995 * ce + r.gauss(0, 0.12 * sigma); cn = 0.995 * cn + r.gauss(0, 0.12 * sigma);
    const mp = r() < 0.03 ? r.gauss(0, 4 * sigma) : 0; // многолучёвость
    const re = ce + r.gauss(0, 0.5 * sigma) + mp, rn = cn + r.gauss(0, 0.5 * sigma) + mp * 0.7;
    const be = ce + r.gauss(0, 0.5 * sigma * 0.3), bn = cn + r.gauss(0, 0.5 * sigma * 0.3); // базовая станция
    raw.push([re, rn]);
    const de = re - be, dn = rn - bn;
    dgps.push([de, dn]);
    se += re; sn += rn; avg.push([se / (k + 1), sn / (k + 1)]);
    // статический фильтр Калмана с отбраковкой по невязке (поверх DGPS)
    const R = (0.5 * sigma) ** 2 * 1.1;
    P += 1e-4;
    const innov = Math.hypot(de - xe, dn - xn);
    if (k < 5 || innov < 3 * Math.sqrt(P + R) + 0.5) {
      const K = P / (P + R);
      xe += K * (de - xe); xn += K * (dn - xn); P = (1 - K) * P;
    }
    kf.push([xe, xn]);
  }
  const e = (a) => a.map(([x, y]) => Math.hypot(x, y));
  const stat = (a) => { const er = e(a); return { cep50: quantile(er, 0.5), cep95: quantile(er, 0.95) }; };
  return { raw, dgps, avg, kf, s: { raw: stat(raw), avg: stat(avg.slice(n / 2)), dgps: stat(dgps), kf: stat(kf.slice(n / 4)) } };
}

export default function Stage6() {
  const [lat, setLat] = useState(44.8152);
  const [lon, setLon] = useState(65.4622);
  const [h, setH] = useState(188.5);
  const [sigma, setSigma] = useState(2.5);
  const [show, setShow] = useState('dgps');

  const conv = useMemo(() => {
    const ecef = geodeticToEcef(lat, lon, h);
    const enu = ecefToEnu(...ecef, AREA.lat0, AREA.lon0, AREA.h0);
    const back = enuToGeodetic(...enu, AREA.lat0, AREA.lon0, AREA.h0);
    const back2 = ecefToGeodetic(...ecef);
    const utm = toUTM(lat, lon), utm0 = toUTM(AREA.lat0, AREA.lon0);
    const rtErr = haversine(lat, lon, back[0], back[1]) * 1000;
    return { ecef, enu, back, back2, utm, utm0, rtErr, hav: haversine(AREA.lat0, AREA.lon0, lat, lon), az: (Math.atan2(enu[0], enu[1]) * 180 / Math.PI + 360) % 360 };
  }, [lat, lon, h]);

  const exp = useMemo(() => gnssExperiment(600, sigma, 21), [sigma]);
  const wps = [[3, 37, 'H (старт)'], [30, 5, 'Цель 1'], [55, 35, 'Цель 2'], [12, 20, 'Цель 3'], [45, 28, 'Цель 4']].map(([gx, gy, name]) => {
    const e = gx * 10 - 300, n = (40 - gy) * 10 - 200, u = 30;
    const [la, lo, hh] = enuToGeodetic(e, n, u, AREA.lat0, AREA.lon0, AREA.h0);
    const ut = toUTM(la, lo);
    return { name, e, n, u, la, lo, hh, ut };
  });

  const showPts = exp[show].map((p) => [p[0], p[1]]);
  const lim = Math.max(6, sigma * 3.5);
  return (
    <>
      <StageHeader
        num={6}
        title="Изменение координат"
        subtitle="Преобразование GPS-координат (WGS-84) в локальную систему БПЛА: геодезические → геоцентрические ECEF → локальные ENU (восток-север-верх) с началом в точке старта, а также в проекцию UTM. Повышение точности позиционирования методами DGPS и фильтрации."
        tags={['WGS-84', 'ECEF', 'ENU', 'UTM (зона 41N)', 'DGPS', 'Фильтр Калмана']}
      />
      <div className="grid g2">
        <div className="card"><h3>Выполняемые задачи</h3><Tasks items={['Геодезические расчёты (эллипсоид WGS-84, формула гаверсинусов)', 'Преобразование координат WGS-84 ↔ ECEF ↔ ENU, UTM', 'Повышение точности (дифференциальная коррекция, фильтр Калмана)']} /></div>
        <div className="card small">
          <h3>Формулы</h3>
          <p style={{ fontFamily: 'var(--mono)', fontSize: 12.5, margin: 0 }}>
            N = a / √(1 − e²·sin²φ)<br />
            X = (N + h)·cosφ·cosλ, Y = (N + h)·cosφ·sinλ, Z = (N(1 − e²) + h)·sinφ<br />
            [e n u]ᵀ = R(φ₀, λ₀) · [X − X₀, Y − Y₀, Z − Z₀]ᵀ<br />
            a = {WGS84.a} м, f = 1/298.257223563, e² = {WGS84.e2.toFixed(9)}
          </p>
        </div>
      </div>

      <SectionTitle n="6.1">Калькулятор преобразования координат</SectionTitle>
      <div className="grid g-side">
        <div className="card">
          <div className="grid g3">
            <label className="ctl">Широта φ, °<input type="number" step="0.0001" value={lat} onChange={(e) => setLat(parseFloat(e.target.value) || 0)} /></label>
            <label className="ctl">Долгота λ, °<input type="number" step="0.0001" value={lon} onChange={(e) => setLon(parseFloat(e.target.value) || 0)} /></label>
            <label className="ctl">Высота h, м<input type="number" step="0.1" value={h} onChange={(e) => setH(parseFloat(e.target.value) || 0)} /></label>
          </div>
          <div className="table-wrap" style={{ marginTop: 14 }}>
            <table>
              <thead><tr><th>Система координат</th><th className="num">Компонента 1</th><th className="num">Компонента 2</th><th className="num">Компонента 3</th></tr></thead>
              <tbody>
                <tr><td>WGS-84 (геодезические)</td><td className="num">φ = {lat.toFixed(7)}°</td><td className="num">λ = {lon.toFixed(7)}°</td><td className="num">h = {h.toFixed(2)} м</td></tr>
                <tr><td>ECEF (геоцентрические)</td><td className="num">X = {conv.ecef[0].toFixed(3)}</td><td className="num">Y = {conv.ecef[1].toFixed(3)}</td><td className="num">Z = {conv.ecef[2].toFixed(3)}</td></tr>
                <tr><td><b>ENU (локальная, от точки старта)</b></td><td className="num"><b>E = {conv.enu[0].toFixed(3)} м</b></td><td className="num"><b>N = {conv.enu[1].toFixed(3)} м</b></td><td className="num"><b>U = {conv.enu[2].toFixed(3)} м</b></td></tr>
                <tr><td>UTM {conv.utm.zone}{conv.utm.hemi}</td><td className="num">E = {conv.utm.E.toFixed(2)}</td><td className="num">N = {conv.utm.N.toFixed(2)}</td><td className="num">ΔE,ΔN от старта: {(conv.utm.E - conv.utm0.E).toFixed(1)}, {(conv.utm.N - conv.utm0.N).toFixed(1)}</td></tr>
                <tr><td>Обратно ECEF → WGS-84</td><td className="num">{conv.back2[0].toFixed(7)}°</td><td className="num">{conv.back2[1].toFixed(7)}°</td><td className="num">{conv.back2[2].toFixed(3)} м</td></tr>
              </tbody>
            </table>
          </div>
        </div>
        <div className="metrics" style={{ alignContent: 'start' }}>
          <Metric k="Расстояние от старта (гаверсинус), м" v={fmt(conv.hav, 2)} color="#22d3ee" />
          <Metric k="Горизонт. расстояние в ENU, м" v={fmt(Math.hypot(conv.enu[0], conv.enu[1]), 2)} />
          <Metric k="Азимут на точку, °" v={fmt(conv.az, 2)} />
          <Metric k="Ошибка прямого+обратного преобразования, мм" v={conv.rtErr.toExponential(1)} color="#4ade80" />
        </div>
      </div>

      <SectionTitle n="6.2">Путевые точки маршрута (этап 2) в разных системах координат</SectionTitle>
      <div className="card table-wrap">
        <table>
          <thead><tr><th>Точка</th><th className="num">E, м</th><th className="num">N, м</th><th className="num">U, м</th><th className="num">Широта</th><th className="num">Долгота</th><th className="num">h, м</th><th className="num">UTM E</th><th className="num">UTM N</th></tr></thead>
          <tbody>
            {wps.map((w) => (
              <tr key={w.name}><td>{w.name}</td><td className="num">{w.e}</td><td className="num">{w.n}</td><td className="num">{w.u}</td><td className="num">{w.la.toFixed(6)}</td><td className="num">{w.lo.toFixed(6)}</td><td className="num">{w.hh.toFixed(2)}</td><td className="num">{w.ut.E.toFixed(1)}</td><td className="num">{w.ut.N.toFixed(1)}</td></tr>
            ))}
          </tbody>
        </table>
        <p className="small muted" style={{ marginTop: 8 }}>Начало ENU — точка {AREA.lat0}°, {AREA.lon0}°, {AREA.h0} м — центр участка 600 × 400 м. Высота полёта 30 м над точкой старта; разница h и U отражает кривизну эллипсоида (≈ d²/2R).</p>
      </div>

      <SectionTitle n="6.3">Повышение точности позиционирования</SectionTitle>
      <div className="grid g-side">
        <div className="card tight">
          <div className="row" style={{ marginBottom: 8 }}>
            <Seg value={show} onChange={setShow} options={[['raw', 'Одиночный GPS'], ['avg', 'Осреднение'], ['dgps', 'DGPS'], ['kf', 'DGPS + Калман']]} />
          </div>
          <LineChart
            height={320}
            xMin={-lim} xMax={lim} yMin={-lim} yMax={lim}
            xLabel="E, м" yLabel="N, м"
            series={[{ name: '600 измерений неподвижной точки (истина в 0,0)', data: showPts, color: show === 'raw' ? '#f87171' : show === 'avg' ? '#fbbf24' : show === 'dgps' ? '#38bdf8' : '#4ade80', scatter: true, r: 1.8, opacity: 0.6 }]}
          />
        </div>
        <div className="card tight controls">
          <Slider label="СКО ошибки GNSS, м" value={sigma} min={0.5} max={5} step={0.5} onChange={setSigma} digits={1} />
          <div className="small muted">CEP50 / CEP95 горизонтальной ошибки, м</div>
          <BarChart width={330} height={150} items={[
            { label: 'Одиночный GPS', value: exp.s.raw.cep95, color: '#f87171' },
            { label: 'Осреднение', value: exp.s.avg.cep95, color: '#fbbf24' },
            { label: 'DGPS', value: exp.s.dgps.cep95, color: '#38bdf8' },
            { label: 'DGPS + Калман', value: exp.s.kf.cep95, color: '#4ade80' },
          ]} />
          <table>
            <thead><tr><th>Метод</th><th className="num">CEP50</th><th className="num">CEP95</th></tr></thead>
            <tbody>
              {[['Одиночный', 'raw'], ['Осреднение', 'avg'], ['DGPS', 'dgps'], ['DGPS+KF', 'kf']].map(([n, k]) => (
                <tr key={k}><td>{n}</td><td className="num">{fmt(exp.s[k].cep50)}</td><td className="num">{fmt(exp.s[k].cep95)}</td></tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
      <p className="muted small">Модель ошибки: общая для приёмников атмосферная составляющая (коррелированный процесс), шум приёмника и редкие выбросы многолучёвости. Базовая станция с известными координатами вычисляет поправку и передаёт её на борт (DGPS) — общая составляющая устраняется. Осреднение медленно сходится из-за коррелированной ошибки, а фильтр Калмана поверх DGPS дополнительно подавляет шум и отбраковывает выбросы.</p>

      <SectionTitle n="6.4">Программная реализация</SectionTitle>
      <Code title="coords.py" code={`
import numpy as np
A, F = 6378137.0, 1 / 298.257223563
E2 = F * (2 - F)

def geodetic_to_ecef(lat, lon, h):
    p, l = np.radians(lat), np.radians(lon)
    N = A / np.sqrt(1 - E2 * np.sin(p)**2)
    return np.array([(N + h)*np.cos(p)*np.cos(l), (N + h)*np.cos(p)*np.sin(l), (N*(1 - E2) + h)*np.sin(p)])

def ecef_to_enu(xyz, lat0, lon0, h0):
    p, l = np.radians(lat0), np.radians(lon0)
    R = np.array([[-np.sin(l),              np.cos(l),             0        ],
                  [-np.sin(p)*np.cos(l), -np.sin(p)*np.sin(l),  np.cos(p)],
                  [ np.cos(p)*np.cos(l),  np.cos(p)*np.sin(l),  np.sin(p)]])
    return R @ (xyz - geodetic_to_ecef(lat0, lon0, h0))

def enu_to_geodetic(e, n, u, lat0, lon0, h0):
    p, l = np.radians(lat0), np.radians(lon0)
    R = np.array([[-np.sin(l), -np.sin(p)*np.cos(l), np.cos(p)*np.cos(l)],
                  [ np.cos(l), -np.sin(p)*np.sin(l), np.cos(p)*np.sin(l)],
                  [ 0,          np.cos(p),            np.sin(p)          ]])
    X, Y, Z = geodetic_to_ecef(lat0, lon0, h0) + R @ np.array([e, n, u])
    return ecef_to_geodetic(X, Y, Z)          # формула Боуринга

LAT0, LON0, H0 = ${AREA.lat0}, ${AREA.lon0}, ${AREA.h0}
print(ecef_to_enu(geodetic_to_ecef(44.8152, 65.4622, 188.5), LAT0, LON0, H0))

# DGPS: поправка базовой станции с известными координатами
corr = base_measured - base_true
rover_corrected = rover_measured - corr
`} />
      <div className="note section">
        <b>Вывод.</b> Локальная система ENU позволяет работать с метрическими координатами (м) вместо градусов: на широте 44,8° один градус долготы ≈ 79 км, широты ≈ 111 км. Прямое и обратное преобразования согласованы с субмиллиметровой точностью. Дифференциальная коррекция и фильтр Калмана уменьшают круговую ошибку CEP95 в несколько раз по сравнению с одиночным GPS.
      </div>
      <Pager href="/etap-6" />
    </>
  );
}
