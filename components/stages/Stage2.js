'use client';
import { useEffect, useMemo, useRef, useState } from 'react';
import { StageHeader, Tasks, SectionTitle, Pager, Slider, Seg, Metric, Code } from '@/components/UI';
import { astar, smoothPath, pathLen, twoOpt } from '@/lib/astar';
import { fmt } from '@/lib/math';
import { AREA } from '@/lib/pages';

const GW = 60, GH = 40, CELL = 10, PX = 10; // 600×400 м, 1 клетка = 10 м
const BASE = [3, 37];
const OBST = [
  { type: 'circle', x: 48, y: 8, r: 6, h: Infinity, name: 'Посёлок (бесполётная зона)' },
  { type: 'circle', x: 36, y: 16, r: 2, h: 60, name: 'Вышка связи, 60 м' },
  { type: 'rect', x0: 20, y0: 12, x1: 23, y1: 26, h: 18, name: 'Лесополоса, 18 м' },
  { type: 'rect', x0: 8, y0: 5, x1: 14, y1: 10, h: 15, name: 'Здания фермы, 15 м' },
  { type: 'line', x0: 6, y0: 31, x1: 40, y1: 24, w: 0.8, h: 30, name: 'ЛЭП 35 кВ, 30 м' },
];
function inside(o, x, y, m = 0) {
  if (o.type === 'circle') return Math.hypot(x - o.x, y - o.y) <= o.r + m;
  if (o.type === 'rect') return x >= o.x0 - m && x <= o.x1 + m && y >= o.y0 - m && y <= o.y1 + m;
  const dx = o.x1 - o.x0, dy = o.y1 - o.y0;
  const t = Math.max(0, Math.min(1, ((x - o.x0) * dx + (y - o.y0) * dy) / (dx * dx + dy * dy)));
  return Math.hypot(x - (o.x0 + t * dx), y - (o.y0 + t * dy)) <= o.w + m;
}
// препятствием считается объект выше (высота полёта − 10 м) или бесполётная зона
const active = (o, alt) => o.h >= alt - 10;
function buildGrid(margin, alt) {
  return Array.from({ length: GH }, (_, y) => Array.from({ length: GW }, (_, x) => (OBST.some((o) => active(o, alt) && inside(o, x, y, margin)) ? 1 : 0)));
}
const free = (grid, p) => {
  const x = Math.round(p[0]), y = Math.round(p[1]);
  return x >= 0 && y >= 0 && x < GW && y < GH && !grid[y][x];
};

// Параметрические траектории + коррекция участков, пересекающих препятствия, через A*
function parametric(kind, cx, cy, R) {
  const pts = [];
  for (let i = 0; i <= 240; i++) {
    const t = (i / 240) * 2 * Math.PI;
    if (kind === 'circle') pts.push([cx + R * Math.cos(t), cy + R * Math.sin(t)]);
    else pts.push([cx + R * 1.35 * Math.sin(t), cy + R * Math.sin(t) * Math.cos(t)]);
  }
  return pts;
}
function repair(grid, pts) {
  const out = [];
  let conflicts = 0, i = 0;
  while (i < pts.length) {
    if (free(grid, pts[i])) { out.push(pts[i]); i++; continue; }
    conflicts++;
    let j = i;
    while (j < pts.length && !free(grid, pts[j])) j++;
    const a = out.length ? out[out.length - 1] : null;
    const b = pts[Math.min(j, pts.length - 1)];
    if (a && free(grid, b)) {
      const r = astar(grid, a.map(Math.round), b.map(Math.round));
      if (r.path) out.push(...smoothPath(grid, r.path));
    }
    i = j;
  }
  return { out, conflicts };
}
function legsPath(grid, points) {
  let path = [], expanded = 0, expandedDij = 0, raw = 0;
  for (let k = 1; k < points.length; k++) {
    const r = astar(grid, points[k - 1], points[k]);
    const d = astar(grid, points[k - 1], points[k], { heuristic: false });
    expanded += r.expanded; expandedDij += d.expanded;
    if (!r.path) return { path: null };
    raw += pathLen(r.path);
    const sm = smoothPath(grid, r.path);
    path = path.concat(k > 1 ? sm.slice(1) : sm);
  }
  return { path, expanded, expandedDij, raw };
}

export default function Stage2() {
  const [mode, setMode] = useState('targets');
  const [R, setR] = useState(14);
  const [margin, setMargin] = useState(1);
  const [alt, setAlt] = useState(60);
  const [overlap, setOverlap] = useState(0.3);
  const [targets, setTargets] = useState([[30, 5], [55, 35], [12, 20], [45, 28], [26, 33], [56, 18]]);
  const [optimize, setOptimize] = useState(true);
  const canvas = useRef(null);
  const flightAlt = mode === 'cover' ? alt : 25;
  const grid = useMemo(() => buildGrid(margin, flightAlt), [margin, flightAlt]);

  const result = useMemo(() => {
    const t0 = performance.now();
    let res = { path: null };
    if (mode === 'circle' || mode === 'eight') {
      const pts = parametric(mode, 30, 20, R);
      const { out, conflicts } = repair(grid, pts);
      res = { path: out, ideal: pts, conflicts, raw: pathLen(pts) };
    } else if (mode === 'targets') {
      const pts = [BASE, ...targets.filter((t) => free(grid, t))];
      // матрица расстояний по A*
      const n = pts.length;
      const D = Array.from({ length: n }, () => new Array(n).fill(0));
      for (let i = 0; i < n; i++) for (let j = i + 1; j < n; j++) { const c = astar(grid, pts[i], pts[j]).cost; D[i][j] = D[j][i] = c; }
      const naive = [...Array(n).keys()];
      const nLen = naive.reduce((s, v, i) => (i ? s + D[naive[i - 1]][v] : 0), 0) + D[naive[n - 1]][0];
      // жадный старт (ближайший сосед) + 2-opt
      const nn = [0];
      const used = new Set([0]);
      while (nn.length < n) {
        const last = nn[nn.length - 1];
        let best = -1;
        for (let j = 0; j < n; j++) if (!used.has(j) && (best < 0 || D[last][j] < D[last][best])) best = j;
        nn.push(best); used.add(best);
      }
      const order = optimize ? twoOpt(nn, D) : naive;
      const seq = [...order.map((i) => pts[i]), BASE];
      res = { ...legsPath(grid, seq), order, naiveLen: nLen, seq };
    } else {
      const fov = (73.7 * Math.PI) / 180;
      const foot = 2 * alt * Math.tan(fov / 2); // ширина полосы захвата, м
      const step = Math.max(1, (foot * (1 - overlap)) / CELL);
      const wps = [BASE];
      let k = 0;
      for (let x = 2; x < GW - 1; x += step, k++) {
        const xi = Math.round(x);
        const ys = k % 2 === 0 ? [37, 2] : [2, 37];
        for (const y of ys) {
          let p = [xi, y];
          // если точка в препятствии — сдвигаем вдоль галса
          let tries = 0;
          while (!free(grid, p) && tries < 20) { p = [xi, p[1] + (y > 20 ? -1 : 1)]; tries++; }
          if (free(grid, p)) wps.push(p);
        }
      }
      wps.push(BASE);
      res = { ...legsPath(grid, wps), foot, step, lanes: k };
    }
    res.ms = performance.now() - t0;
    res.len = pathLen(res.path) * CELL;
    return res;
  }, [mode, R, grid, targets, optimize, alt, overlap]);

  useEffect(() => {
    const c = canvas.current;
    if (!c) return;
    c.width = GW * PX; c.height = GH * PX;
    const ctx = c.getContext('2d');
    // поля (рисовые чеки)
    for (let py = 0; py < 2; py++)
      for (let px = 0; px < 4; px++) {
        ctx.fillStyle = ['#2f6b3a', '#3a7a3f', '#4c7f35', '#356f45'][(px + py) % 4];
        ctx.fillRect(px * 150 + 2, py * 200 + 2, 146, 196);
      }
    ctx.fillStyle = '#2b6ea8';
    ctx.fillRect(0, 196, 600, 8); // оросительный канал
    ctx.fillStyle = '#9a8a6a';
    ctx.fillRect(440, 0, 6, 400); // дорога
    // зона безопасности и препятствия
    for (let y = 0; y < GH; y++)
      for (let x = 0; x < GW; x++)
        if (grid[y][x]) { ctx.fillStyle = 'rgba(248,113,113,0.28)'; ctx.fillRect(x * PX, y * PX, PX, PX); }
    for (const o of OBST) {
      const on = active(o, flightAlt);
      ctx.fillStyle = on ? '#7f1d1d' : 'rgba(127,29,29,.35)'; ctx.strokeStyle = on ? '#f87171' : 'rgba(248,113,113,.4)'; ctx.lineWidth = 2;
      ctx.beginPath();
      if (o.type === 'circle') ctx.arc(o.x * PX + 5, o.y * PX + 5, o.r * PX, 0, 7);
      else if (o.type === 'rect') ctx.rect(o.x0 * PX, o.y0 * PX, (o.x1 - o.x0 + 1) * PX, (o.y1 - o.y0 + 1) * PX);
      if (o.type === 'line') { ctx.moveTo(o.x0 * PX + 5, o.y0 * PX + 5); ctx.lineTo(o.x1 * PX + 5, o.y1 * PX + 5); ctx.lineWidth = 4; ctx.stroke(); }
      else { ctx.fill(); ctx.stroke(); }
    }
    ctx.font = '11px sans-serif'; ctx.fillStyle = '#fecaca';
    for (const o of OBST) {
      const x = o.type === 'circle' ? (o.r > 3 ? o.x - o.r : o.x - 4) : o.type === 'rect' ? o.x0 : (o.x0 + o.x1) / 2;
      const y = o.type === 'circle' ? (o.r > 3 ? o.y + o.r + 1.6 : o.y - o.r - 0.6) : o.type === 'rect' ? o.y0 - 0.6 : (o.y0 + o.y1) / 2 + 2.2;
      ctx.fillText(o.name, x * PX - 10, y * PX);
    }
    // идеальная траектория
    if (result.ideal) {
      ctx.setLineDash([5, 5]); ctx.strokeStyle = '#e2e8f0'; ctx.lineWidth = 1.2; ctx.beginPath();
      result.ideal.forEach((p, i) => (i ? ctx.lineTo(p[0] * PX + 5, p[1] * PX + 5) : ctx.moveTo(p[0] * PX + 5, p[1] * PX + 5)));
      ctx.stroke(); ctx.setLineDash([]);
    }
    if (result.path) {
      ctx.strokeStyle = '#22d3ee'; ctx.lineWidth = 3; ctx.lineJoin = 'round'; ctx.beginPath();
      result.path.forEach((p, i) => (i ? ctx.lineTo(p[0] * PX + 5, p[1] * PX + 5) : ctx.moveTo(p[0] * PX + 5, p[1] * PX + 5)));
      ctx.stroke();
      // стрелки направления
      for (let i = 4; i < result.path.length; i += 6) {
        const a = result.path[i - 1], b = result.path[i];
        const ang = Math.atan2(b[1] - a[1], b[0] - a[0]);
        ctx.save(); ctx.translate(b[0] * PX + 5, b[1] * PX + 5); ctx.rotate(ang);
        ctx.fillStyle = '#22d3ee'; ctx.beginPath(); ctx.moveTo(6, 0); ctx.lineTo(-4, -4); ctx.lineTo(-4, 4); ctx.fill(); ctx.restore();
      }
    }
    if (mode === 'targets') {
      const order = result.order || [];
      const pts = [BASE, ...targets.filter((t) => free(grid, t))];
      pts.slice(1).forEach((t, i) => {
        ctx.fillStyle = '#fbbf24'; ctx.beginPath(); ctx.arc(t[0] * PX + 5, t[1] * PX + 5, 8, 0, 7); ctx.fill();
        ctx.fillStyle = '#1a1a1a'; ctx.font = 'bold 11px sans-serif'; ctx.textAlign = 'center';
        ctx.fillText(String(order.indexOf(i + 1)), t[0] * PX + 5, t[1] * PX + 9); ctx.textAlign = 'left';
      });
    }
    ctx.fillStyle = '#4ade80'; ctx.beginPath(); ctx.arc(BASE[0] * PX + 5, BASE[1] * PX + 5, 9, 0, 7); ctx.fill();
    ctx.fillStyle = '#052e16'; ctx.font = 'bold 10px sans-serif'; ctx.fillText('H', BASE[0] * PX + 1.5, BASE[1] * PX + 9);
    // масштабная линейка
    ctx.fillStyle = '#e6ecf7'; ctx.fillRect(470, 385, 100, 3); ctx.font = '11px sans-serif'; ctx.fillText('100 м', 505, 380);
    ctx.fillText('N ↑', 575, 18);
  }, [result, grid, mode, targets, flightAlt]);

  const onClick = (e) => {
    if (mode !== 'targets') return;
    const rect = canvas.current.getBoundingClientRect();
    const x = Math.floor(((e.clientX - rect.left) / rect.width) * GW), y = Math.floor(((e.clientY - rect.top) / rect.height) * GH);
    if (!free(grid, [x, y])) return;
    setTargets((t) => [...t, [x, y]].slice(-9));
  };

  const speed = 10;
  return (
    <>
      <StageHeader
        num={2}
        title="Генерация траектории"
        subtitle={`Расчёт наиболее эффективного маршрута БПЛА: круговая, восьмёрочная и произвольная траектории, облёт препятствий и оптимизация порядка обхода целей. Район: ${AREA.name}.`}
        tags={['A* (октильная эвристика)', 'Оптимизация порядка: ближайший сосед + 2-opt', 'Натягивание нити', 'Покрытие поля «змейкой»']}
      />
      <div className="grid g2">
        <div className="card"><h3>Выполняемые задачи</h3><Tasks items={['Применение алгоритмов ИИ (эвристический поиск A*, комбинаторная оптимизация 2-opt)', 'Учёт препятствий с зоной безопасности', 'Определение оптимального пути и порядка облёта целей']} /></div>
        <div className="card">
          <h3>Модель района</h3>
          <p className="small muted" style={{ margin: 0 }}>Сетка 60 × 40 клеток по 10 м (600 × 400 м), высота полёта 30–60 м. Препятствия выше высоты полёта либо запретные зоны: посёлок, вышка связи, лесополоса, постройки фермы, ЛЭП. Вокруг каждого препятствия добавляется зона безопасности (инфляция). Старт и посадка — точка H.</p>
        </div>
      </div>

      <SectionTitle n="2.1">Интерактивный планировщик</SectionTitle>
      <div className="grid g-side">
        <div className="card tight">
          <div className="row" style={{ marginBottom: 10 }}>
            <Seg value={mode} onChange={setMode} options={[['circle', 'Круговая'], ['eight', 'Восьмёрка'], ['targets', 'Произвольная (цели)'], ['cover', 'Покрытие поля']]} />
          </div>
          <canvas ref={canvas} onClick={onClick} className={mode === 'targets' ? 'canvas-click' : ''} style={{ width: '100%' }} />
          <div className="caption">
            {mode === 'targets' ? 'Кликните по карте, чтобы добавить цель. Цифры — оптимальный порядок облёта.' : mode === 'cover' ? `Галсы строятся по ширине полосы захвата камеры. На высоте ${alt} м препятствиями остаются только объекты выше ${alt - 10} м и бесполётная зона (бледные объекты пролетаются сверху).` : 'Высота полёта 25 м. Пунктир — исходная параметрическая траектория, голубая линия — скорректированная с облётом препятствий.'}
          </div>
        </div>
        <div className="card tight controls">
          <Slider label="Зона безопасности, клеток (×10 м)" value={margin} min={0} max={3} step={1} onChange={setMargin} />
          {(mode === 'circle' || mode === 'eight') && <Slider label="Радиус / амплитуда, клеток" value={R} min={6} max={19} step={1} onChange={setR} />}
          {mode === 'targets' && (
            <>
              <label className="row small"><input type="checkbox" checked={optimize} onChange={(e) => setOptimize(e.target.checked)} /> Оптимизировать порядок (NN + 2-opt)</label>
              <div className="row">
                <button className="btn small" onClick={() => setTargets([])} type="button">Очистить цели</button>
                <button className="btn small" onClick={() => setTargets([[30, 5], [55, 35], [12, 20], [45, 28], [26, 33], [56, 18]])} type="button">По умолчанию</button>
              </div>
            </>
          )}
          {mode === 'cover' && (
            <>
              <Slider label="Высота съёмки, м" value={alt} min={30} max={120} step={5} onChange={setAlt} />
              <Slider label="Перекрытие снимков" value={overlap} min={0} max={0.7} step={0.05} onChange={setOverlap} />
            </>
          )}
          <div className="metrics">
            <Metric k="Длина маршрута, м" v={fmt(result.len, 0)} color="#22d3ee" />
            <Metric k={`Время при ${speed} м/с`} v={result.len ? `${Math.floor(result.len / speed / 60)} мин ${Math.round((result.len / speed) % 60)} с` : '—'} />
            <Metric k="Время расчёта, мс" v={fmt(result.ms, 1)} />
            {result.conflicts !== undefined && <Metric k="Конфликтов с препятствиями" v={result.conflicts} color={result.conflicts ? '#fbbf24' : '#4ade80'} />}
            {result.expanded !== undefined && <Metric k="Узлов: A* / Дейкстра" v={`${result.expanded} / ${result.expandedDij}`} />}
            {result.naiveLen !== undefined && <Metric k="Выигрыш от оптимизации" v={`${fmt((1 - (result.raw || 0) / (result.naiveLen || 1)) * 100, 1)}%`} color="#4ade80" />}
            {result.foot !== undefined && <Metric k="Полоса захвата / шаг галсов, м" v={`${fmt(result.foot, 0)} / ${fmt(result.step * CELL, 0)}`} />}
          </div>
        </div>
      </div>

      <SectionTitle n="2.2">Используемые алгоритмы</SectionTitle>
      <div className="grid g3">
        <div className="card"><h3>A* (поиск пути)</h3><p className="small">f(n) = g(n) + h(n), где g — пройденная стоимость, h — октильная эвристика √2·min(dx,dy) + |dx−dy|. Эвристика допустима, поэтому путь оптимален на сетке, а число раскрытых узлов в разы меньше, чем у алгоритма Дейкстры.</p></div>
        <div className="card"><h3>Оптимизация порядка целей</h3><p className="small">Матрица расстояний строится по реальным длинам A*-путей (а не по прямой). Начальный маршрут — жадный «ближайший сосед», затем 2-opt переставляет участки, пока длина уменьшается (задача коммивояжёра).</p></div>
        <div className="card"><h3>Сглаживание и облёт</h3><p className="small">Ступенчатый сеточный путь упрощается «натягиванием нити» с проверкой прямой видимости. Для круговой и восьмёрочной траекторий участки, попадающие в препятствие, заменяются локальным A*-объездом.</p></div>
      </div>

      <SectionTitle n="2.3">Программная реализация</SectionTitle>
      <Code title="trajectory.py" code={`
import numpy as np, heapq

def astar(grid, start, goal):
    H, W = grid.shape
    h = lambda p: (lambda dx, dy: dx + dy + (np.sqrt(2) - 2) * min(dx, dy))(abs(p[0]-goal[0]), abs(p[1]-goal[1]))
    g = {start: 0}; came = {}; open_ = [(h(start), start)]
    while open_:
        _, cur = heapq.heappop(open_)
        if cur == goal: break
        for dx, dy in [(1,0),(-1,0),(0,1),(0,-1),(1,1),(1,-1),(-1,1),(-1,-1)]:
            nx, ny = cur[0] + dx, cur[1] + dy
            if not (0 <= nx < W and 0 <= ny < H) or grid[ny, nx]: continue
            ng = g[cur] + np.hypot(dx, dy)
            if ng < g.get((nx, ny), np.inf):
                g[(nx, ny)] = ng; came[(nx, ny)] = cur
                heapq.heappush(open_, (ng + h((nx, ny)), (nx, ny)))
    path = [goal]
    while path[-1] != start: path.append(came[path[-1]])
    return path[::-1]

def circle(cx, cy, R, n=240):
    t = np.linspace(0, 2*np.pi, n); return np.c_[cx + R*np.cos(t), cy + R*np.sin(t)]

def figure_eight(cx, cy, A, n=240):   # лемниската Жероно
    t = np.linspace(0, 2*np.pi, n); return np.c_[cx + 1.35*A*np.sin(t), cy + A*np.sin(t)*np.cos(t)]

def two_opt(order, D):
    best = order; improved = True
    length = lambda o: sum(D[o[i-1], o[i]] for i in range(1, len(o))) + D[o[-1], o[0]]
    while improved:
        improved = False
        for i in range(1, len(best) - 1):
            for k in range(i + 1, len(best)):
                cand = best[:i] + best[i:k+1][::-1] + best[k+1:]
                if length(cand) < length(best): best, improved = cand, True
    return best
`} />
      <div className="note section">
        <b>Вывод.</b> A* находит оптимальный по длине путь, раскрывая в 3–6 раз меньше узлов, чем Дейкстра. Оптимизация порядка облёта целей (NN + 2-opt) сокращает маршрут относительно порядка «как задано», а сглаживание убирает лишние повороты, что снижает энергозатраты БПЛА.
      </div>
      <Pager href="/etap-2" />
    </>
  );
}
