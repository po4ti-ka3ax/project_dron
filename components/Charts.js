'use client';
// Лёгкие SVG-графики без внешних зависимостей

function niceTicks(lo, hi, n = 5) {
  if (!Number.isFinite(lo) || !Number.isFinite(hi)) return [0, 1];
  if (hi === lo) { hi = lo + 1; lo = lo - 1; }
  const span = hi - lo;
  const step0 = span / n;
  const mag = 10 ** Math.floor(Math.log10(step0));
  const err = step0 / mag;
  const step = (err >= 7.5 ? 10 : err >= 3.5 ? 5 : err >= 1.5 ? 2 : 1) * mag;
  const start = Math.ceil(lo / step) * step;
  const ticks = [];
  for (let v = start; v <= hi + step * 1e-6; v += step) ticks.push(+v.toFixed(10));
  return ticks;
}
const tickLabel = (v) => {
  const a = Math.abs(v);
  if (a >= 10000) return (v / 1000).toFixed(0) + 'k';
  if (a >= 100 || Number.isInteger(v)) return String(Math.round(v * 100) / 100);
  if (a >= 1) return v.toFixed(1);
  return v.toFixed(2);
};

export function LineChart({ series, height = 240, width = 640, xLabel, yLabel, yMin, yMax, xMin, xMax, markers = [], hLines = [], scatter = false, areas = [] }) {
  const pad = { l: 48, r: 14, t: 12, b: 34 };
  const pts = series.flatMap((s) => s.data);
  const xs = pts.map((p) => p[0]).filter(Number.isFinite);
  const ys = pts.map((p) => p[1]).filter(Number.isFinite);
  let x0 = xMin ?? Math.min(...xs), x1 = xMax ?? Math.max(...xs);
  let y0 = yMin ?? Math.min(...ys, ...hLines.map((h) => h.y)), y1 = yMax ?? Math.max(...ys, ...hLines.map((h) => h.y));
  if (!Number.isFinite(x0)) { x0 = 0; x1 = 1; }
  if (!Number.isFinite(y0)) { y0 = 0; y1 = 1; }
  if (y1 === y0) { y1 += 1; y0 -= 1; }
  if (x1 === x0) x1 += 1;
  if (yMin === undefined && yMax === undefined) { const m = (y1 - y0) * 0.06; y0 -= m; y1 += m; }
  const W = width - pad.l - pad.r, H = height - pad.t - pad.b;
  const sx = (x) => pad.l + ((x - x0) / (x1 - x0)) * W;
  const sy = (y) => pad.t + H - ((y - y0) / (y1 - y0)) * H;
  const xt = niceTicks(x0, x1, 6), yt = niceTicks(y0, y1, 5);
  const path = (d) => {
    let s = '', pen = false;
    for (const [x, y] of d) {
      if (!Number.isFinite(y) || !Number.isFinite(x)) { pen = false; continue; }
      s += (pen ? 'L' : 'M') + sx(x).toFixed(1) + ',' + sy(Math.max(y0, Math.min(y1, y))).toFixed(1);
      pen = true;
    }
    return s;
  };
  return (
    <div>
      <svg viewBox={`0 0 ${width} ${height}`} className="chart">
        {areas.map((a, i) => (
          <rect key={'a' + i} x={sx(Math.max(a.x0, x0))} y={pad.t} width={Math.max(0, sx(Math.min(a.x1, x1)) - sx(Math.max(a.x0, x0)))} height={H} fill={a.color || '#f8717122'} />
        ))}
        {yt.map((v) => (
          <g key={'y' + v}>
            <line x1={pad.l} x2={pad.l + W} y1={sy(v)} y2={sy(v)} stroke="#24324f" strokeDasharray="3 4" />
            <text x={pad.l - 6} y={sy(v) + 4} fill="#93a2bf" fontSize="11" textAnchor="end">{tickLabel(v)}</text>
          </g>
        ))}
        {xt.map((v) => (
          <text key={'x' + v} x={sx(v)} y={pad.t + H + 16} fill="#93a2bf" fontSize="11" textAnchor="middle">{tickLabel(v)}</text>
        ))}
        <line x1={pad.l} x2={pad.l + W} y1={pad.t + H} y2={pad.t + H} stroke="#3a4a6b" />
        <line x1={pad.l} x2={pad.l} y1={pad.t} y2={pad.t + H} stroke="#3a4a6b" />
        {hLines.map((h, i) => (
          <g key={'h' + i}>
            <line x1={pad.l} x2={pad.l + W} y1={sy(h.y)} y2={sy(h.y)} stroke={h.color || '#fbbf24'} strokeDasharray="6 4" />
            {h.label ? <text x={pad.l + W - 4} y={sy(h.y) - 4} fill={h.color || '#fbbf24'} fontSize="11" textAnchor="end">{h.label}</text> : null}
          </g>
        ))}
        {series.map((s, i) =>
          s.scatter || scatter ? (
            <g key={i}>
              {s.data.filter((p) => Number.isFinite(p[1])).map((p, j) => (
                <circle key={j} cx={sx(p[0])} cy={sy(p[1])} r={s.r || 2.5} fill={p[2] || s.color} opacity={s.opacity ?? 0.85} />
              ))}
            </g>
          ) : (
            <path key={i} d={path(s.data)} fill="none" stroke={s.color} strokeWidth={s.width || 2} strokeDasharray={s.dash} opacity={s.opacity ?? 1} />
          )
        )}
        {markers.map((m, i) => (
          <g key={'m' + i}>
            <line x1={sx(m.x)} x2={sx(m.x)} y1={pad.t} y2={pad.t + H} stroke={m.color || '#a78bfa'} strokeDasharray="4 4" />
            {m.label ? <text x={sx(m.x) + 4} y={pad.t + 12} fill={m.color || '#a78bfa'} fontSize="11">{m.label}</text> : null}
          </g>
        ))}
        {xLabel ? <text x={pad.l + W / 2} y={height - 4} fill="#93a2bf" fontSize="11" textAnchor="middle">{xLabel}</text> : null}
        {yLabel ? <text x={12} y={pad.t + H / 2} fill="#93a2bf" fontSize="11" textAnchor="middle" transform={`rotate(-90 12 ${pad.t + H / 2})`}>{yLabel}</text> : null}
      </svg>
      <div className="legend">
        {series.filter((s) => s.name).map((s) => (
          <span key={s.name}><i style={{ background: s.color }} />{s.name}</span>
        ))}
      </div>
    </div>
  );
}

export function BarChart({ items, height = 220, width = 520, max, unit = '', digits = 2 }) {
  const pad = { l: 150, r: 60, t: 6, b: 6 };
  const m = max ?? Math.max(...items.map((i) => i.value), 1e-9);
  const bh = (height - pad.t - pad.b) / items.length;
  return (
    <svg viewBox={`0 0 ${width} ${height}`} className="chart">
      {items.map((it, i) => {
        const w = ((width - pad.l - pad.r) * Math.max(0, it.value)) / m;
        const y = pad.t + i * bh;
        return (
          <g key={it.label}>
            <text x={pad.l - 8} y={y + bh / 2 + 4} fill="#c7d3ea" fontSize="12" textAnchor="end">{it.label}</text>
            <rect x={pad.l} y={y + bh * 0.18} width={w} height={bh * 0.64} rx="4" fill={it.color || '#38bdf8'} />
            <text x={pad.l + w + 6} y={y + bh / 2 + 4} fill="#e6ecf7" fontSize="12" fontFamily="monospace">{it.value.toFixed(digits)}{unit}</text>
          </g>
        );
      })}
    </svg>
  );
}
