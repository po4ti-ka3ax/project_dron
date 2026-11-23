'use client';
import { useRouter } from 'next/navigation';
import { useState } from 'react';

// Интерактивная версия рисунка 1: каждый блок-этап — кнопка перехода на страницу
const NODES = [
  { id: 'cam', x: 20, y: 18, w: 180, h: 54, kind: 'input', lines: ['Вход: поток', 'данных с камеры'] },
  { id: 's1', x: 225, y: 82, w: 175, h: 78, kind: 'stage', href: '/etap-1', lines: ['Этап 1', 'Подготовка', 'данных и ПО'] },
  { id: 's4', x: 445, y: 18, w: 175, h: 70, kind: 'stage', href: '/etap-4', lines: ['Этап 4', 'Обнаружение объектов'] },
  { id: 'map', x: 800, y: 18, w: 180, h: 58, kind: 'out', lines: ['Создание', 'локальной карты'] },
  { id: 's5', x: 620, y: 112, w: 165, h: 70, kind: 'stage', href: '/etap-5', lines: ['Этап 5', 'Построение карты'] },
  { id: 's6', x: 815, y: 200, w: 165, h: 70, kind: 'stage', href: '/etap-6', lines: ['Этап 6', 'Изменение координат'] },
  { id: 'iis', x: 20, y: 228, w: 770, h: 46, kind: 'core', lines: ['Интеллектуальная информационная система'] },
  { id: 'in2', x: 20, y: 318, w: 160, h: 78, kind: 'input', lines: ['Вход: карта,', 'позиция,', 'цель'] },
  { id: 's2', x: 210, y: 318, w: 175, h: 78, kind: 'stage', href: '/etap-2', lines: ['Этап 2', 'Генерация', 'траектории'] },
  { id: 's3', x: 425, y: 318, w: 175, h: 78, kind: 'stage', href: '/etap-3', lines: ['Этап 3', 'Управление', 'полётом'] },
  { id: 's7', x: 640, y: 318, w: 165, h: 78, kind: 'stage', href: '/etap-7', lines: ['Этап 7', 'Принятие', 'решений'] },
  { id: 'gps', x: 830, y: 330, w: 150, h: 92, kind: 'input', lines: ['Вход: GPS,', 'препятствия', 'Выход: прогноз,', 'обучение'] },
  { id: 'iu', x: 210, y: 440, w: 595, h: 44, kind: 'core2', lines: ['ИНТЕЛЛЕКТУАЛЬНОЕ УПРАВЛЕНИЕ'] },
  { id: 'prog', x: 210, y: 512, w: 285, h: 50, kind: 'extra', href: '/program', lines: ['Программа по блок-схеме (рис. 2)'] },
  { id: 'ndvi', x: 520, y: 512, w: 285, h: 50, kind: 'extra', href: '/ndvi', lines: ['Расчёт NDVI и решение (рис. 3)'] },
];

const STYLE = {
  input: { fill: '#0c2c44', stroke: '#38bdf8', text: '#cfeaff' },
  stage: { fill: '#3b1d0b', stroke: '#fb923c', text: '#ffedd5' },
  out: { fill: '#3b2d0b', stroke: '#fbbf24', text: '#fef3c7' },
  core: { fill: '#3b0b2e', stroke: '#e879f9', text: '#fae8ff' },
  core2: { fill: '#0b2a3b', stroke: '#22d3ee', text: '#cffafe' },
  extra: { fill: '#0f3b2a', stroke: '#4ade80', text: '#dcfce7' },
};
const STAGE_COLORS = { s1: '#ef4444', s2: '#22c55e', s3: '#f59e0b', s4: '#f59e0b', s5: '#f97316', s6: '#38bdf8', s7: '#ef4444' };

const EDGES = [
  ['cam', 's1', 'Desired functionality'],
  ['s1', 'iis', 'Documentation'],
  ['s4', 'iis', 'Existing documentation'],
  ['s5', 'iis', 'Existing Models'],
  ['s6', 'iis', 'Resurs System'],
  ['in2', 's2', 'Библиотеки'],
  ['iis', 's2', 'Version specs'],
  ['iis', 's3', 'Design specs'],
  ['iis', 's7', 'Advantages'],
  ['s2', 's3', ''],
  ['s3', 's7', 'Маневры'],
  ['s7', 'gps', 'Спецификация'],
  ['s2', 'iu', ''],
  ['s3', 'iu', ''],
  ['s7', 'iu', ''],
  ['s4', 'map', 'Progress reports'],
  ['s5', 'map', 'ОТЧЕТЫ'],
  ['s6', 'map', 'ОТЧЕТЫ'],
  ['cam', 'map', 'Progress reports'],
  ['iu', 'prog', ''],
  ['iu', 'ndvi', ''],
];

function center(n) { return [n.x + n.w / 2, n.y + n.h / 2]; }
function anchor(a, b) {
  // точка на границе прямоугольника a по направлению к центру b
  const [cx, cy] = center(a), [tx, ty] = center(b);
  const dx = tx - cx, dy = ty - cy;
  const sx = a.w / 2 / Math.abs(dx || 1e-9), sy = a.h / 2 / Math.abs(dy || 1e-9);
  const s = Math.min(sx, sy);
  return [cx + dx * s, cy + dy * s];
}

export default function SystemScheme() {
  const router = useRouter();
  const [hover, setHover] = useState(null);
  const byId = Object.fromEntries(NODES.map((n) => [n.id, n]));
  return (
    <svg viewBox="0 0 1000 580" className="chart flow-svg" role="img" aria-label="Схема интеллектуальной системы">
      <defs>
        <marker id="arr" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
          <path d="M0,0 L10,5 L0,10 z" fill="#8aa0c7" />
        </marker>
        <pattern id="gridp" width="20" height="20" patternUnits="userSpaceOnUse">
          <path d="M20 0 L0 0 0 20" fill="none" stroke="#16223b" strokeWidth="1" />
        </pattern>
      </defs>
      <rect x="0" y="0" width="1000" height="580" fill="url(#gridp)" rx="12" />
      {EDGES.map(([a, b, label], i) => {
        const A = byId[a], B = byId[b];
        const [x1, y1] = anchor(A, B), [x2, y2] = anchor(B, A);
        const hot = hover && (hover === a || hover === b);
        return (
          <g key={i}>
            <line x1={x1} y1={y1} x2={x2} y2={y2} stroke={hot ? '#38bdf8' : '#5a6f96'} strokeWidth={hot ? 2.4 : 1.4} markerEnd="url(#arr)" />
            {label && Math.hypot(x2 - x1, y2 - y1) > 70 ? (
              <text x={(x1 + x2) / 2 + 4} y={(y1 + y2) / 2 - 4} fill={hot ? '#bae6fd' : '#8aa0c7'} fontSize="10.5">{label}</text>
            ) : null}
          </g>
        );
      })}
      {NODES.map((n) => {
        const st = STYLE[n.kind];
        const clickable = !!n.href;
        const isHover = hover === n.id;
        const stroke = STAGE_COLORS[n.id] || st.stroke;
        return (
          <g
            key={n.id}
            onMouseEnter={() => setHover(n.id)}
            onMouseLeave={() => setHover(null)}
            onClick={() => clickable && router.push(n.href)}
            style={{ cursor: clickable ? 'pointer' : 'default' }}
          >
            <rect x={n.x} y={n.y} width={n.w} height={n.h} rx="10" fill={st.fill} stroke={stroke} strokeWidth={isHover && clickable ? 3 : 1.8} />
            {n.lines.map((l, i) => (
              <text
                key={i}
                x={n.x + n.w / 2}
                y={n.y + n.h / 2 + (i - (n.lines.length - 1) / 2) * 17 + 5}
                textAnchor="middle"
                fill={st.text}
                fontSize={i === 0 && n.kind === 'stage' ? 15 : n.kind.startsWith('core') ? 17 : 13.5}
                fontWeight={i === 0 && (n.kind === 'stage' || n.kind.startsWith('core') || n.kind === 'extra') ? 700 : 500}
              >
                {l}
              </text>
            ))}
            {clickable ? (
              <text x={n.x + n.w - 10} y={n.y + 16} textAnchor="end" fill={stroke} fontSize="13">↗</text>
            ) : null}
          </g>
        );
      })}
    </svg>
  );
}
