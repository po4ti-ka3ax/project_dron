'use client';
import { useEffect, useMemo, useRef, useState } from 'react';
import { StageHeader, Tasks, SectionTitle, Pager, Slider, Metric, Code } from '@/components/UI';
import { LineChart } from '@/components/Charts';
import { rng, fmt, enuToGeodetic } from '@/lib/math';
import { generateField } from '@/lib/scene';
import { AREA } from '@/lib/pages';

const W = 750, H = 500;
const CAM = { f: 8.8, sw: 13.2, iw: 5472 }; // мм, мм, px (камера 1")
export const CLASSES = {
  person: { ru: 'человек', color: '#f472b6', size: [9, 9] },
  car: { ru: 'автомобиль', color: '#38bdf8', size: [16, 30] },
  truck: { ru: 'трактор/грузовик', color: '#fbbf24', size: [24, 42] },
  cow: { ru: 'животное', color: '#fb923c', size: [14, 24] },
  tree: { ru: 'дерево (препятствие)', color: '#4ade80', size: [34, 34] },
  pole: { ru: 'опора ЛЭП', color: '#f87171', size: [18, 18] },
};

export function makeScene(seed) {
  const r = rng(seed);
  const objs = [];
  const add = (cls, n) => {
    for (let k = 0; k < n; k++) {
      const [w0, h0] = CLASSES[cls].size;
      const rot = r() < 0.5;
      const w = (rot ? h0 : w0) * r.range(0.9, 1.15), h = (rot ? w0 : h0) * r.range(0.9, 1.15);
      let x, y, ok = false, tries = 0;
      while (!ok && tries++ < 50) {
        x = r.range(20, W - 20 - w); y = r.range(20, H - 20 - h);
        ok = objs.every((o) => x + w < o.x - 6 || x > o.x + o.w + 6 || y + h < o.y - 6 || y > o.y + o.h + 6);
      }
      objs.push({ cls, x, y, w, h, id: objs.length });
    }
  };
  add('tree', r.int(3, 5)); add('pole', 2); add('truck', r.int(1, 2)); add('car', r.int(3, 5)); add('cow', r.int(2, 4)); add('person', r.int(3, 6));
  return objs;
}

export function drawObject(ctx, o) {
  const cx = o.x + o.w / 2, cy = o.y + o.h / 2;
  ctx.save();
  ctx.fillStyle = 'rgba(0,0,0,.35)';
  ctx.beginPath(); ctx.ellipse(cx + 4, cy + 4, o.w / 2, o.h / 2, 0, 0, 7); ctx.fill();
  if (o.cls === 'tree') {
    ctx.fillStyle = '#14532d'; ctx.beginPath(); ctx.arc(cx, cy, o.w / 2, 0, 7); ctx.fill();
    ctx.fillStyle = '#166534'; ctx.beginPath(); ctx.arc(cx - 4, cy - 4, o.w / 3, 0, 7); ctx.fill();
  } else if (o.cls === 'pole') {
    ctx.strokeStyle = '#cbd5e1'; ctx.lineWidth = 2;
    ctx.strokeRect(o.x, o.y, o.w, o.h); ctx.beginPath(); ctx.moveTo(o.x, o.y); ctx.lineTo(o.x + o.w, o.y + o.h); ctx.moveTo(o.x + o.w, o.y); ctx.lineTo(o.x, o.y + o.h); ctx.stroke();
  } else if (o.cls === 'person') {
    ctx.fillStyle = '#1e293b'; ctx.beginPath(); ctx.arc(cx, cy, o.w / 2, 0, 7); ctx.fill();
    ctx.fillStyle = '#fca5a5'; ctx.beginPath(); ctx.arc(cx, cy, o.w / 4, 0, 7); ctx.fill();
  } else if (o.cls === 'cow') {
    ctx.fillStyle = '#7c4a1e'; ctx.beginPath(); ctx.ellipse(cx, cy, o.w / 2, o.h / 2, 0, 0, 7); ctx.fill();
    ctx.fillStyle = '#f5f5f4'; ctx.beginPath(); ctx.arc(cx - o.w / 6, cy, o.w / 6, 0, 7); ctx.fill();
  } else {
    ctx.fillStyle = o.cls === 'car' ? ['#dc2626', '#e5e7eb', '#1d4ed8', '#111827'][o.id % 4] : '#eab308';
    ctx.fillRect(o.x, o.y, o.w, o.h);
    ctx.fillStyle = 'rgba(148,197,255,.55)';
    if (o.h > o.w) ctx.fillRect(o.x + 2, o.y + o.h * 0.2, o.w - 4, o.h * 0.22); else ctx.fillRect(o.x + o.w * 0.2, o.y + 2, o.w * 0.22, o.h - 4);
  }
  ctx.restore();
}

export function iou(a, b) {
  const x0 = Math.max(a.x, b.x), y0 = Math.max(a.y, b.y), x1 = Math.min(a.x + a.w, b.x + b.w), y1 = Math.min(a.y + a.h, b.y + b.h);
  const inter = Math.max(0, x1 - x0) * Math.max(0, y1 - y0);
  return inter / (a.w * a.h + b.w * b.h - inter);
}

// Модель выхода детектора YOLOv8 (сырые предсказания до порога и NMS)
export function rawPredictions(objs, seed) {
  const r = rng(seed * 31 + 5);
  const preds = [];
  for (const o of objs) {
    const area = o.w * o.h;
    const base = Math.min(0.92, 0.22 + 0.62 * Math.min(1, Math.sqrt(area) / 32));
    const conf = Math.max(0.05, Math.min(0.98, base + r.gauss(0, 0.14)));
    const loc = r() < 0.12 ? 0.3 : 0.06; // иногда — плохая локализация рамки
    const j = () => r.gauss(0, loc);
    const box = { x: o.x + o.w * j(), y: o.y + o.h * j(), w: o.w * (1 + j()), h: o.h * (1 + j()) };
    preds.push({ ...box, cls: o.cls, conf });
    if (r() < 0.45) preds.push({ x: box.x + o.w * 0.12, y: box.y - o.h * 0.1, w: box.w * 0.95, h: box.h * 1.05, cls: o.cls, conf: conf * r.range(0.55, 0.9) });
  }
  const keys = Object.keys(CLASSES);
  for (let k = 0; k < 5; k++) {
    const s = r.range(10, 30);
    preds.push({ x: r.range(0, W - s), y: r.range(0, H - s), w: s, h: s * r.range(0.8, 1.6), cls: keys[r.int(0, keys.length - 1)], conf: r.range(0.08, 0.48) });
  }
  return preds;
}

export function nms(preds, thr, iouThr) {
  const kept = [];
  const sorted = preds.filter((p) => p.conf >= thr).sort((a, b) => b.conf - a.conf);
  for (const p of sorted) if (kept.every((k) => k.cls !== p.cls || iou(k, p) < iouThr)) kept.push(p);
  return kept;
}

export function evaluate(dets, objs) {
  const used = new Set();
  let tp = 0;
  const marks = dets.map((d) => {
    let best = -1, bi = 0;
    objs.forEach((o, i) => { if (!used.has(i) && o.cls === d.cls) { const v = iou(d, o); if (v > bi) { bi = v; best = i; } } });
    if (bi >= 0.5) { used.add(best); tp++; return true; }
    return false;
  });
  return { tp, fp: dets.length - tp, fn: objs.length - tp, marks };
}

function prCurve(preds, objs, iouThr) {
  const pts = [];
  for (let t = 0.95; t >= 0.04; t -= 0.03) {
    const d = nms(preds, t, iouThr);
    const e = evaluate(d, objs);
    pts.push([e.tp / (objs.length || 1), d.length ? e.tp / d.length : 1]);
  }
  // AP: площадь под монотонной огибающей PR
  let ap = 0, prevR = 0;
  const env = pts.map((p, i) => [p[0], Math.max(...pts.slice(i).map((q) => q[1]))]);
  for (const [rc, pr] of env) { if (rc > prevR) { ap += (rc - prevR) * pr; prevR = rc; } }
  return { pts: env, ap };
}

export default function Stage4() {
  const [seed, setSeed] = useState(4);
  const [thr, setThr] = useState(0.4);
  const [iouThr, setIouThr] = useState(0.5);
  const [alt, setAlt] = useState(80);
  const [heading, setHeading] = useState(0);
  const [showGT, setShowGT] = useState(false);
  const canvas = useRef(null);
  const bg = useMemo(() => generateField(375, 250, seed + 20, { nStress: 3, stressLevel: 0.6 }), [seed]);
  const objs = useMemo(() => makeScene(seed), [seed]);
  const raw = useMemo(() => rawPredictions(objs, seed), [objs, seed]);
  const dets = useMemo(() => nms(raw, thr, iouThr), [raw, thr, iouThr]);
  const ev = useMemo(() => evaluate(dets, objs), [dets, objs]);
  const pr = useMemo(() => prCurve(raw, objs, iouThr), [raw, objs, iouThr]);

  const gsd = (CAM.sw * alt * 100) / (CAM.f * CAM.iw); // см/пиксель исходного снимка
  const scale = CAM.iw / W;
  const located = dets.map((d) => {
    const u = d.x + d.w / 2 - W / 2, v = d.y + d.h / 2 - H / 2;
    const xr = (u * scale * gsd) / 100, yr = (-v * scale * gsd) / 100;
    const th = (heading * Math.PI) / 180;
    const e = xr * Math.cos(th) + yr * Math.sin(th), n = -xr * Math.sin(th) + yr * Math.cos(th);
    const [lat, lon] = enuToGeodetic(e, n, 0, AREA.lat0, AREA.lon0, AREA.h0);
    const obstacle = ['tree', 'pole', 'truck'].includes(d.cls) && Math.abs(xr) < 15 + (d.w * scale * gsd) / 200 && yr > -5;
    return { ...d, e, n, lat, lon, dist: Math.hypot(xr, yr), obstacle, sizeM: (Math.max(d.w, d.h) * scale * gsd) / 100 };
  });

  useEffect(() => {
    const c = canvas.current;
    if (!c) return;
    c.width = W; c.height = H;
    const ctx = c.getContext('2d');
    const off = document.createElement('canvas');
    off.width = bg.w; off.height = bg.h;
    off.getContext('2d').putImageData(new ImageData(new Uint8ClampedArray(bg.rgba), bg.w, bg.h), 0, 0);
    ctx.imageSmoothingEnabled = true;
    ctx.drawImage(off, 0, 0, W, H);
    objs.forEach((o) => drawObject(ctx, o));
    // коридор полёта
    const corr = (15 * 100) / (gsd * scale);
    ctx.fillStyle = 'rgba(56,189,248,.08)'; ctx.fillRect(W / 2 - corr, 0, corr * 2, H / 2);
    ctx.strokeStyle = 'rgba(56,189,248,.5)'; ctx.setLineDash([6, 6]); ctx.strokeRect(W / 2 - corr, 0, corr * 2, H / 2); ctx.setLineDash([]);
    ctx.strokeStyle = '#fff'; ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(W / 2 - 10, H / 2); ctx.lineTo(W / 2 + 10, H / 2); ctx.moveTo(W / 2, H / 2 - 10); ctx.lineTo(W / 2, H / 2 + 10); ctx.stroke();
    if (showGT) objs.forEach((o) => { ctx.strokeStyle = '#ffffff'; ctx.setLineDash([3, 3]); ctx.strokeRect(o.x, o.y, o.w, o.h); ctx.setLineDash([]); });
    located.forEach((d, i) => {
      const col = d.obstacle ? '#ef4444' : CLASSES[d.cls].color;
      ctx.strokeStyle = col; ctx.lineWidth = d.obstacle ? 3 : 2;
      ctx.strokeRect(d.x, d.y, d.w, d.h);
      const label = `${d.cls} ${d.conf.toFixed(2)}`;
      ctx.font = 'bold 11px sans-serif';
      const tw = ctx.measureText(label).width + 6;
      ctx.fillStyle = col; ctx.fillRect(d.x, d.y - 14, tw, 14);
      ctx.fillStyle = '#0b1220'; ctx.fillText(label, d.x + 3, d.y - 3);
      if (!ev.marks[i]) { ctx.fillStyle = '#fff'; ctx.fillText('FP', d.x + d.w + 2, d.y + 10); }
    });
    ctx.fillStyle = 'rgba(0,0,0,.55)'; ctx.fillRect(8, H - 28, 330, 20);
    ctx.fillStyle = '#e6ecf7'; ctx.font = '12px monospace';
    ctx.fillText(`YOLOv8n · 640px · H=${alt} м · GSD=${gsd.toFixed(2)} см/px`, 14, H - 14);
  }, [bg, objs, located, showGT, ev, alt, gsd, scale]);

  const prec = dets.length ? ev.tp / dets.length : 0, rec = ev.tp / objs.length;
  return (
    <>
      <StageHeader
        num={4}
        title="Обнаружение объектов (на основе БПЛА и ГИС)"
        subtitle="Распознавание объектов на кадре с камеры нейронной сетью YOLOv8, подавление дублей (NMS), расчёт географических координат каждого объекта и выявление препятствий на курсе полёта."
        tags={['YOLOv8 (CNN, one-stage)', 'NMS', 'IoU / mAP@0.5', 'GSD → WGS-84']}
      />
      <div className="grid g2">
        <div className="card"><h3>Выполняемые задачи</h3><Tasks items={['Распознавание объектов с помощью камеры', 'Применение нейронной сети YOLOv8', 'Определение координат объекта', 'Обнаружение препятствий']} /></div>
        <div className="card">
          <h3>Как работает YOLOv8</h3>
          <p className="small" style={{ margin: 0 }}>Кадр 640×640 проходит через свёрточный backbone CSPDarknet (C2f-блоки), шею PAN-FPN и безъякорную (anchor-free) голову, которая на трёх масштабах предсказывает класс и рамку. Затем отбрасываются предсказания с уверенностью ниже порога, а перекрывающиеся рамки одного класса удаляются NMS по IoU.</p>
        </div>
      </div>

      <SectionTitle n="4.1">Детекция на кадре с БПЛА</SectionTitle>
      <div className="note warn" style={{ marginBottom: 12 }}>
        В браузере работает модель выхода детектора: сырые предсказания (рамки, классы, уверенности, дубли и ложные срабатывания) формируются по разметке сцены, а порог уверенности, NMS, метрики (Precision, Recall, AP) и пересчёт координат вычисляются честно. Полный код инференса и обучения YOLOv8 (Ultralytics) приведён ниже.
      </div>
      <div className="grid g-side">
        <div className="card tight">
          <canvas ref={canvas} style={{ width: '100%' }} />
          <div className="legend">
            {Object.entries(CLASSES).map(([k, c]) => <span key={k}><i style={{ background: c.color }} />{c.ru}</span>)}
            <span><i style={{ background: '#ef4444', height: 6 }} />препятствие на курсе</span>
          </div>
        </div>
        <div className="card tight controls">
          <div className="row">
            <button className="btn small primary" onClick={() => setSeed((s) => s + 1)} type="button">Новый кадр</button>
            <label className="row small"><input type="checkbox" checked={showGT} onChange={(e) => setShowGT(e.target.checked)} /> Разметка (GT)</label>
          </div>
          <Slider label="Порог уверенности conf" value={thr} min={0.05} max={0.9} step={0.05} onChange={setThr} />
          <Slider label="Порог IoU для NMS" value={iouThr} min={0.2} max={0.9} step={0.05} onChange={setIouThr} />
          <Slider label="Высота полёта, м" value={alt} min={30} max={150} step={5} onChange={setAlt} />
          <Slider label="Курс БПЛА, °" value={heading} min={0} max={355} step={5} onChange={setHeading} />
          <div className="metrics">
            <Metric k="Сырых предсказаний" v={raw.length} />
            <Metric k="После порога и NMS" v={dets.length} />
            <Metric k="Precision" v={fmt(prec)} color="#4ade80" />
            <Metric k="Recall" v={fmt(rec)} color="#38bdf8" />
            <Metric k="F1" v={fmt((2 * prec * rec) / (prec + rec || 1))} />
            <Metric k="AP@0.5 (кадр)" v={fmt(pr.ap)} color="#fbbf24" />
          </div>
        </div>
      </div>

      <div className="grid g2 section">
        <div className="card table-wrap">
          <h3>Координаты обнаруженных объектов</h3>
          <table>
            <thead><tr><th>Класс</th><th className="num">conf</th><th className="num">E, м</th><th className="num">N, м</th><th className="num">Широта</th><th className="num">Долгота</th><th className="num">Размер, м</th></tr></thead>
            <tbody>
              {located.map((d, i) => (
                <tr key={i} style={d.obstacle ? { background: '#3b141455' } : null}>
                  <td>{CLASSES[d.cls].ru} {d.obstacle && <span className="badge bad">на курсе</span>}{!ev.marks[i] && <span className="badge warn">FP</span>}</td>
                  <td className="num">{d.conf.toFixed(2)}</td>
                  <td className="num">{d.e.toFixed(1)}</td>
                  <td className="num">{d.n.toFixed(1)}</td>
                  <td className="num">{d.lat.toFixed(6)}</td>
                  <td className="num">{d.lon.toFixed(6)}</td>
                  <td className="num">{d.sizeM.toFixed(1)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <div className="card">
          <h3>Кривая Precision–Recall</h3>
          <LineChart height={230} xMin={0} xMax={1} yMin={0} yMax={1.02} xLabel="Recall" yLabel="Precision" series={[{ data: pr.pts, color: '#fbbf24', name: `AP@0.5 = ${pr.ap.toFixed(3)}` }]} />
          <h3 style={{ marginTop: 12 }}>Расчёт координат объекта</h3>
          <p className="small" style={{ margin: 0 }}>
            GSD = (ширина матрицы · H) / (f · ширина кадра) = ({CAM.sw} мм · {alt} м) / ({CAM.f} мм · {CAM.iw} px) = <b>{gsd.toFixed(2)} см/px</b>. Смещение центра рамки от центра кадра (пиксели) × GSD даёт смещение в метрах; поворот на курс БПЛА переводит его в восток/север (ENU), а преобразование ENU → WGS-84 (этап 6) — в широту/долготу. Опорная точка — положение БПЛА {AREA.lat0}°, {AREA.lon0}°.
          </p>
        </div>
      </div>

      <SectionTitle n="4.2">Программная реализация (Ultralytics YOLOv8)</SectionTitle>
      <div className="grid g2">
        <Code title="detect.py" code={`
from ultralytics import YOLO
import cv2, numpy as np

model = YOLO('yolov8n.pt')            # или best.pt после дообучения на VisDrone
F_MM, SENSOR_W, IMG_W = 8.8, 13.2, 5472

def detect(frame, conf=0.4, iou=0.5):
    res = model.predict(frame, imgsz=640, conf=conf, iou=iou, verbose=False)[0]
    return [(res.names[int(c)], float(p), b.tolist())
            for b, p, c in zip(res.boxes.xyxy, res.boxes.conf, res.boxes.cls)]

def pixel_to_enu(u, v, img_w, img_h, alt, heading_deg):
    gsd = SENSOR_W * alt / (F_MM * IMG_W) * (IMG_W / img_w)   # м/пиксель кадра
    xr, yr = (u - img_w / 2) * gsd, -(v - img_h / 2) * gsd
    th = np.radians(heading_deg)
    return xr*np.cos(th) + yr*np.sin(th), -xr*np.sin(th) + yr*np.cos(th)

frame = cv2.imread('frame_0421.jpg')
for name, p, (x0, y0, x1, y1) in detect(frame):
    e, n = pixel_to_enu((x0+x1)/2, (y0+y1)/2, frame.shape[1], frame.shape[0], alt=80, heading_deg=0)
    lat, lon, _ = enu_to_geodetic(e, n, 0, LAT0, LON0, H0)   # см. этап 6
    print(f'{name:10s} {p:.2f}  E={e:7.1f}  N={n:7.1f}  {lat:.6f} {lon:.6f}')
`} />
        <Code title="train.py" code={`
from ultralytics import YOLO

# data.yaml: path: VisDrone, train/val, names: {0: pedestrian, 1: people, 2: bicycle,
#            3: car, 4: van, 5: truck, 6: tricycle, 7: awning-tricycle, 8: bus, 9: motor}
model = YOLO('yolov8n.pt')                   # перенос обучения с COCO
model.train(data='VisDrone.yaml', epochs=100, imgsz=1280, batch=16,
            mosaic=1.0, scale=0.5, degrees=10, flipud=0.5,   # аугментации для вида сверху
            optimizer='AdamW', lr0=1e-3, patience=20)
metrics = model.val()
print('mAP50 =', metrics.box.map50, ' mAP50-95 =', metrics.box.map)
model.export(format='onnx')                  # для бортового компьютера (Jetson)
`} />
      </div>
      <div className="note section">
        <b>Вывод.</b> Повышение порога уверенности увеличивает Precision, но снижает Recall (мелкие объекты — люди — теряются первыми). NMS убирает дублирующие рамки. С ростом высоты полёта GSD растёт линейно, поэтому погрешность координат увеличивается; объекты-препятствия в коридоре ±15 м по курсу передаются в модуль принятия решений (этап 7).
      </div>
      <Pager href="/etap-4" />
    </>
  );
}
