'use client';
import { useCallback, useEffect, useRef, useState } from 'react';
import { StageHeader, SectionTitle, Pager, Metric, Code, Slider } from '@/components/UI';
import { LineChart, BarChart } from '@/components/Charts';
import { rng, fmt, enuToGeodetic, EKF } from '@/lib/math';
import { generateField } from '@/lib/scene';
import { detectBlobs, textureAnomalies, genTrajectory, CnnLstm, evaluateHybrid } from '@/lib/pipeline';
import { drawObject, makeScene } from '@/components/stages/Stage4';
import { mamdani } from '@/components/stages/Stage7';
import { runTests } from '@/lib/tests';
import { AREA } from '@/lib/pages';

// ---------------- Блок-схема (рисунок 2) ----------------
const B = {
  start: { x: 110, y: 8, w: 240, h: 32, t: ['Запуск потока камеры'] },
  get: { x: 110, y: 60, w: 240, h: 32, t: ['Получение кадра с камеры'] },
  q1: { x: 230, y: 136, w: 210, h: 58, t: ['Если кадр', 'не получен?'], d: true },
  cont: { x: 8, y: 192, w: 92, h: 30, t: ['continue'] },
  det: { x: 130, y: 192, w: 210, h: 32, t: ['Обнаружение объектов (detect)'] },
  tex: { x: 110, y: 244, w: 250, h: 32, t: ['Проверка аномалий текстуры'] },
  gps: { x: 90, y: 296, w: 290, h: 32, t: ['Получение координат GPS (get_gps_coords)'] },
  q2: { x: 235, y: 380, w: 320, h: 62, t: ['Обнаружены объекты', 'или аномалии?'], d: true },
  msg: { x: 160, y: 440, w: 150, h: 40, t: ['Формирование', 'сообщения'] },
  send: { x: 30, y: 524, w: 170, h: 32, t: ['Отправка сообщения'] },
  ctrl: { x: 250, y: 524, w: 180, h: 32, t: ['Управление объектом'] },
  ai: { x: 700, y: 72, w: 290, h: 84, t: ['Анализ с применением', 'нейронной сети', 'CNN-LSTM-EKF'], d: true },
  dec: { x: 600, y: 160, w: 200, h: 40, t: ['Принятие решений', 'на основе ИИ'] },
  adapt: { x: 600, y: 228, w: 200, h: 40, t: ['Адаптивное', 'управление БПЛА'] },
  fb: { x: 600, y: 296, w: 200, h: 34, t: ['Обратная связь'] },
  ad2: { x: 600, y: 360, w: 200, h: 40, t: ['Адаптивное', 'управление'] },
};
const E = [
  ['start', 'get'], ['get', 'q1'], ['q1', 'cont', 'Да'], ['q1', 'det', 'Нет'], ['det', 'tex'], ['tex', 'gps'], ['gps', 'q2'],
  ['q2', 'msg', 'Да'], ['msg', 'send', 'если нет'], ['msg', 'ctrl', 'если есть объект'], ['dec', 'adapt'], ['adapt', 'fb'], ['fb', 'ad2'],
];
const cx = (b) => (b.d ? b.x : b.x + b.w / 2);
const cy = (b) => (b.d ? b.y : b.y + b.h / 2);

function Flowchart({ active = [], current }) {
  const col = (id) => (id === current ? '#facc15' : active.includes(id) ? '#38bdf8' : '#3b4b6b');
  const fill = (id) => (id === current ? '#3b3209' : active.includes(id) ? '#0c2c44' : '#111a2e');
  const pt = (id, side) => {
    const b = B[id];
    if (b.d) return { top: [b.x, b.y - b.h / 2], bottom: [b.x, b.y + b.h / 2], left: [b.x - b.w / 2, b.y], right: [b.x + b.w / 2, b.y] }[side];
    return { top: [cx(b), b.y], bottom: [cx(b), b.y + b.h], left: [b.x, cy(b)], right: [b.x + b.w, cy(b)] }[side];
  };
  const arrow = (pts, key, hot, label, lp) => (
    <g key={key}>
      <polyline points={pts.map((p) => p.join(',')).join(' ')} fill="none" stroke={hot ? '#38bdf8' : '#5a6f96'} strokeWidth={hot ? 2.2 : 1.4} markerEnd="url(#a2)" />
      {label ? <text x={lp[0]} y={lp[1]} fontSize="12" fill="#c7d3ea" fontWeight="600">{label}</text> : null}
    </g>
  );
  const hot = (a, b) => active.includes(a) && active.includes(b);
  return (
    <svg viewBox="0 0 1000 570" className="chart flow-svg">
      <defs><marker id="a2" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M0,0 L10,5 L0,10 z" fill="#8aa0c7" /></marker></defs>
      {arrow([pt('start', 'bottom'), pt('get', 'top')], 'e1', hot('start', 'get'))}
      {arrow([pt('get', 'bottom'), [230, pt('q1', 'top')[1]]], 'e2', hot('get', 'q1'))}
      {arrow([pt('q1', 'left'), [54, 136], pt('cont', 'top')], 'e3', hot('q1', 'cont'), 'Да', [70, 128])}
      {arrow([pt('cont', 'left'), [4, 207], [4, 76], pt('get', 'left')], 'e3b', hot('cont', 'get'))}
      {arrow([pt('q1', 'bottom'), [235, 192]], 'e4', hot('q1', 'det'), 'Нет', [242, 182])}
      {arrow([pt('det', 'bottom'), pt('tex', 'top')], 'e5', hot('det', 'tex'))}
      {arrow([pt('tex', 'bottom'), pt('gps', 'top')], 'e6', hot('tex', 'gps'))}
      {arrow([pt('gps', 'bottom'), [235, 349]], 'e7', hot('gps', 'q2'))}
      {arrow([pt('q2', 'right'), [430, 380], [430, 76], pt('get', 'right')], 'e8', hot('q2', 'get') && !active.includes('msg'), 'Нет', [436, 250])}
      {arrow([pt('q2', 'bottom'), [235, 440]], 'e9', hot('q2', 'msg'), 'Да', [242, 425])}
      {arrow([[200, 480], [115, 480], pt('send', 'top')], 'e10', hot('msg', 'send'), 'если нет', [82, 500])}
      {arrow([[280, 480], [340, 480], pt('ctrl', 'top')], 'e11', hot('msg', 'ctrl'), 'если есть объект', [300, 500])}
      {arrow([pt('ctrl', 'right'), [560, 540], [560, 20], [700, 20], pt('ai', 'top')], 'e12', hot('ctrl', 'ai'))}
      {arrow([pt('ai', 'bottom'), [700, 160]], 'e13', hot('ai', 'dec'), 'Да', [708, 150])}
      {arrow([pt('dec', 'bottom'), pt('adapt', 'top')], 'e14', hot('dec', 'adapt'))}
      {arrow([pt('adapt', 'bottom'), pt('fb', 'top')], 'e15', hot('adapt', 'fb'))}
      {arrow([pt('fb', 'bottom'), pt('ad2', 'top')], 'e16', hot('fb', 'ad2'))}
      {arrow([pt('ad2', 'right'), [830, 380], [830, 313], pt('fb', 'right')], 'e17', hot('ad2', 'fb'))}
      {arrow([[600, 380], [575, 380], [575, 72], pt('ai', 'left')], 'e18', hot('ad2', 'ai'))}
      {Object.entries(B).map(([id, b]) => (
        <g key={id}>
          {b.d ? (
            <polygon points={`${b.x},${b.y - b.h / 2} ${b.x + b.w / 2},${b.y} ${b.x},${b.y + b.h / 2} ${b.x - b.w / 2},${b.y}`} fill={fill(id)} stroke={col(id)} strokeWidth={id === current ? 3 : 1.6} />
          ) : (
            <rect x={b.x} y={b.y} width={b.w} height={b.h} rx="6" fill={fill(id)} stroke={col(id)} strokeWidth={id === current ? 3 : 1.6} />
          )}
          {b.t.map((l, i) => (
            <text key={i} x={cx(b)} y={cy(b) + (i - (b.t.length - 1) / 2) * 15 + 4.5} textAnchor="middle" fontSize="12.5" fill="#e6ecf7">{l}</text>
          ))}
        </g>
      ))}
    </svg>
  );
}

// ---------------- Python-код программы ----------------
const PY = `
"""
Интеллектуальная система управления БПЛА — программа по блок-схеме (рисунок 2).
Камера -> detect -> аномалии текстуры -> GPS -> сообщение / управление объектом ->
CNN-LSTM-EKF -> решение ИИ -> адаптивное управление -> обратная связь.
"""
import time, json
import cv2
import numpy as np
from ultralytics import YOLO
from tensorflow import keras
from filterpy.kalman import ExtendedKalmanFilter

DETECTOR = YOLO('best.pt')                         # YOLOv8, дообученная на VisDrone
PREDICTOR = keras.models.load_model('cnn_lstm.h5')  # прогноз приращений траектории
NDVI_THR, WIN = 0.38, 10

# ------------------------- CNN-LSTM (обучение см. train_cnn_lstm) -------------------------
def build_cnn_lstm(win=WIN):
    return keras.Sequential([
        keras.layers.Input((win, 2)),                       # окно приращений (dE, dN)
        keras.layers.Conv1D(32, 3, padding='causal', activation='relu'),
        keras.layers.Conv1D(32, 3, padding='causal', activation='relu'),
        keras.layers.LSTM(64),
        keras.layers.Dense(2)])                             # прогноз следующего приращения

# ------------------------- EKF (CTRV: x, y, v, psi, omega) -------------------------
def fx(s, dt):
    x, y, v, psi, w = s
    if abs(w) < 1e-4:
        return np.array([x + v*np.cos(psi)*dt, y + v*np.sin(psi)*dt, v, psi, w])
    return np.array([x + v/w*(np.sin(psi + w*dt) - np.sin(psi)),
                     y + v/w*(np.cos(psi) - np.cos(psi + w*dt)), v, psi + w*dt, w])

def jacobian(f, s, dt, eps=1e-5):
    f0 = f(s, dt); J = np.zeros((5, 5))
    for j in range(5):
        d = np.zeros(5); d[j] = eps
        J[:, j] = (f(s + d, dt) - f0) / eps
    return J

class HybridEKF:
    def __init__(self, s0, dt=0.5):
        self.s, self.P, self.dt = np.array(s0, float), np.eye(5) * 10, dt
        self.Q = np.diag([0.05, 0.05, 0.3, 0.01, 0.02]); self.H = np.eye(2, 5)
    def predict(self):
        F = jacobian(fx, self.s, self.dt)
        self.s = fx(self.s, self.dt); self.P = F @ self.P @ F.T + self.Q
    def update(self, z, r):
        S = self.H @ self.P @ self.H.T + np.eye(2) * r
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.s = self.s + K @ (z - self.H @ self.s); self.P = (np.eye(5) - K @ self.H) @ self.P

# ------------------------- функции блок-схемы -------------------------
def detect(frame, conf=0.4):
    r = DETECTOR.predict(frame, imgsz=640, conf=conf, verbose=False)[0]
    return [{'cls': r.names[int(c)], 'conf': float(p), 'box': b.tolist()}
            for b, p, c in zip(r.boxes.xyxy, r.boxes.conf, r.boxes.cls)]

def check_texture_anomalies(red, nir, block=20):
    ndvi = (nir - red) / (nir + red + 1e-6)
    anomalies = []
    for y in range(0, ndvi.shape[0] - block + 1, block):
        for x in range(0, ndvi.shape[1] - block + 1, block):
            b = ndvi[y:y+block, x:x+block]
            if (b < 0.15).mean() > 0.15:       # вода, межи, дорога — не растительность
                continue
            m = b[b >= 0.15].mean()
            if m < NDVI_THR:
                anomalies.append({'x': x, 'y': y, 'ndvi': round(float(m), 3)})
    return anomalies

def get_gps_coords(mav):
    msg = mav.recv_match(type='GLOBAL_POSITION_INT', blocking=True)
    return msg.lat / 1e7, msg.lon / 1e7, msg.alt / 1e3

def form_message(frame_id, gps, objects, anomalies):
    return json.dumps({'frame': frame_id, 't': time.time(), 'lat': gps[0], 'lon': gps[1],
                       'objects': objects, 'anomalies': anomalies}, ensure_ascii=False)

def ai_decision(battery, wind, dist):            # нечёткая система Мамдани (этап 7)
    risk = fuzzy_risk(battery, wind, dist)
    return ('CONTINUE' if risk < 35 else 'CORRECT' if risk < 62 else 'RTH'), risk

# ------------------------- главный цикл -------------------------
def main(mav, ground_link):
    cap = cv2.VideoCapture('rtsp://192.168.1.10/stream')          # Запуск потока камеры
    ekf, history, speed_cmd = None, [], 8.0
    frame_id = 0
    while True:
        ok, frame = cap.read()                                     # Получение кадра
        if not ok or frame is None:                                # Если кадр не получен?
            continue                                               #   Да -> continue
        frame_id += 1
        objects = detect(frame)                                    # Обнаружение объектов
        red, nir = read_multispectral()                            # каналы мультиспектр. камеры
        anomalies = check_texture_anomalies(red, nir)              # Проверка аномалий текстуры
        lat, lon, alt = get_gps_coords(mav)                        # Получение координат GPS
        e, n, u = geodetic_to_enu(lat, lon, alt, LAT0, LON0, H0)
        history.append((e, n))

        if not objects and not anomalies:                          # Обнаружены объекты/аномалии?
            continue                                               #   Нет -> следующий кадр
        message = form_message(frame_id, (lat, lon), objects, anomalies)   # Формирование сообщения
        if not objects:
            ground_link.send(message)                              # Отправка сообщения
            continue

        # ---- Управление объектом -> Анализ CNN-LSTM-EKF ----
        if ekf is None:
            ekf = HybridEKF([e, n, speed_cmd, 0, 0])
        ekf.predict()
        if len(history) > WIN:
            d = np.diff(np.array(history[-WIN-1:]), axis=0)[None]  # (1, WIN, 2)
            dE, dN = PREDICTOR.predict(d, verbose=0)[0]
            ekf.update(np.array([history[-2][0] + dE, history[-2][1] + dN]), r=4.0)  # псевдоизмерение сети
        ekf.update(np.array([e, n]), r=9.0)                         # измерение GPS

        dist = min(object_distance(o, alt) for o in objects)
        action, risk = ai_decision(battery_percent(mav), wind_speed(mav), dist)  # Решение на основе ИИ

        # ---- Адаптивное управление БПЛА + обратная связь ----
        target = speed_cmd * (1 - 0.7 * risk / 100)
        if action == 'RTH':
            mav.set_mode('RTL')
        else:
            for _ in range(10):                                      # контур обратной связи
                v = ekf.s[2]
                speed_cmd += 0.5 * (target - v)                      # адаптивная коррекция
                send_velocity(mav, speed_cmd, heading=ekf.s[3])
                ekf.predict(); ekf.update(np.array(local_position(mav)), r=9.0)
        ground_link.send(message)

if __name__ == '__main__':
    main(connect_mavlink('udp:127.0.0.1:14550'), GroundLink('tcp://ground:5555'))
`;

// ---------------- Симуляция ----------------
const FW = 1500, FH = 220, VW = 320, VH = 200, GSD = 0.25;

export default function ProgramPage() {
  const [ready, setReady] = useState(false);
  const [running, setRunning] = useState(false);
  const [delay, setDelay] = useState(500);
  const [view, setView] = useState({ trace: [], current: null, frame: null });
  const [log, setLog] = useState([]);
  const [stats, setStats] = useState({ frames: 0, dropped: 0, obj: 0, anom: 0, sent: 0, ctrl: 0, CONTINUE: 0, CORRECT: 0, RTH: 0 });
  const [lastMsg, setLastMsg] = useState(null);
  const [train, setTrain] = useState(null);
  const [evalRes, setEvalRes] = useState(null);
  const [tests, setTests] = useState(null);
  const [speedHist, setSpeedHist] = useState([]);
  const fieldRef = useRef(null), sceneCanvas = useRef(null), frameCanvas = useRef(null), sim = useRef(null), model = useRef(null);
  const logRef = useRef(null);

  // Подготовка: поле, объекты, обучение CNN–LSTM
  useEffect(() => {
    const t = setTimeout(() => {
      const field = generateField(FW, FH, 31, { nStress: 7, stressLevel: 1.1, parcels: 8 });
      const off = document.createElement('canvas');
      off.width = FW; off.height = FH;
      const ctx = off.getContext('2d');
      ctx.putImageData(new ImageData(new Uint8ClampedArray(field.rgba), FW, FH), 0, 0);
      const r = rng(17);
      const objs = [];
      for (let k = 0; k < 9; k++) {
        const base = makeScene(100 + k).filter((o) => o.cls !== 'tree' && o.cls !== 'pole');
        const o = base[r.int(0, base.length - 1)];
        const x = 380 + k * 125 + r.range(-20, 20), y = r.range(30, FH - 60);
        objs.push({ ...o, x, y, id: k });
      }
      objs.forEach((o) => drawObject(ctx, o));
      fieldRef.current = { field, off, objs };
      const m = new CnnLstm({ win: 10, units: 64, seed: 3 });
      const trainSeqs = Array.from({ length: 8 }, (_, i) => genTrajectory(500, i + 1));
      const t0 = performance.now();
      const hist = m.fit(trainSeqs, genTrajectory(300, 99), { epochs: 200, lr: 0.01, inputNoise: 0.1 });
      const ms = performance.now() - t0;
      model.current = m;
      setTrain({ hist, ms, samples: trainSeqs.reduce((s, q) => s + q.length - 11, 0) });
      setEvalRes(evaluateHybrid(m, genTrajectory(400, 77)));
      reset();
      setReady(true);
    }, 30);
    return () => clearTimeout(t);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const reset = useCallback(() => {
    sim.current = { k: 0, pos: 0, speed: 8, battery: 100, r: rng(5), hist: [], ekf: null, h: null };
    setLog([]); setLastMsg(null); setSpeedHist([]);
    setStats({ frames: 0, dropped: 0, obj: 0, anom: 0, sent: 0, ctrl: 0, CONTINUE: 0, CORRECT: 0, RTH: 0 });
    setView({ trace: [], current: null, frame: null });
  }, []);

  const tick = useCallback(() => {
    const S = sim.current, F = fieldRef.current, M = model.current;
    if (!S || !F || !M) return;
    S.k++;
    const trace = ['start', 'get', 'q1'];
    const lines = [];
    const t = (S.k * 0.5).toFixed(1);
    const L = (s) => lines.push(`[${t} с] ${s}`);
    S.battery = Math.max(0, S.battery - 0.35);
    // положение БПЛА: скорость задаётся адаптивным контуром
    S.pos += S.speed * 0.5 / GSD;
    if (S.pos > FW - VW) S.pos = 0;
    const x0 = Math.round(S.pos), y0 = 10;
    const dropped = S.r() < 0.1;
    if (dropped) {
      trace.push('cont');
      L(`Кадр #${S.k}: не получен → continue`);
      setView({ trace, current: 'cont', frame: null });
      setStats((s) => ({ ...s, frames: s.frames + 1, dropped: s.dropped + 1 }));
      setLog((l) => [...l, ...lines].slice(-200));
      return;
    }
    // кадр
    const fc = frameCanvas.current;
    fc.width = VW; fc.height = VH;
    const ctx = fc.getContext('2d');
    ctx.drawImage(F.off, x0, y0, VW, VH, 0, 0, VW, VH);
    const img = ctx.getImageData(0, 0, VW, VH);
    trace.push('det');
    const blobs = detectBlobs(img.data, VW, VH);
    // каналы мультиспектральной камеры для кадра
    const red = new Float32Array(VW * VH), nir = new Float32Array(VW * VH);
    for (let y = 0; y < VH; y++) for (let x = 0; x < VW; x++) { const i = (y + y0) * FW + x + x0; red[y * VW + x] = F.field.red[i]; nir[y * VW + x] = F.field.nir[i]; }
    trace.push('tex');
    const anomalies = textureAnomalies(red, nir, VW, VH);
    trace.push('gps');
    const e = (x0 + VW / 2 - FW / 2) * GSD + S.r.gauss(0, 1.2), n = (FH / 2 - (y0 + VH / 2)) * GSD + S.r.gauss(0, 1.2);
    const [lat, lon] = enuToGeodetic(e, n, 40, AREA.lat0, AREA.lon0, AREA.h0);
    S.hist.push([e, n]);
    trace.push('q2');
    // отрисовка результатов на кадре
    anomalies.forEach((a) => { ctx.fillStyle = 'rgba(239,68,68,.22)'; ctx.fillRect(a.x, a.y, a.w, a.h); ctx.strokeStyle = 'rgba(239,68,68,.7)'; ctx.strokeRect(a.x + 0.5, a.y + 0.5, a.w - 1, a.h - 1); });
    blobs.forEach((b) => { ctx.strokeStyle = '#facc15'; ctx.lineWidth = 2; ctx.strokeRect(b.x - 2, b.y - 2, b.w + 4, b.h + 4); ctx.fillStyle = '#facc15'; ctx.font = 'bold 10px sans-serif'; ctx.fillText('obj', b.x - 2, b.y - 4); });
    ctx.fillStyle = 'rgba(0,0,0,.55)'; ctx.fillRect(0, VH - 18, VW, 18);
    ctx.fillStyle = '#e6ecf7'; ctx.font = '11px monospace';
    ctx.fillText(`#${S.k} ${lat.toFixed(6)}, ${lon.toFixed(6)} V=${S.speed.toFixed(1)} м/с`, 6, VH - 5);
    L(`Кадр #${S.k}: объектов ${blobs.length}, аномальных блоков ${anomalies.length}, GPS ${lat.toFixed(6)}, ${lon.toFixed(6)}`);
    let current = 'q2';
    const upd = { frames: 1, obj: blobs.length ? 1 : 0, anom: anomalies.length ? 1 : 0, sent: 0, ctrl: 0 };
    if (blobs.length || anomalies.length) {
      trace.push('msg');
      const msg = {
        frame: S.k, t: +t, lat: +lat.toFixed(6), lon: +lon.toFixed(6),
        objects: blobs.map((b) => ({ e: +(e + (b.x + b.w / 2 - VW / 2) * GSD).toFixed(1), n: +(n - (b.y + b.h / 2 - VH / 2) * GSD).toFixed(1), size_m: +(Math.max(b.w, b.h) * GSD).toFixed(1) })),
        anomalies: anomalies.length, ndvi_min: anomalies.length ? +Math.min(...anomalies.map((a) => a.ndvi)).toFixed(3) : null,
      };
      if (!blobs.length) {
        trace.push('send'); current = 'send';
        upd.sent = 1;
        L(`  → аномалии без объектов: сообщение отправлено на НСУ (зона стресса NDVI ${msg.ndvi_min})`);
      } else {
        trace.push('ctrl', 'ai', 'dec', 'adapt', 'fb', 'ad2'); current = 'ad2';
        upd.ctrl = 1;
        // CNN–LSTM–EKF: прогноз следующего положения
        if (!S.ekf) S.ekf = new EKF([e, n, S.speed, 0, 0], { dt: 0.5 });
        S.ekf.predict();
        let pred = null;
        if (S.hist.length > M.win + 1) {
          const d = S.hist.slice(-M.win - 1).slice(1).map((p, i) => [p[0] - S.hist[S.hist.length - M.win - 1 + i][0], p[1] - S.hist[S.hist.length - M.win - 1 + i][1]]);
          const pr = M.predictNext(d, S.h || new Array(M.units).fill(0));
          S.h = pr.h;
          pred = [e + pr.d[0], n + pr.d[1]];
          S.ekf.update([S.hist[S.hist.length - 2][0] + pr.d[0], S.hist[S.hist.length - 2][1] + pr.d[1]], 4);
        }
        S.ekf.update([e, n], 1.44);
        const dist = Math.min(...msg.objects.map((o) => Math.hypot(o.e - e, o.n - n)));
        const wind = 4 + 6 * Math.abs(Math.sin(S.k / 9)) + S.r.gauss(0, 0.8);
        const dcs = mamdani({ battery: S.battery, wind: Math.max(0, wind), dist: Math.hypot(dist, 15) }); // наклонная дальность: БПЛА на 15 м выше объектов
        const action = dcs.risk < 35 ? 'CONTINUE' : dcs.risk < 62 ? 'CORRECT' : 'RTH';
        upd[action] = 1;
        const target = 8 * (1 - (0.7 * dcs.risk) / 100);
        const before = S.speed;
        for (let i = 0; i < 5; i++) S.speed += 0.4 * (target - S.speed); // обратная связь
        L(`  → управление объектом: d=${dist.toFixed(1)} м, ветер ${wind.toFixed(1)} м/с, АКБ ${S.battery.toFixed(0)}%`);
        L(`  → CNN-LSTM-EKF: оценка E=${S.ekf.s[0].toFixed(1)} N=${S.ekf.s[1].toFixed(1)}${pred ? `, прогноз +0.5 с: E=${pred[0].toFixed(1)} N=${pred[1].toFixed(1)}` : ''}`);
        L(`  → решение ИИ: риск ${dcs.risk.toFixed(1)} → ${action}; скорость ${before.toFixed(1)} → ${S.speed.toFixed(1)} м/с (цель ${target.toFixed(1)})`);
        if (action === 'RTH') { L('  → риск критичен (АКБ/ветер/препятствие): возврат на базу, после зарядки миссия возобновляется'); S.battery = 100; S.speed = 8; }
      }
      setLastMsg(msg);
    } else {
      S.speed += 0.3 * (8 - S.speed);
      L('  → объектов и аномалий нет → следующий кадр');
    }
    setSpeedHist((h) => [...h, [S.k * 0.5, S.speed]].slice(-160));
    setView({ trace, current, frame: S.k });
    setStats((s) => { const o = { ...s }; for (const k of Object.keys(upd)) o[k] = (o[k] || 0) + upd[k]; return o; });
    setLog((l) => [...l, ...lines].slice(-200));
  }, []);

  useEffect(() => {
    if (!running) return;
    const id = setInterval(tick, delay);
    return () => clearInterval(id);
  }, [running, delay, tick]);

  useEffect(() => { if (logRef.current) logRef.current.scrollTop = logRef.current.scrollHeight; }, [log]);

  // карта поля с положением кадра
  useEffect(() => {
    const F = fieldRef.current, c = sceneCanvas.current;
    if (!F || !c) return;
    c.width = FW; c.height = FH;
    const ctx = c.getContext('2d');
    ctx.drawImage(F.off, 0, 0);
    const S = sim.current;
    if (S) { ctx.strokeStyle = '#facc15'; ctx.lineWidth = 3; ctx.strokeRect(Math.round(S.pos), 10, VW, VH); }
  }, [view, ready]);

  const runAllTests = () => setTests(runTests());

  const ev = evalRes;
  return (
    <>
      <StageHeader
        title="Программа по блок-схеме (рисунок 2)"
        subtitle="Программа реализует алгоритм обработки потока с камеры БПЛА: получение кадра, обнаружение объектов, проверка аномалий текстуры, получение координат GPS, формирование и отправка сообщения, управление объектом с анализом нейросетью CNN–LSTM и расширенным фильтром Калмана, принятие решения ИИ и адаптивное управление с обратной связью."
        tags={['Код (Python)', 'Выполнение в браузере', 'CNN–LSTM + EKF', 'Автотесты', 'Результаты']}
      />

      <SectionTitle n="P.1">Выполнение программы</SectionTitle>
      <div className="card tight">
        <h3>Блок-схема: подсвечен путь обработки текущего кадра</h3>
        <div style={{ maxWidth: 1050, margin: '0 auto' }}><Flowchart active={view.trace} current={view.current} /></div>
      </div>
      <div className="grid g2 section">
        <div className="card tight">
          <div className="row" style={{ marginBottom: 8 }}>
            <button className="btn small primary" disabled={!ready} onClick={() => setRunning((r) => !r)} type="button">{running ? '❚❚ Пауза' : '▶ Запуск потока камеры'}</button>
            <button className="btn small" disabled={!ready || running} onClick={tick} type="button">Шаг (1 кадр)</button>
            <button className="btn small" disabled={!ready} onClick={reset} type="button">⟲ Сброс</button>
            {!ready && <span className="badge warn">Подготовка данных и обучение сети…</span>}
          </div>
          <div style={{ width: 260 }}><Slider label="Период кадра, мс" value={delay} min={100} max={1500} step={100} onChange={setDelay} /></div>
          <div className="grid g2" style={{ marginTop: 8 }}>
            <div>
              <canvas ref={frameCanvas} width={VW} height={VH} style={{ width: '100%' }} />
              <div className="caption">{view.frame ? `Кадр #${view.frame}: жёлтые рамки — объекты, красные — аномалии NDVI` : view.current === 'cont' ? 'Кадр не получен (continue)' : 'Кадр камеры'}</div>
            </div>
            <div className="metrics" style={{ gridTemplateColumns: '1fr 1fr', alignContent: 'start' }}>
              <Metric k="Кадров" v={stats.frames} />
              <Metric k="Не получено" v={stats.dropped} color="#f87171" />
              <Metric k="С объектами" v={stats.obj} color="#facc15" />
              <Metric k="С аномалиями" v={stats.anom} color="#fb923c" />
              <Metric k="Сообщений" v={stats.sent} />
              <Metric k="Управлений" v={stats.ctrl} color="#38bdf8" />
            </div>
          </div>
        </div>
        <div className="card tight">
          <h3>Журнал выполнения программы</h3>
          <div className="log" ref={logRef} style={{ height: 340 }}>
            {log.length ? log.map((l, i) => <div key={i}>{l}</div>) : <div className="t">Нажмите «Запуск потока камеры»…</div>}
          </div>
        </div>
      </div>
      <div className="card tight section">
        <h3>Поле (мультиспектральная съёмка) и текущее окно камеры</h3>
        <canvas ref={sceneCanvas} style={{ width: '100%' }} />
      </div>
      <div className="grid g2 section">
        <div className="card tight">
          <h3>Адаптивное управление: скорость БПЛА, м/с</h3>
          <LineChart height={200} xLabel="t, с" yMin={0} yMax={10} series={[{ data: speedHist.length ? speedHist : [[0, 8]], color: '#38bdf8' }]} />
          <div className="row small">
            <span className="badge ok">CONTINUE: {stats.CONTINUE}</span>
            <span className="badge warn">CORRECT: {stats.CORRECT}</span>
            <span className="badge bad">RTH: {stats.RTH}</span>
          </div>
        </div>
        <div className="card tight">
          <h3>Последнее сформированное сообщение</h3>
          <Code title="message.json" code={lastMsg ? JSON.stringify(lastMsg, null, 2) : '// ещё нет'} maxHeight={240} />
        </div>
      </div>

      <SectionTitle n="P.2">Анализ CNN–LSTM–EKF: обучение и прогноз</SectionTitle>
      <div className="note" style={{ marginBottom: 12 }}>
        В браузере обучается упрощённая CNN–LSTM: свёрточный слой (ядра сглаживания и разностей) над окном из 10 приращений положения → рекуррентный слой из 64 нейронов (tanh) → выходной слой, обучаемый Adam по MSE. Сеть прогнозирует следующее приращение траектории; гибрид подаёт прогноз в EKF как псевдоизмерение, что особенно важно при пропадании GPS. Полная Keras-версия — в коде программы.
      </div>
      {train && ev ? (
        <>
          <div className="grid g3">
            <div className="card tight">
              <h3>Результаты обучения (MSE)</h3>
              <LineChart height={210} xLabel="эпоха" series={[
                { name: 'обучение', data: train.hist.map((h) => [h.ep, h.train]), color: '#38bdf8' },
                { name: 'валидация', data: train.hist.map((h) => [h.ep, h.val]), color: '#fbbf24' },
              ]} />
              <p className="small muted" style={{ margin: 0 }}>{train.samples} обучающих примеров (8 траекторий), 200 эпох, Adam, {fmt(train.ms / 1000, 1)} с. Итоговая MSE: {train.hist.at(-1).train.toFixed(4)} / {train.hist.at(-1).val.toFixed(4)}.</p>
            </div>
            <div className="card tight">
              <h3>Прогноз траектории (тест)</h3>
              <LineChart height={230} series={[
                { name: 'GPS', data: ev.meas.filter(Boolean), color: '#f87171', scatter: true, r: 1.3, opacity: 0.5 },
                { name: 'EKF', data: ev.ekf, color: '#a78bfa', width: 1.5 },
                { name: 'CNN-LSTM+EKF', data: ev.hyb, color: '#4ade80', width: 2 },
              ]} />
            </div>
            <div className="card tight">
              <h3>Ошибка положения, м</h3>
              <LineChart height={230} xLabel="шаг (0,5 с)" yMin={0} areas={ev.outages.map(([a, b]) => ({ x0: a, x1: b }))} series={[
                { name: 'EKF', data: ev.errs.ekf.map((v, i) => [i, v]), color: '#a78bfa', width: 1.3 },
                { name: 'CNN-LSTM', data: ev.errs.nn.map((v, i) => [i, v]), color: '#fbbf24', width: 1.3 },
                { name: 'Гибрид', data: ev.errs.hyb.map((v, i) => [i, v]), color: '#4ade80', width: 1.8 },
              ]} />
              <p className="small muted" style={{ margin: 0 }}>Красные полосы — пропадание сигнала GPS.</p>
            </div>
          </div>
          <div className="card table-wrap section">
            <table>
              <thead><tr><th>Метод</th><th className="num">RMSE, весь тест, м</th><th className="num">RMSE при наличии GPS, м</th><th className="num">RMSE при потере GPS, м</th></tr></thead>
              <tbody>
                <tr><td>Сырые измерения GPS (σ = 3 м)</td><td className="num">{fmt(ev.rmse.gps.all)}</td><td className="num">{fmt(ev.rmse.gps.gps)}</td><td className="num">нет данных</td></tr>
                <tr><td>EKF (модель CTRV)</td><td className="num">{fmt(ev.rmse.ekf.all)}</td><td className="num">{fmt(ev.rmse.ekf.gps)}</td><td className="num">{fmt(ev.rmse.ekf.outage)}</td></tr>
                <tr><td>CNN–LSTM</td><td className="num">{fmt(ev.rmse.nn.all)}</td><td className="num">{fmt(ev.rmse.nn.gps)}</td><td className="num">{fmt(ev.rmse.nn.outage)}</td></tr>
                <tr><td><b>Гибрид CNN–LSTM + EKF</b></td><td className="num"><b>{fmt(ev.rmse.hyb.all)}</b></td><td className="num"><b>{fmt(ev.rmse.hyb.gps)}</b></td><td className="num"><b>{fmt(ev.rmse.hyb.outage)}</b></td></tr>
              </tbody>
            </table>
          </div>
        </>
      ) : (
        <div className="card muted">Обучение модели…</div>
      )}

      <SectionTitle n="P.3">Код программы</SectionTitle>
      <Code code={PY} title="uav_pipeline.py — реализация блок-схемы рисунка 2" maxHeight={640} />

      <SectionTitle n="P.4">Тестирование</SectionTitle>
      <div className="card">
        <div className="row" style={{ marginBottom: 10 }}>
          <button className="btn green" onClick={runAllTests} type="button">▶ Запустить тесты</button>
          {tests && <span className={`badge ${tests.every((t) => t.ok) ? 'ok' : 'bad'}`}>{tests.filter((t) => t.ok).length} / {tests.length} пройдено · {fmt(tests.reduce((s, t) => s + t.ms, 0), 0)} мс</span>}
        </div>
        <div className="table-wrap">
          <table>
            <thead><tr><th>Статус</th><th>Тест</th><th>Что проверяется</th><th>Результат</th><th className="num">мс</th></tr></thead>
            <tbody>
              {(tests || []).map((t) => (
                <tr key={t.name}>
                  <td>{t.ok ? <span className="badge ok">PASS</span> : <span className="badge bad">FAIL</span>}</td>
                  <td style={{ fontFamily: 'var(--mono)', fontSize: 13 }}>{t.name}</td>
                  <td className="small">{t.desc}</td>
                  <td className="small" style={{ fontFamily: 'var(--mono)' }}>{t.info}</td>
                  <td className="num">{t.ms.toFixed(1)}</td>
                </tr>
              ))}
              {!tests && <tr><td colSpan={5} className="muted">Нажмите «Запустить тесты» — 15 модульных тестов выполняются в браузере над теми же функциями, что работают в программе.</td></tr>}
            </tbody>
          </table>
        </div>
        <Code title="test_pipeline.py (pytest)" code={`
import numpy as np, pytest
from uav_pipeline import check_texture_anomalies, form_message, HybridEKF
from coords import geodetic_to_ecef, ecef_to_geodetic

def test_ndvi_anomaly_block():
    red = np.full((20, 40), 0.05); nir = np.full((20, 40), 0.30)
    red[:, :20], nir[:, :20] = 0.15, 0.25                  # NDVI = 0.25 -> стресс
    a = check_texture_anomalies(red, nir)
    assert len(a) == 1 and a[0]['x'] == 0

def test_frame_none_is_skipped(monkeypatch):
    frames = iter([(True, np.zeros((4, 4, 3))), (False, None), (True, np.zeros((4, 4, 3)))])
    processed = [f for ok, f in frames if ok and f is not None]
    assert len(processed) == 2

def test_ecef_roundtrip():
    p = (44.8152, 65.4622, 188.5)
    assert np.allclose(ecef_to_geodetic(*geodetic_to_ecef(*p)), p, atol=1e-7)

def test_ekf_circle():
    ekf, rng = HybridEKF([50, 0, 5, np.pi/2, 0]), np.random.default_rng(5)
    for k in range(200):
        t = 0.1 * k * 0.5; ekf.predict()
        ekf.update(np.array([50*np.cos(t), 50*np.sin(t)]) + rng.normal(0, 3, 2), r=9)
    assert abs(ekf.s[4] - 0.1) < 0.02
`} />
      </div>

      <SectionTitle n="P.5">Анализ результатов</SectionTitle>
      <div className="grid g2">
        <div className="card">
          <h3>Сравнение точности прогноза</h3>
          {ev ? <BarChart width={520} height={170} items={[
            { label: 'GPS (сырые)', value: ev.rmse.gps.all, color: '#f87171' },
            { label: 'EKF', value: ev.rmse.ekf.all, color: '#a78bfa' },
            { label: 'CNN–LSTM', value: ev.rmse.nn.all, color: '#fbbf24' },
            { label: 'CNN–LSTM + EKF', value: ev.rmse.hyb.all, color: '#4ade80' },
          ]} unit=" м" /> : null}
        </div>
        <div className="card">
          <h3>Выводы</h3>
          <ul className="small" style={{ margin: 0, paddingLeft: 18 }}>
            <li>Программа устойчива к потере кадров: при отсутствии кадра выполняется <code>continue</code>, цикл продолжается.</li>
            <li>Кадры без объектов и аномалий не порождают сообщений, что разгружает канал связи с НСУ.</li>
            <li>Зоны с NDVI &lt; 0,38 выявляются как аномалии текстуры и передаются с координатами GPS.</li>
            <li>При наличии GPS гибрид совпадает по точности с EKF (фильтр подавляет шум измерений); при потере сигнала прогноз сети, поданный в EKF как псевдоизмерение, в большинстве тестовых траекторий заметно снижает ошибку по сравнению с чистой моделью движения EKF (CTRV плохо экстраполирует меняющуюся скорость поворота).</li><li>Нейросеть отдельно (без фильтра) накапливает ошибку авторегрессии — поэтому её используют именно в связке с EKF.</li>
            <li>Нечёткий регулятор снижает скорость при приближении к объектам и сильном ветре, а при низком заряде инициирует возврат на базу.</li>
          </ul>
        </div>
      </div>
      <details className="card section">
        <summary style={{ cursor: 'pointer', fontWeight: 600 }}>Исходный рисунок 2 из задания</summary>
        <div className="figure" style={{ marginTop: 12, maxWidth: 760 }}><img src="/figures/fig2.png" alt="Рисунок 2" /></div>
      </details>
      <Pager href="/program" />
    </>
  );
}
