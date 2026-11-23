'use client';
import { useEffect, useMemo, useRef, useState } from 'react';
import { StageHeader, Tasks, SectionTitle, Pager, Slider, Seg, Metric, Code } from '@/components/UI';
import { LineChart } from '@/components/Charts';
import { rng, fmt, mean } from '@/lib/math';

const DT = 0.05, TMAX = 40, M = 1.5, DRAG = 0.35, FMAX = 16;
const OBS = { step: { x: 12, y: 7.3, r: 3 }, circle: { x: 16, y: 15, r: 3 }, eight: { x: 14.1, y: 11, r: 3 } };

function reference(kind, t) {
  if (kind === 'step') return t < 1 ? [0, 0] : [25, 12];
  if (kind === 'circle') { const w = 0.3; return [15 * Math.sin(w * t), 15 - 15 * Math.cos(w * t)]; }
  const w = 0.25; return [20 * Math.sin(w * t), 10 * Math.sin(2 * w * t)];
}

export function simulate({ kp, ki, kd, wind, kind, avoid, seed = 3 }) {
  const r = rng(seed);
  let x = 0, y = 0, vx = 0, vy = 0, ix = 0, iy = 0;
  let pex = 0, pey = 0;
  const out = [];
  let gx = 0, gy = 0;
  const OB = OBS[kind];
  for (let k = 0; k <= TMAX / DT; k++) {
    const t = k * DT;
    let [rx, ry] = reference(kind, t);
    // манёвр облёта: смещение опорной точки полем отталкивания препятствия
    if (avoid) {
      const dx = rx - OB.x, dy = ry - OB.y, d = Math.hypot(dx, dy);
      const infl = OB.r + 4;
      if (d < infl) { const push = (infl - d) * 1.1; rx += (dx / (d || 1)) * push; ry += (dy / (d || 1)) * push; }
    }
    const ex = rx - x, ey = ry - y;
    ix = Math.max(-20, Math.min(20, ix + ex * DT)); // анти-насыщение интегратора
    iy = Math.max(-20, Math.min(20, iy + ey * DT));
    const dex = k ? (ex - pex) / DT : 0, dey = k ? (ey - pey) / DT : 0;
    pex = ex; pey = ey;
    let ux = kp * ex + ki * ix + kd * dex, uy = kp * ey + ki * iy + kd * dey;
    const un = Math.hypot(ux, uy);
    if (un > FMAX) { ux *= FMAX / un; uy *= FMAX / un; }
    // ветер: постоянная составляющая + порывы (марковский процесс)
    gx = 0.97 * gx + r.gauss(0, 0.25 * wind); gy = 0.97 * gy + r.gauss(0, 0.25 * wind);
    const wx = wind * 0.8 + gx, wy = -wind * 0.4 + gy;
    const ax = (ux - DRAG * (vx - wx)) / M, ay = (uy - DRAG * (vy - wy)) / M;
    vx += ax * DT; vy += ay * DT; x += vx * DT; y += vy * DT;
    const dOb = Math.hypot(x - OB.x, y - OB.y);
    out.push({ t, x, y, rx, ry, e: Math.hypot(ex, ey), v: Math.hypot(vx, vy), hd: ((Math.atan2(vx, vy) * 180) / Math.PI + 360) % 360, u: Math.hypot(ux, uy), dOb });
  }
  const errs = out.filter((s) => s.t > 2).map((s) => s.e);
  const res = { out, rmse: Math.sqrt(mean(errs.map((e) => e * e))), maxE: Math.max(...errs), energy: out.reduce((s, p) => s + p.u * p.u * DT, 0), minOb: Math.min(...out.map((p) => p.dOb)) };
  if (kind === 'step') {
    const D = Math.hypot(25, 12);
    const proj = out.map((s) => (s.x * 25 + s.y * 12) / D);
    res.overshoot = Math.max(0, (Math.max(...proj) - D) / D) * 100;
    let ts = 0;
    for (let i = out.length - 1; i >= 0; i--) if (out[i].e > 0.02 * D) { ts = out[i].t; break; }
    res.settle = ts - 1;
  }
  return res;
}

export default function Stage3() {
  const [kp, setKp] = useState(6);
  const [ki, setKi] = useState(1.2);
  const [kd, setKd] = useState(4.5);
  const [wind, setWind] = useState(3);
  const [kind, setKind] = useState('eight');
  const [avoid, setAvoid] = useState(false);
  const [frame, setFrame] = useState(0);
  const [play, setPlay] = useState(true);
  const canvas = useRef(null);

  const sim = useMemo(() => simulate({ kp, ki, kd, wind, kind, avoid }), [kp, ki, kd, wind, kind, avoid]);
  const cmp = useMemo(
    () => [
      ['P', kp, 0, 0], ['PI', kp, ki, 0], ['PD', kp, 0, kd], ['PID', kp, ki, kd],
    ].map(([n, a, b, c]) => ({ n, ...simulate({ kp: a, ki: b, kd: c, wind, kind, avoid }) })),
    [kp, ki, kd, wind, kind, avoid]
  );

  useEffect(() => {
    if (!play) return;
    let id, last = 0;
    const loop = (ts) => {
      if (ts - last > 30) { setFrame((f) => (f + 2) % sim.out.length); last = ts; }
      id = requestAnimationFrame(loop);
    };
    id = requestAnimationFrame(loop);
    return () => cancelAnimationFrame(id);
  }, [play, sim]);

  useEffect(() => {
    const c = canvas.current;
    if (!c) return;
    const Wc = 560, Hc = 380;
    c.width = Wc; c.height = Hc;
    const ctx = c.getContext('2d');
    const xs = sim.out.flatMap((s) => [s.x, s.rx]), ys = sim.out.flatMap((s) => [s.y, s.ry]);
    const x0 = Math.min(...xs) - 5, x1 = Math.max(...xs) + 5, y0 = Math.min(...ys) - 5, y1 = Math.max(...ys) + 5;
    const sc = Math.min(Wc / (x1 - x0), Hc / (y1 - y0));
    const ox = (Wc - (x1 - x0) * sc) / 2, oy = (Hc - (y1 - y0) * sc) / 2;
    const P = (x, y) => [ox + (x - x0) * sc, Hc - (oy + (y - y0) * sc)];
    ctx.fillStyle = '#07101f'; ctx.fillRect(0, 0, Wc, Hc);
    ctx.strokeStyle = '#16223b';
    for (let gx = Math.ceil(x0 / 5) * 5; gx < x1; gx += 5) { const [a] = P(gx, 0); ctx.beginPath(); ctx.moveTo(a, 0); ctx.lineTo(a, Hc); ctx.stroke(); }
    for (let gy = Math.ceil(y0 / 5) * 5; gy < y1; gy += 5) { const [, b] = P(0, gy); ctx.beginPath(); ctx.moveTo(0, b); ctx.lineTo(Wc, b); ctx.stroke(); }
    const OB = OBS[kind];
    if (avoid) {
      const [a, b] = P(OB.x, OB.y);
      ctx.fillStyle = 'rgba(248,113,113,.35)'; ctx.strokeStyle = '#f87171';
      ctx.beginPath(); ctx.arc(a, b, OB.r * sc, 0, 7); ctx.fill(); ctx.stroke();
      ctx.fillStyle = '#fecaca'; ctx.font = '11px sans-serif'; ctx.fillText('препятствие', a - 30, b - OB.r * sc - 6);
    }
    const line = (key, color, w, dash) => {
      ctx.strokeStyle = color; ctx.lineWidth = w; ctx.setLineDash(dash || []); ctx.beginPath();
      sim.out.forEach((s, i) => { const [a, b] = P(key ? s.rx : s.x, key ? s.ry : s.y); i ? ctx.lineTo(a, b) : ctx.moveTo(a, b); });
      ctx.stroke(); ctx.setLineDash([]);
    };
    line(true, '#94a3b8', 1.5, [6, 5]);
    ctx.strokeStyle = '#22d3ee'; ctx.lineWidth = 2.2; ctx.beginPath();
    sim.out.slice(0, frame + 1).forEach((s, i) => { const [a, b] = P(s.x, s.y); i ? ctx.lineTo(a, b) : ctx.moveTo(a, b); });
    ctx.stroke();
    const s = sim.out[Math.min(frame, sim.out.length - 1)];
    const [ra, rb] = P(s.rx, s.ry);
    ctx.fillStyle = '#fbbf24'; ctx.beginPath(); ctx.arc(ra, rb, 5, 0, 7); ctx.fill();
    const [a, b] = P(s.x, s.y);
    // дрон-квадрокоптер
    ctx.save(); ctx.translate(a, b);
    ctx.strokeStyle = '#e6ecf7'; ctx.lineWidth = 2;
    ctx.beginPath(); ctx.moveTo(-9, -9); ctx.lineTo(9, 9); ctx.moveTo(9, -9); ctx.lineTo(-9, 9); ctx.stroke();
    ctx.fillStyle = '#38bdf8';
    for (const [dx, dy] of [[-9, -9], [9, 9], [9, -9], [-9, 9]]) { ctx.beginPath(); ctx.arc(dx, dy, 4.5, 0, 7); ctx.fill(); }
    ctx.restore();
    // ветер
    ctx.strokeStyle = '#a78bfa'; ctx.fillStyle = '#a78bfa'; ctx.lineWidth = 2;
    const wl = 10 + wind * 6;
    ctx.beginPath(); ctx.moveTo(20, 24); ctx.lineTo(20 + wl, 24 + wl * 0.5 * 0.5); ctx.stroke();
    ctx.font = '12px sans-serif'; ctx.fillText(`ветер ${wind} м/с`, 20, 16);
    ctx.fillStyle = '#e6ecf7'; ctx.fillText(`t = ${s.t.toFixed(1)} с   ошибка = ${s.e.toFixed(2)} м   V = ${s.v.toFixed(1)} м/с   курс = ${s.hd.toFixed(0)}°`, 20, Hc - 12);
  }, [sim, frame, avoid, wind, kind]);

  const ser = (k) => sim.out.filter((_, i) => i % 2 === 0).map((s) => [s.t, s[k]]);
  return (
    <>
      <StageHeader
        num={3}
        title="Управление полётом"
        subtitle="Управление движением дрона в реальном времени: ПИД-регулятор отслеживает заданную траекторию при ветровых возмущениях, корректирует скорость и курс, выполняет манёвр облёта препятствия. Альтернатива — обучение с подкреплением (PPO)."
        tags={['PID', 'Анти-насыщение интегратора', 'Потенциальное поле (облёт)', 'PPO (Reinforcement Learning)']}
      />
      <div className="grid g2">
        <div className="card"><h3>Выполняемые задачи</h3><Tasks items={['Применение алгоритмов PID или PPO', 'Коррекция скорости и направления', 'Маневрирование (облёт препятствий)']} /></div>
        <div className="card">
          <h3>Закон управления</h3>
          <p style={{ fontFamily: 'var(--mono)', fontSize: 14 }}>u(t) = K<sub>p</sub>·e(t) + K<sub>i</sub>·∫e(τ)dτ + K<sub>d</sub>·de/dt</p>
          <p className="small muted" style={{ margin: 0 }}>Модель: материальная точка m = 1,5 кг, аэродинамическое сопротивление, ограничение тяги 16 Н, ветер = постоянная составляющая + порывы (процесс Гаусса–Маркова). Шаг интегрирования 50 мс (20 Гц — частота контура управления).</p>
        </div>
      </div>

      <SectionTitle n="3.1">Симуляция в реальном времени</SectionTitle>
      <div className="grid g-side">
        <div className="card tight">
          <div className="row" style={{ marginBottom: 10 }}>
            <Seg value={kind} onChange={(v) => { setKind(v); setFrame(0); }} options={[['step', 'Переход в точку'], ['circle', 'Круг'], ['eight', 'Восьмёрка']]} />
            <span className="spacer" />
            <button className="btn small" onClick={() => setPlay((p) => !p)} type="button">{play ? '❚❚ Пауза' : '▶ Пуск'}</button>
            <button className="btn small" onClick={() => setFrame(0)} type="button">⟲</button>
          </div>
          <canvas ref={canvas} style={{ width: '100%' }} />
          <div className="legend"><span><i style={{ background: '#94a3b8' }} />Заданная траектория</span><span><i style={{ background: '#22d3ee' }} />Фактическая</span><span><i style={{ background: '#fbbf24' }} />Опорная точка</span></div>
        </div>
        <div className="card tight controls">
          <Slider label="Kp" value={kp} min={0} max={15} step={0.1} onChange={setKp} digits={1} />
          <Slider label="Ki" value={ki} min={0} max={5} step={0.1} onChange={setKi} digits={1} />
          <Slider label="Kd" value={kd} min={0} max={10} step={0.1} onChange={setKd} digits={1} />
          <Slider label="Сила ветра, м/с" value={wind} min={0} max={10} step={0.5} onChange={setWind} digits={1} />
          <label className="row small"><input type="checkbox" checked={avoid} onChange={(e) => setAvoid(e.target.checked)} /> Препятствие на маршруте (манёвр облёта)</label>
          <div className="metrics">
            <Metric k="RMSE слежения, м" v={fmt(sim.rmse)} color="#22d3ee" />
            <Metric k="Макс. ошибка, м" v={fmt(sim.maxE)} />
            {sim.overshoot !== undefined && <Metric k="Перерегулирование" v={fmt(sim.overshoot, 1) + '%'} />}
            {sim.settle !== undefined && <Metric k="Время установления, с" v={fmt(sim.settle, 1)} />}
            {avoid && <Metric k="Мин. дистанция до препятствия, м" v={fmt(sim.minOb - OBS[kind].r, 1)} color={sim.minOb > OBS[kind].r ? '#4ade80' : '#f87171'} />}
          </div>
        </div>
      </div>

      <div className="grid g3 section">
        <div className="card tight"><h3>Ошибка слежения, м</h3><LineChart height={200} xLabel="t, с" series={[{ data: ser('e'), color: '#f87171' }]} /></div>
        <div className="card tight"><h3>Коррекция скорости, м/с</h3><LineChart height={200} xLabel="t, с" series={[{ data: ser('v'), color: '#4ade80' }]} /></div>
        <div className="card tight"><h3>Коррекция курса, °</h3><LineChart height={200} xLabel="t, с" yMin={0} yMax={360} series={[{ data: ser('hd'), color: '#a78bfa', scatter: true, r: 1.4 }]} /></div>
      </div>

      <SectionTitle n="3.2">Сравнение регуляторов при текущих условиях</SectionTitle>
      <div className="card table-wrap">
        <table>
          <thead><tr><th>Регулятор</th><th className="num">RMSE, м</th><th className="num">Макс. ошибка, м</th>{kind === 'step' && <><th className="num">Перерегулирование, %</th><th className="num">Установление, с</th></>}<th className="num">Энергия управления ∫u²dt</th></tr></thead>
          <tbody>
            {cmp.map((c) => {
              const best = Math.min(...cmp.map((x) => +x.rmse.toFixed(2)));
              return (
                <tr key={c.n}>
                  <td><b>{c.n}</b> {+c.rmse.toFixed(2) === best && <span className="badge ok">лучший</span>}</td>
                  <td className="num">{fmt(c.rmse)}</td>
                  <td className="num">{fmt(c.maxE)}</td>
                  {kind === 'step' && <><td className="num">{fmt(c.overshoot, 1)}</td><td className="num">{fmt(c.settle, 1)}</td></>}
                  <td className="num">{fmt(c.energy, 0)}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>

      <SectionTitle n="3.3">PPO — обучение с подкреплением</SectionTitle>
      <div className="grid g2">
        <div className="card">
          <p className="small">Proximal Policy Optimization обучает нейросетевую политику π(a|s), которая по состоянию (ошибка положения, скорость, ветер) выдаёт тягу. Целевая функция с отсечением:</p>
          <p style={{ fontFamily: 'var(--mono)', fontSize: 13 }}>L = E[min(r·A, clip(r, 1−ε, 1+ε)·A)], r = π<sub>θ</sub>/π<sub>θold</sub></p>
          <p className="small">Награда: −‖e‖² − 0,01‖u‖² − 100·[столкновение]. В отличие от ПИД, политика PPO учитывает нелинейности и ограничения, но требует миллионов шагов симуляции и не даёт формальных гарантий устойчивости, поэтому в системе ПИД используется как базовый контур, а PPO — как адаптивная надстройка (этап 7, программа).</p>
        </div>
        <Code title="ppo_train.py" code={`
import gymnasium as gym
from stable_baselines3 import PPO

class DroneEnv(gym.Env):
    # obs: [ex, ey, vx, vy, wind_x, wind_y]; action: [Fx, Fy] in [-1, 1]
    ...
    def step(self, a):
        self.state = self.dynamics(self.state, a * F_MAX, self.wind())
        e = self.ref(self.t) - self.state[:2]
        reward = -(e @ e) - 0.01 * (a @ a) - 100 * self.collision()
        return self.obs(), reward, self.done(), False, {}

model = PPO('MlpPolicy', DroneEnv(), n_steps=2048, batch_size=64,
            learning_rate=3e-4, clip_range=0.2, gamma=0.99, verbose=1)
model.learn(total_timesteps=2_000_000)
model.save('ppo_drone')
`} />
      </div>

      <div className="note section">
        <b>Вывод.</b> Пропорциональная составляющая задаёт быстродействие, дифференциальная гасит колебания и перерегулирование, интегральная устраняет статическую ошибку от постоянного ветра. ПИД (и близкий к нему ПД) даёт наименьшую ошибку слежения, а без дифференциальной составляющей (P, PI) возникают колебания; при включении препятствия траектория деформируется полем отталкивания и дрон проходит препятствие с безопасным зазором.
      </div>
      <Pager href="/etap-3" />
    </>
  );
}
