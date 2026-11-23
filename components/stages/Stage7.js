'use client';
import { useMemo, useState } from 'react';
import Link from 'next/link';
import { StageHeader, Tasks, SectionTitle, Pager, Slider, Metric, Code, Tabs } from '@/components/UI';
import { LineChart, BarChart } from '@/components/Charts';
import { rng, fmt, polyfit, polyval, clamp } from '@/lib/math';
import { buildTree, treeProba, trainForest, forestProba, argmax, confusion, trainSVM, trap } from '@/lib/ml';

// ================= Нечёткая система Мамдани =================
const MF = {
  battery: { range: [0, 100], unit: '%', label: 'Заряд батареи', sets: { 'низкий': [0, 0, 20, 40], 'средний': [25, 45, 55, 75], 'высокий': [60, 80, 100, 100] } },
  wind: { range: [0, 20], unit: ' м/с', label: 'Скорость ветра', sets: { 'слабый': [0, 0, 3, 7], 'умеренный': [4, 8, 10, 14], 'сильный': [11, 15, 20, 20] } },
  dist: { range: [0, 50], unit: ' м', label: 'Дистанция до препятствия', sets: { 'близко': [0, 0, 5, 12], 'средне': [8, 15, 22, 30], 'далеко': [25, 35, 50, 50] } },
  risk: { range: [0, 100], unit: '', label: 'Уровень риска', sets: { 'низкий': [0, 0, 20, 40], 'средний': [30, 45, 55, 70], 'высокий': [60, 80, 100, 100] } },
};
const RULES = [
  [{ battery: 'низкий' }, 'высокий'],
  [{ wind: 'сильный' }, 'высокий'],
  [{ dist: 'близко' }, 'высокий'],
  [{ battery: 'средний', wind: 'умеренный' }, 'средний'],
  [{ dist: 'средне' }, 'средний'],
  [{ battery: 'высокий', wind: 'умеренный' }, 'средний'],
  [{ battery: 'высокий', wind: 'слабый', dist: 'далеко' }, 'низкий'],
  [{ battery: 'средний', wind: 'слабый', dist: 'далеко' }, 'низкий'],
  [{ battery: 'высокий', dist: 'далеко' }, 'низкий'],
];
export function mamdani(inp) {
  const mu = (v, s) => trap(inp[v], ...MF[v].sets[s]);
  const fired = RULES.map(([ant, out]) => ({ ant, out, w: Math.min(...Object.entries(ant).map(([v, s]) => mu(v, s))) }));
  const agg = [];
  let num = 0, den = 0;
  for (let x = 0; x <= 100; x += 1) {
    let m = 0;
    for (const f of fired) m = Math.max(m, Math.min(f.w, trap(x, ...MF.risk.sets[f.out])));
    agg.push([x, m]);
    num += x * m; den += m;
  }
  const risk = den ? num / den : 50;
  const decision = risk < 35 ? { t: 'Продолжить миссию', c: '#4ade80' } : risk < 62 ? { t: 'Коррекция: снизить скорость, облёт препятствия', c: '#fbbf24' } : { t: 'Возврат на базу (RTH) / посадка', c: '#f87171' };
  return { fired, agg, risk, decision };
}
function mfSeries(v) {
  const cols = ['#f87171', '#fbbf24', '#4ade80'];
  const [lo, hi] = MF[v].range;
  const names = Object.keys(MF[v].sets);
  const order = v === 'battery' || v === 'dist' ? [0, 1, 2] : [2, 1, 0];
  return names.map((n, i) => ({ name: n, color: cols[order[i]], data: Array.from({ length: 101 }, (_, k) => { const x = lo + ((hi - lo) * k) / 100; return [x, trap(x, ...MF[v].sets[n])]; }) }));
}

function FuzzyBlock() {
  const [inp, setInp] = useState({ battery: 55, wind: 9, dist: 18 });
  const res = useMemo(() => mamdani(inp), [inp]);
  const set = (k) => (v) => setInp((s) => ({ ...s, [k]: v }));
  return (
    <>
      <div className="grid g-side">
        <div className="grid g3">
          {['battery', 'wind', 'dist'].map((v) => (
            <div className="card tight" key={v}>
              <Slider label={MF[v].label} value={inp[v]} min={MF[v].range[0]} max={MF[v].range[1]} step={v === 'wind' ? 0.5 : 1} onChange={set(v)} unit={MF[v].unit} digits={v === 'wind' ? 1 : 0} />
              <LineChart height={150} width={320} yMin={0} yMax={1.05} series={mfSeries(v)} markers={[{ x: inp[v], color: '#e6ecf7' }]} />
            </div>
          ))}
        </div>
        <div className="card tight">
          <h3>Вывод (центроид)</h3>
          <LineChart height={170} width={320} yMin={0} yMax={1.05} series={[{ data: res.agg, color: '#a78bfa', name: 'агрегированное множество' }]} markers={[{ x: res.risk, color: res.decision.c, label: res.risk.toFixed(1) }]} />
          <div className="metric" style={{ marginTop: 8, borderColor: res.decision.c }}>
            <div className="v" style={{ color: res.decision.c, fontSize: 18 }}>{res.decision.t}</div>
            <div className="k">Риск = {res.risk.toFixed(1)} / 100</div>
          </div>
        </div>
      </div>
      <div className="card table-wrap section">
        <table>
          <thead><tr><th>#</th><th>Правило (ЕСЛИ … ТО …)</th><th className="num">Степень срабатывания</th></tr></thead>
          <tbody>
            {res.fired.map((f, i) => (
              <tr key={i} style={{ opacity: f.w > 0 ? 1 : 0.45 }}>
                <td>{i + 1}</td>
                <td>ЕСЛИ {Object.entries(f.ant).map(([v, s]) => `${MF[v].label.toLowerCase()} = «${s}»`).join(' И ')} ТО риск = «{f.out}»</td>
                <td className="num"><span style={{ display: 'inline-block', width: 80, height: 8, background: '#1f2a44', borderRadius: 4, marginRight: 8, verticalAlign: 'middle' }}><span style={{ display: 'block', width: `${f.w * 100}%`, height: 8, background: '#a78bfa', borderRadius: 4 }} /></span>{f.w.toFixed(2)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </>
  );
}

// ================= Деревья решений / Random Forest =================
const FEATS = ['NDVI', 'NDRE', 'Влажность почвы, %', 'ΔT полог−воздух, °C'];
const CLS = ['Норма', 'Умеренный стресс', 'Сильный стресс'];
const CLS_COL = ['#4ade80', '#fbbf24', '#f87171'];
export function cropDataset(n = 600, seed = 5) {
  const r = rng(seed);
  const X = [], y = [];
  for (let i = 0; i < n; i++) {
    const c = r() < 0.45 ? 0 : r() < 0.55 ? 1 : 2;
    const ndvi = clamp([0.68, 0.48, 0.3][c] + r.gauss(0, 0.08), 0.05, 0.9);
    const ndre = clamp(ndvi * 0.55 + r.gauss(0, 0.05), 0.02, 0.6);
    const sm = clamp([32, 22, 14][c] + r.gauss(0, 5), 3, 45);
    const dt = [-1.5, 1, 3.2][c] + r.gauss(0, 1.2);
    X.push([ndvi, ndre, sm, dt]); y.push(c);
  }
  return { X, y };
}

function TreesBlock() {
  const [depth, setDepth] = useState(4);
  const [nTrees, setNTrees] = useState(30);
  const [q, setQ] = useState([0.42, 0.22, 18, 2]);
  const data = useMemo(() => cropDataset(), []);
  const split = Math.floor(data.X.length * 0.7);
  const Xtr = data.X.slice(0, split), ytr = data.y.slice(0, split), Xte = data.X.slice(split), yte = data.y.slice(split);
  const model = useMemo(() => {
    const t0 = performance.now();
    const impT = new Array(4).fill(0);
    const tree = buildTree(Xtr, ytr, [...Xtr.keys()], { maxDepth: depth, K: 3, imp: impT });
    const t1 = performance.now();
    const forest = trainForest(Xtr, ytr, { nTrees, maxDepth: depth + 2, K: 3, seed: 3 });
    const t2 = performance.now();
    const pT = Xte.map((x) => argmax(treeProba(tree, x))), pF = Xte.map((x) => argmax(forestProba(forest, x)));
    const trT = Xtr.map((x) => argmax(treeProba(tree, x)));
    const acc = (p, yy) => p.filter((v, i) => v === yy[i]).length / yy.length;
    return { tree, forest, accT: acc(pT, yte), accF: acc(pF, yte), accTrT: acc(trT, ytr), cm: confusion(yte, pF, 3), msT: t1 - t0, msF: t2 - t1 };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [depth, nTrees, data]);
  const prob = forestProba(model.forest, q);
  const k = argmax(prob);
  const renderTree = (n, d = 0) => {
    if (d > 2) return null;
    if (n.leaf) return <div className="small" style={{ marginLeft: d * 18 }}>→ <b style={{ color: CLS_COL[argmax(n.p)] }}>{CLS[argmax(n.p)]}</b> <span className="muted">({n.n} обр.)</span></div>;
    return (
      <div style={{ marginLeft: d * 18 }}>
        <div className="small" style={{ fontFamily: 'var(--mono)' }}>{FEATS[n.f]} ≤ {n.thr.toFixed(2)} ? <span className="muted">(n={n.n})</span></div>
        {renderTree(n.left, d + 1)}
        {renderTree(n.right, d + 1)}
      </div>
    );
  };
  return (
    <>
      <div className="grid g-side">
        <div className="grid g2">
          <div className="card tight">
            <h3>Важность признаков (Random Forest)</h3>
            <BarChart width={420} height={160} digits={3} items={FEATS.map((f, i) => ({ label: f, value: model.forest.importance[i], color: '#38bdf8' }))} />
            <h3 style={{ marginTop: 10 }}>Матрица ошибок (тест, RF)</h3>
            <table>
              <thead><tr><th>Факт \ Прогноз</th>{CLS.map((c) => <th key={c} className="num">{c}</th>)}</tr></thead>
              <tbody>{model.cm.map((row, i) => <tr key={i}><td style={{ color: CLS_COL[i] }}>{CLS[i]}</td>{row.map((v, j) => <td key={j} className="num" style={{ background: i === j ? '#0f3b2a' : v ? '#3b141455' : undefined }}>{v}</td>)}</tr>)}</tbody>
            </table>
          </div>
          <div className="card tight">
            <h3>Дерево решений (верхние уровни)</h3>
            {renderTree(model.tree)}
          </div>
        </div>
        <div className="card tight controls">
          <Slider label="Глубина дерева" value={depth} min={1} max={8} step={1} onChange={setDepth} />
          <Slider label="Число деревьев в лесу" value={nTrees} min={1} max={60} step={1} onChange={setNTrees} />
          <div className="metrics">
            <Metric k="Точность DT (тест)" v={fmt(model.accT * 100, 1) + '%'} />
            <Metric k="Точность RF (тест)" v={fmt(model.accF * 100, 1) + '%'} color="#4ade80" />
            <Metric k="DT на обучении" v={fmt(model.accTrT * 100, 1) + '%'} />
            <Metric k="Время обучения RF, мс" v={fmt(model.msF, 0)} />
          </div>
          <div className="small muted">Классификация нового участка поля:</div>
          {FEATS.map((f, i) => (
            <Slider key={f} label={f} value={q[i]} min={[0, 0, 0, -5][i]} max={[0.9, 0.6, 45, 7][i]} step={[0.01, 0.01, 1, 0.5][i]} onChange={(v) => setQ((s) => s.map((x, j) => (j === i ? v : x)))} digits={[2, 2, 0, 1][i]} />
          ))}
          <div className="metric" style={{ borderColor: CLS_COL[k] }}>
            <div className="v" style={{ color: CLS_COL[k], fontSize: 18 }}>{CLS[k]}</div>
            <div className="k">P = {prob.map((p) => p.toFixed(2)).join(' / ')}</div>
          </div>
        </div>
      </div>
      <p className="small muted">Набор данных: 600 участков (70% обучение / 30% тест), признаки — спектральные индексы NDVI и NDRE с мультиспектральной камеры, влажность почвы и разность температур полога и воздуха (тепловизор). XGBoost отличается от Random Forest последовательным (бустинговым) построением деревьев на остатках — его код приведён ниже.</p>
    </>
  );
}

// ================= SVM и регрессия =================
function MLBlock() {
  const [deg, setDeg] = useState(2);
  const data = useMemo(() => cropDataset(400, 8), []);
  const svm = useMemo(() => {
    const pts = data.X.map((x, i) => ({ a: x[0], b: x[3], y: data.y[i] === 0 ? -1 : 1 }));
    const ma = pts.reduce((s, p) => s + p.a, 0) / pts.length, mb = pts.reduce((s, p) => s + p.b, 0) / pts.length;
    const sa = Math.sqrt(pts.reduce((s, p) => s + (p.a - ma) ** 2, 0) / pts.length), sb = Math.sqrt(pts.reduce((s, p) => s + (p.b - mb) ** 2, 0) / pts.length);
    const X = pts.map((p) => [(p.a - ma) / sa, (p.b - mb) / sb]), y = pts.map((p) => p.y);
    const m = trainSVM(X, y);
    const acc = X.filter((x, i) => Math.sign(m.w[0] * x[0] + m.w[1] * x[1] + m.b) === y[i]).length / X.length;
    // граница: w0*(a-ma)/sa + w1*(b-mb)/sb + b = 0  ->  b(a)
    const line = [0.05, 0.9].map((a) => [a, mb - (sb / m.w[1]) * (m.w[0] * (a - ma) / sa + m.b)]);
    return { pts, acc, line };
  }, [data]);
  const batt = useMemo(() => {
    const r = rng(4);
    const pts = [];
    for (let t = 0; t <= 14; t += 0.25) pts.push([t, 16.8 - 0.05 * t - 0.0045 * t * t - 0.00018 * t ** 3 + r.gauss(0, 0.04)]);
    return pts;
  }, []);
  const coef = polyfit(batt.map((p) => p[0]), batt.map((p) => p[1]), deg);
  const fit = Array.from({ length: 101 }, (_, i) => { const t = (i / 100) * 30; return [t, polyval(coef, t)]; });
  const cut = fit.find((p) => p[0] > 14 && p[1] <= 14.0);
  const resid = Math.sqrt(batt.reduce((s, p) => s + (p[1] - polyval(coef, p[0])) ** 2, 0) / batt.length);
  return (
    <div className="grid g2">
      <div className="card tight">
        <h3>SVM: стресс / норма по NDVI и ΔT</h3>
        <LineChart height={260} xLabel="NDVI" yLabel="ΔT, °C" xMin={0.05} xMax={0.9} series={[
          { data: svm.pts.map((p) => [p.a, p.b, p.y < 0 ? '#4ade80' : '#f87171']), color: '#4ade80', scatter: true, r: 2.2, opacity: 0.7 },
          { name: 'Разделяющая гиперплоскость SVM', data: svm.line, color: '#e6ecf7', width: 2 },
        ]} />
        <p className="small" style={{ margin: 0 }}>Линейный SVM (алгоритм Pegasos, hinge-loss) на стандартизованных признаках. Точность: <b>{fmt(svm.acc * 100, 1)}%</b>. Зелёные — норма, красные — стресс.</p>
      </div>
      <div className="card tight">
        <h3>Полиномиальная регрессия: прогноз разряда АКБ</h3>
        <Slider label="Степень полинома" value={deg} min={1} max={5} step={1} onChange={setDeg} />
        <LineChart height={220} xLabel="время полёта, мин" yLabel="U, В" yMin={12.5} yMax={17.2} series={[
          { name: 'Телеметрия', data: batt, color: '#38bdf8', scatter: true, r: 2 },
          { name: `Полином ${deg}-й степени`, data: fit, color: '#fbbf24', width: 2 },
        ]} hLines={[{ y: 14.0, label: 'порог 14,0 В (4S)', color: '#f87171' }]} markers={[{ x: 14, label: 'сейчас', color: '#93a2bf' }]} />
        <p className="small" style={{ margin: 0 }}>RMSE аппроксимации: <b>{resid.toFixed(3)} В</b>. Прогноз достижения порога: <b>{cut ? `${cut[0].toFixed(1)} мин` : 'за пределами 30 мин'}</b> → остаток ≈ <b>{cut ? (cut[0] - 14).toFixed(1) : '>16'} мин</b>. Слишком высокая степень хорошо описывает обучающие точки, но даёт неустойчивую экстраполяцию (переобучение).</p>
      </div>
    </div>
  );
}

export default function Stage7() {
  const [tab, setTab] = useState('fuzzy');
  return (
    <>
      <StageHeader
        num={7}
        title="Принятие решений"
        subtitle="Интеллектуальное принятие решений системой: нечёткий вывод Мамдани оценивает риск полёта в условиях неопределённости, деревья решений и Random Forest классифицируют состояние посевов по данным ДЗЗ, SVM и регрессия прогнозируют параметры и тренды."
        tags={['Fuzzy Logic (Мамдани)', 'Decision Tree', 'Random Forest', 'XGBoost', 'SVM', 'Полиномиальная регрессия']}
      />
      <div className="grid g2">
        <div className="card"><h3>Выполняемые задачи</h3><Tasks items={['Создание программы в соответствии с блок-схемой (см. страницу «Программа»)', 'Выполнение программы и анализ решения', 'Оценка неопределённости и экспертные правила', 'Классификация данных ДЗЗ, прогнозирование трендов']} /></div>
        <div className="card">
          <h3>Входы → решение</h3>
          <p className="small" style={{ margin: 0 }}>Модуль получает от предыдущих этапов: заряд АКБ и ветер (телеметрия, этап 3), дистанцию до препятствий (этапы 4–5), индексы растительности (NDVI). Выход — команда для контура управления: продолжить, скорректировать манёвр или вернуться на базу, а также решение о съёмке зоны стресса.</p>
        </div>
      </div>

      <SectionTitle n="7.1">Модели ИИ</SectionTitle>
      <Tabs value={tab} onChange={setTab} tabs={[['fuzzy', 'Нечёткая логика (Мамдани)'], ['trees', 'Decision Tree / Random Forest'], ['ml', 'SVM и регрессия']]} />
      {tab === 'fuzzy' && <FuzzyBlock />}
      {tab === 'trees' && <TreesBlock />}
      {tab === 'ml' && <MLBlock />}

      <SectionTitle n="7.2">Программная реализация</SectionTitle>
      <div className="grid g2">
        <Code title="fuzzy_decision.py" code={`
import numpy as np, skfuzzy as fuzz
from skfuzzy import control as ctrl

battery = ctrl.Antecedent(np.arange(0, 101, 1), 'battery')
wind    = ctrl.Antecedent(np.arange(0, 20.1, 0.5), 'wind')
dist    = ctrl.Antecedent(np.arange(0, 51, 1), 'dist')
risk    = ctrl.Consequent(np.arange(0, 101, 1), 'risk', defuzzify_method='centroid')

battery['низкий']  = fuzz.trapmf(battery.universe, [0, 0, 20, 40])
battery['средний'] = fuzz.trapmf(battery.universe, [25, 45, 55, 75])
battery['высокий'] = fuzz.trapmf(battery.universe, [60, 80, 100, 100])
# ... аналогично wind, dist, risk

rules = [
    ctrl.Rule(battery['низкий'] | wind['сильный'] | dist['близко'], risk['высокий']),
    ctrl.Rule(battery['средний'] & wind['умеренный'], risk['средний']),
    ctrl.Rule(dist['средне'], risk['средний']),
    ctrl.Rule(battery['высокий'] & wind['слабый'] & dist['далеко'], risk['низкий']),
]
sim = ctrl.ControlSystemSimulation(ctrl.ControlSystem(rules))
sim.input['battery'], sim.input['wind'], sim.input['dist'] = 55, 9, 18
sim.compute()
r = sim.output['risk']
action = 'CONTINUE' if r < 35 else 'CORRECT' if r < 62 else 'RTH'
`} />
        <Code title="classify_crop.py" code={`
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from xgboost import XGBClassifier
import pandas as pd

df = pd.read_csv('field_samples.csv')     # NDVI, NDRE, soil_moisture, dT, label
X, y = df[['NDVI', 'NDRE', 'soil_moisture', 'dT']], df['label']
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=42)

models = {
    'DecisionTree': DecisionTreeClassifier(max_depth=4),
    'RandomForest': RandomForestClassifier(n_estimators=30, max_depth=6),
    'XGBoost':      XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.1),
    'SVM':          SVC(kernel='linear', C=1.0),
}
for name, m in models.items():
    m.fit(Xtr, ytr)
    print(name, accuracy_score(yte, m.predict(Xte)))
print(confusion_matrix(yte, models['RandomForest'].predict(Xte)))
print(dict(zip(X.columns, models['RandomForest'].feature_importances_)))
`} />
      </div>

      <SectionTitle n="7.3">Результаты системы</SectionTitle>
      <div className="grid g4">
        {[
          ['Прогнозирование', 'Прогноз траектории CNN–LSTM + EKF и разряда АКБ (регрессия)', '/program'],
          ['Результаты обучения', 'Кривые обучения нейросети, точность RF / DT / SVM', '/program'],
          ['Завершённая карта', '2D/3D карта занятости SLAM с объектами и траекторией', '/etap-5'],
          ['Движение дрона', 'ПИД-управление по траектории, облёт препятствий', '/etap-3'],
        ].map(([t, d, h]) => (
          <Link key={t} href={h} className="pager-card"><small>Результат</small><b>{t}</b><div className="small muted" style={{ marginTop: 4 }}>{d}</div></Link>
        ))}
      </div>
      <div className="note section">
        <b>Вывод.</b> Нечёткая логика формализует экспертные правила и плавно меняет решение при изменении условий (нет резких переключений на границах). Random Forest точнее одиночного дерева на тестовой выборке, так как усредняет ансамбль и меньше переобучается; наиболее информативны NDVI и ΔT. Линейный SVM уверенно отделяет «норму» от «стресса», а полиномиальная регрессия 2–3 степени даёт устойчивый прогноз остатка времени полёта.
      </div>
      <Pager href="/etap-7" />
    </>
  );
}
