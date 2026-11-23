import Link from 'next/link';
import SystemScheme from '@/components/SystemScheme';
import { Pager, StageHeader } from '@/components/UI';
import { PAGES, AREA } from '@/lib/pages';

const METHODS = [
  ['1', 'Машинное обучение (ML)', 'Линейная / полиномиальная регрессия, SVM', 'Прогнозирование параметров и трендов', 'Этап 1, 7'],
  ['2', 'Глубокое обучение (DL)', 'Свёрточные и рекуррентные сети (CNN–LSTM), YOLOv8', 'Анализ временных рядов, прогноз динамики БПЛА и природных процессов', 'Этап 4, Программа'],
  ['3', 'Деревья решений', 'Decision Tree, Random Forest, XGBoost', 'Классификация данных ДЗЗ и ГИС, выбор оптимальных параметров', 'Этап 7, NDVI'],
  ['4', 'Нечёткая логика (Fuzzy Logic)', 'Нечёткие правила, система Мамдани', 'Оценка неопределённости, моделирование экспертных решений', 'Этап 7'],
  ['5', 'Гибридные подходы (AI + EKF)', 'CNN–LSTM + расширенный фильтр Калмана (EKF)', 'Повышение устойчивости и точности прогнозных моделей', 'Программа'],
];

const STAGE_INFO = {
  1: 'Сбор данных с камеры и GPS, датасеты, фильтрация шума, нормализация, коррекция GPS.',
  2: 'Круговая, восьмёрочная и произвольная траектории, A* с учётом препятствий, оптимальный маршрут.',
  3: 'ПИД-регулятор в реальном времени, коррекция скорости и курса, манёвр облёта, PPO.',
  4: 'YOLOv8: распознавание объектов, NMS, расчёт координат объектов, препятствия.',
  5: 'SLAM: сопоставление сканов, сетка занятости, 2D/3D карта с траекторией дрона.',
  6: 'WGS-84 → ECEF → ENU, UTM, геодезические расчёты, повышение точности (Калман).',
  7: 'Нечёткая логика Мамдани, Random Forest, SVM, регрессия, итоговые результаты.',
};

export default function Home() {
  return (
    <>
      <StageHeader
        title="Интеллектуальная система управления БПЛА"
        subtitle="Веб-приложение по теме «Проектирование интеллектуальных систем управления». Система собирает и обрабатывает данные, анализирует их, применяет алгоритмы ИИ, принимает решения, управляет дроном и рассчитывает NDVI. Нажмите на любой этап схемы, чтобы перейти к подробному описанию и интерактивной демонстрации."
        tags={['7 этапов', 'Программа по блок-схеме', 'NDVI', 'ML · DL · RF · Fuzzy · EKF']}
      />

      <div className="card">
        <div className="row" style={{ marginBottom: 8 }}>
          <h2 style={{ margin: 0 }}>Схема системы (рисунок 1)</h2>
          <span className="spacer" />
          <span className="badge ok">Блоки-этапы кликабельны</span>
        </div>
        <SystemScheme />
      </div>

      <div className="section-title"><h2 style={{ margin: 0 }}>Структура приложения</h2></div>
      <div className="grid g3">
        {PAGES.slice(1).map((p) => (
          <Link key={p.href} href={p.href} className="pager-card">
            <small>{p.num ? `Этап ${p.num}` : p.label}</small>
            <b>{p.title}</b>
            <div className="muted small" style={{ marginTop: 6 }}>
              {p.num ? STAGE_INFO[p.num] : p.href === '/program'
                ? 'Код по блок-схеме рис. 2, пошаговое выполнение, CNN–LSTM–EKF, результаты и автотесты.'
                : 'Расчёт NDVI = (NIR − Red)/(NIR + Red), классификация фитостресса, решение о моменте съёмки.'}
            </div>
          </Link>
        ))}
      </div>

      <div className="section-title"><h2 style={{ margin: 0 }}>Методы искусственного интеллекта</h2></div>
      <div className="card table-wrap">
        <table>
          <thead>
            <tr><th>№</th><th>Класс методов ИИ</th><th>Используемый алгоритм</th><th>Назначение / тип задачи</th><th>Где показан</th></tr>
          </thead>
          <tbody>
            {METHODS.map((m) => (
              <tr key={m[0]}>{m.map((c, i) => <td key={i}>{c}</td>)}</tr>
            ))}
          </tbody>
        </table>
      </div>

      <div className="grid g2 section">
        <div className="card">
          <h3>Что делает система</h3>
          <ol style={{ margin: 0, paddingLeft: 20 }}>
            <li>Собирает данные, описывает набор данных (датасет) и его использование.</li>
            <li>Обрабатывает (фильтрация, нормализация, коррекция).</li>
            <li>Анализирует (детекция объектов, карта, временные ряды).</li>
            <li>Использует алгоритмы ИИ (ML, DL, деревья, нечёткая логика, гибрид с EKF).</li>
            <li>Принимает решения.</li>
            <li>Управляет дроном.</li>
            <li>Рассчитывает NDVI.</li>
          </ol>
        </div>
        <div className="card">
          <h3>Район работ</h3>
          <p className="muted">{AREA.name}</p>
          <dl className="kv">
            <dt>Точка старта (WGS-84)</dt><dd>{AREA.lat0.toFixed(4)}° с.ш., {AREA.lon0.toFixed(4)}° в.д.</dd>
            <dt>Высота над эллипсоидом</dt><dd>{AREA.h0} м</dd>
            <dt>Зона UTM</dt><dd>41N</dd>
            <dt>Культура</dt><dd>рис, пшеница (мониторинг NDVI)</dd>
          </dl>
          <p className="small muted" style={{ marginTop: 10 }}>
            Все вычисления (фильтры, A*, ПИД, SLAM, EKF, обучение моделей) выполняются прямо в браузере на синтетических данных, сгенерированных с фиксированным зерном, поэтому результаты воспроизводимы.
          </p>
        </div>
      </div>

      <details className="card section">
        <summary style={{ cursor: 'pointer', fontWeight: 600 }}>Исходный рисунок 1 из задания</summary>
        <div className="figure" style={{ marginTop: 12, maxWidth: 760 }}>
          <img src="/figures/fig1.png" alt="Рисунок 1 — схема системы" />
        </div>
      </details>

      <Pager href="/" />
    </>
  );
}
