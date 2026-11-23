'use client';
import Link from 'next/link';
import { useState } from 'react';
import { neighbours, PAGES } from '@/lib/pages';

export function StageHeader({ num, title, subtitle, tags = [] }) {
  return (
    <div className="hero stage-head">
      {num ? <div className="stage-num">{num}</div> : null}
      <div>
        <div className="eyebrow">{num ? `Этап ${num} из 7` : 'Интеллектуальная система управления БПЛА'}</div>
        <h1>{title}</h1>
        {subtitle ? <p className="lead">{subtitle}</p> : null}
        {tags.length ? (
          <div className="tag-row">
            {tags.map((t) => (
              <span key={t} className="badge info">{t}</span>
            ))}
          </div>
        ) : null}
      </div>
    </div>
  );
}

export function Tasks({ items }) {
  return (
    <ul className="tasks">
      {items.map((t) => (
        <li key={t}>{t}</li>
      ))}
    </ul>
  );
}

export function SectionTitle({ n, children }) {
  return (
    <div className="section-title">
      {n ? <span className="num">{n}</span> : null}
      <h2 style={{ margin: 0 }}>{children}</h2>
    </div>
  );
}

export function Pager({ href }) {
  const { prev, next } = neighbours(href);
  const name = (p) => (p.num ? `Этап ${p.num}: ${p.title}` : p.title);
  return (
    <div className="pager">
      {prev ? (
        <Link href={prev.href} className="pager-card">
          <small>← Назад</small>
          <b>{name(prev)}</b>
        </Link>
      ) : <span />}
      {next ? (
        <Link href={next.href} className="pager-card cta next">
          <small>Следующая страница →</small>
          <b>{name(next)}</b>
        </Link>
      ) : (
        <Link href={PAGES[0].href} className="pager-card cta next">
          <small>Готово →</small>
          <b>Вернуться к схеме системы</b>
        </Link>
      )}
    </div>
  );
}

export function Metric({ k, v, color }) {
  return (
    <div className="metric">
      <div className="v" style={color ? { color } : null}>{v}</div>
      <div className="k">{k}</div>
    </div>
  );
}

export function Slider({ label, value, min, max, step = 1, onChange, unit = '', digits }) {
  const d = digits ?? (step < 1 ? String(step).split('.')[1]?.length || 2 : 0);
  return (
    <label className="ctl">
      <span className="row">
        {label}
        <span className="spacer" />
        <span className="val">{Number(value).toFixed(d)}{unit}</span>
      </span>
      <input type="range" min={min} max={max} step={step} value={value} onChange={(e) => onChange(parseFloat(e.target.value))} />
    </label>
  );
}

export function Seg({ options, value, onChange }) {
  return (
    <div className="seg">
      {options.map(([v, l]) => (
        <button key={v} className={v === value ? 'on' : ''} onClick={() => onChange(v)} type="button">
          {l}
        </button>
      ))}
    </div>
  );
}

export function Tabs({ tabs, value, onChange }) {
  return (
    <div className="tabs">
      {tabs.map(([v, l]) => (
        <button key={v} className={v === value ? 'on' : ''} onClick={() => onChange(v)} type="button">
          {l}
        </button>
      ))}
    </div>
  );
}

// ---------- Подсветка кода (Python / JS) ----------
const KW = new Set(
  'import from as def return if elif else for while in not and or is None True False class with try except finally lambda yield break continue pass const let var function new export default async await of this null undefined true false print self'.split(' ')
);
function highlight(src) {
  const re = /(#[^\n]*|\/\/[^\n]*)|("""[\s\S]*?"""|'[^'\n]*'|"[^"\n]*"|`[^`]*`)|(\b\d+(?:\.\d+)?(?:e-?\d+)?\b)|([A-Za-z_][A-Za-z0-9_]*)(?=\s*\()|([A-Za-z_][A-Za-z0-9_]*)/g;
  const out = [];
  let last = 0, m, i = 0;
  while ((m = re.exec(src))) {
    if (m.index > last) out.push(src.slice(last, m.index));
    const [tok, c, s, n, f, w] = m;
    if (c) out.push(<span key={i++} className="tok-c">{tok}</span>);
    else if (s) out.push(<span key={i++} className="tok-s">{tok}</span>);
    else if (n) out.push(<span key={i++} className="tok-n">{tok}</span>);
    else if (f) out.push(<span key={i++} className={KW.has(f) ? 'tok-k' : 'tok-f'}>{tok}</span>);
    else if (w && KW.has(w)) out.push(<span key={i++} className="tok-k">{tok}</span>);
    else out.push(tok);
    last = m.index + tok.length;
  }
  out.push(src.slice(last));
  return out;
}

export function Code({ code, title = 'Python', maxHeight }) {
  const [copied, setCopied] = useState(false);
  const copy = async () => {
    try {
      await navigator.clipboard.writeText(code);
      setCopied(true);
      setTimeout(() => setCopied(false), 1500);
    } catch (e) { /* буфер недоступен */ }
  };
  return (
    <div className="code">
      <div className="code-head">
        <span>{title}</span>
        <button className="btn small" onClick={copy} type="button">{copied ? 'Скопировано' : 'Копировать'}</button>
      </div>
      <pre style={maxHeight ? { maxHeight } : null}>{highlight(code.trim())}</pre>
    </div>
  );
}
