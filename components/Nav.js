'use client';
import Link from 'next/link';
import { usePathname } from 'next/navigation';
import { PAGES } from '@/lib/pages';

export default function Nav() {
  const path = usePathname() || '/';
  const idx = PAGES.findIndex((p) => p.href === path);
  return (
    <header className="topbar">
      <div className="topbar-inner">
        <Link href="/" className="brand">
          <span className="brand-logo">AI</span>
          <span>
            БПЛА · ИСУ
            <small>интеллектуальная система управления</small>
          </span>
        </Link>
        <nav className="stepper">
          {PAGES.map((p, i) => (
            <Link key={p.href} href={p.href} className={`step ${i === idx ? 'active' : ''} ${idx > i ? 'done' : ''}`} title={p.title}>
              <span className="dot">{p.num ?? (i === 0 ? '◎' : i === 8 ? 'P' : 'N')}</span>
              {p.label}
            </Link>
          ))}
        </nav>
      </div>
    </header>
  );
}
