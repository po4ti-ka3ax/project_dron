export const PAGES = [
  { href: '/', label: 'Схема', title: 'Интеллектуальная информационная система' },
  { href: '/etap-1', num: 1, label: 'Этап 1', title: 'Подготовка данных и ПО' },
  { href: '/etap-2', num: 2, label: 'Этап 2', title: 'Генерация траектории' },
  { href: '/etap-3', num: 3, label: 'Этап 3', title: 'Управление полётом' },
  { href: '/etap-4', num: 4, label: 'Этап 4', title: 'Обнаружение объектов' },
  { href: '/etap-5', num: 5, label: 'Этап 5', title: 'Создание карты' },
  { href: '/etap-6', num: 6, label: 'Этап 6', title: 'Изменение координат' },
  { href: '/etap-7', num: 7, label: 'Этап 7', title: 'Принятие решений' },
  { href: '/program', label: 'Программа', title: 'Программа по блок-схеме (рис. 2)' },
  { href: '/ndvi', label: 'NDVI', title: 'Расчёт NDVI и управление съёмкой (рис. 3)' },
];

export function neighbours(href) {
  const i = PAGES.findIndex((p) => p.href === href);
  return { prev: i > 0 ? PAGES[i - 1] : null, next: i < PAGES.length - 1 ? PAGES[i + 1] : null };
}

// Район работ: окрестности г. Кызылорда (рисовые поля долины Сырдарьи)
export const AREA = {
  name: 'Кызылординская обл., рисовые поля в долине р. Сырдарья (условный участок 600 × 400 м)',
  lat0: 44.8125,
  lon0: 65.4580,
  h0: 128.0,
};
