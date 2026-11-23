import './globals.css';
import Nav from '@/components/Nav';

export const metadata = {
  title: 'Интеллектуальная система управления БПЛА',
  description: 'Проектирование интеллектуальных систем управления: 7 этапов, программа по блок-схеме и анализ NDVI',
};

export default function RootLayout({ children }) {
  return (
    <html lang="ru">
      <body>
        <Nav />
        <main className="container">{children}</main>
        <footer className="footer">
          Проектирование интеллектуальных систем управления · БПЛА + ИИ (ML, DL, Random Forest, Fuzzy Logic, CNN–LSTM + EKF) · Next.js
        </footer>
      </body>
    </html>
  );
}
