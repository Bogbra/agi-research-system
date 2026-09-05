import type { Metadata } from "next";
import Link from "next/link";

import "./globals.css";

export const metadata: Metadata = {
  title: "AGI Research Console",
  description: "Multi-agent research pipeline: run queue, discovery, and AGI-potential rankings.",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body className="min-h-screen bg-bg">
        <header className="border-b border-border bg-surface">
          <div className="mx-auto flex max-w-6xl items-center justify-between px-6 py-4">
            <Link href="/" className="text-sm font-semibold tracking-wide text-fg">
              AGI Research Console
            </Link>
            <nav className="flex gap-6 text-sm text-muted">
              <Link href="/" className="hover:text-fg">
                Runs
              </Link>
              <Link href="/new" className="hover:text-fg">
                New run
              </Link>
            </nav>
          </div>
        </header>
        <main className="mx-auto max-w-6xl px-6 py-8">{children}</main>
      </body>
    </html>
  );
}
