import type { Metadata } from "next";
import { GeistSans } from "geist/font/sans";
import { GeistMono } from "geist/font/mono";
import { Sidebar } from "@/components/sidebar";
import "./globals.css";

export const metadata: Metadata = {
  title: "Spatio-Temporal Traffic Forecasting",
  description: "METR-LA 207-sensor traffic prediction dashboard",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" className="dark">
      <head>
        <link rel="icon" href="/favicon.svg" type="image/svg+xml" />
      </head>
      <body className={`${GeistSans.variable} ${GeistMono.variable} font-sans antialiased`}>
        <Sidebar />
        <main className="ml-56 min-h-screen p-6">
          {children}
        </main>
      </body>
    </html>
  );
}
