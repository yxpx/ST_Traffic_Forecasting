"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { cn } from "@/lib/utils";
import { LayoutDashboard, LineChart, Layers, Activity, Route, Video } from "lucide-react";

const links = [
  { href: "/", label: "System Overview", icon: LayoutDashboard },
  { href: "/route-planner", label: "Route Planner", icon: Route },
  { href: "/cctv-analytics", label: "CCTV Analytics", icon: Video },
  { href: "/prediction-analysis", label: "Prediction Analysis", icon: LineChart },
  { href: "/feature-attribution", label: "Feature Attribution", icon: Layers },
  { href: "/model-performance", label: "Model Performance", icon: Activity },
];

export function Sidebar() {
  const pathname = usePathname();
  return (
    <aside className="fixed left-0 top-0 z-40 h-screen w-56 border-r border-border bg-background flex flex-col">
      <div className="p-5 border-b border-border">
        <h1 className="text-sm font-semibold tracking-wide text-foreground">
          ST Traffic Forecasting
        </h1>
        <p className="text-[11px] text-muted-foreground mt-0.5">METR-LA · 207 sensors</p>
      </div>
      <nav className="flex-1 p-3 space-y-0.5">
        {links.map(({ href, label, icon: Icon }) => (
          <Link
            key={href}
            href={href}
            className={cn(
              "flex items-center gap-2.5 rounded-md px-3 py-2 text-sm transition-colors",
              pathname === href
                ? "bg-accent text-accent-foreground font-medium"
                : "text-muted-foreground hover:text-foreground hover:bg-muted"
            )}
          >
            <Icon className="h-4 w-4" />
            {label}
          </Link>
        ))}
      </nav>
      <div className="p-4 border-t border-border">
        <p className="text-[10px] text-muted-foreground">ST Traffic Forecasting</p>
      </div>
    </aside>
  );
}
