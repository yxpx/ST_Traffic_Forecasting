// Shared chart color constants for Recharts (oklch for modern browsers)
export const C = {
  chart1: "oklch(0.7137 0.1434 254.6240)",
  chart2: "oklch(0.6231 0.1880 259.8145)",
  chart3: "oklch(0.5461 0.2152 262.8809)",
  chart4: "oklch(0.4882 0.2172 264.3763)",
  chart5: "oklch(0.4244 0.1809 265.6377)",
  destructive: "oklch(0.6368 0.2078 25.3313)",
  foreground: "oklch(0.9219 0 0)",
  muted: "oklch(0.7155 0 0)",
  border: "oklch(0.3715 0 0)",
  card: "oklch(0.2686 0 0)",
  bg: "oklch(0.2046 0 0)",
  primary: "oklch(0.6231 0.1880 259.8145)",
  accent: "oklch(0.3791 0.1378 265.5222)",
  accentFg: "oklch(0.8823 0.0571 254.1284)",
};

export const tooltipStyle = {
  contentStyle: {
    background: "#1a1a1a",
    border: "1px solid #444",
    borderRadius: 8,
    fontSize: 12,
    color: "#eee",
  },
  labelStyle: { color: "#bbb" },
  itemStyle: { color: "#ddd" },
  cursor: false as const,
  formatter: (value: number | string) => typeof value === "number" ? value.toFixed(3) : value,
};

export const gridStroke = C.border;
export const axisStroke = C.muted;
