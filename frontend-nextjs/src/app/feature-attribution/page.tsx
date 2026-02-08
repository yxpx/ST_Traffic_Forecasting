"use client";

import { useEffect, useState } from "react";
import {
  BarChart, Bar, PieChart, Pie, Cell,
  XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Legend,
} from "recharts";
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";
import {
  Table, TableHeader, TableRow, TableHead, TableBody, TableCell,
} from "@/components/ui/table";
import { C, tooltipStyle, gridStroke, axisStroke } from "@/lib/theme";

interface DashboardData {
  feature_importance: { feature: string; importance: number }[];
  sensor_performance: { best_5: { sensor: number; mae: number }[]; worst_5: { sensor: number; mae: number }[] };
}

const CATEGORY_MAP: Record<string, string> = {
  "Speed (z-scored)": "Traffic",
  "Time of Day (sin)": "Temporal",
  "Time of Day (cos)": "Temporal",
  "Visibility": "Weather",
  "Wind Direction": "Weather",
  "Sea-Level Pressure": "Weather",
  "Relative Humidity": "Weather",
  "Air Temperature": "Weather",
  "Pressure (alt)": "Weather",
  "Wind Speed": "Weather",
  "Feature 11": "Other",
};

const CATEGORY_COLORS: Record<string, string> = {
  Traffic: "#3b82f6",
  Temporal: "#f59e0b",
  Weather: "#10b981",
  Other: "#8b5cf6",
};

export default function FeatureAttributionPage() {
  const [data, setData] = useState<DashboardData | null>(null);

  useEffect(() => {
    fetch("/dashboard-data.json").then((r) => r.json()).then(setData);
  }, []);

  if (!data) return <div className="flex items-center justify-center h-96 text-muted-foreground">Loading...</div>;

  const features = data.feature_importance
    .map((f) => ({
      ...f,
      category: CATEGORY_MAP[f.feature] || "Other",
      color: CATEGORY_COLORS[CATEGORY_MAP[f.feature] || "Other"] || "#8b5cf6",
    }))
    .sort((a, b) => b.importance - a.importance);

  // Category aggregation for pie
  const catAgg: Record<string, number> = {};
  features.forEach((f) => {
    catAgg[f.category] = (catAgg[f.category] || 0) + f.importance;
  });
  const pieData = Object.entries(catAgg).map(([name, value]) => ({
    name,
    value: +value.toFixed(4),
    color: CATEGORY_COLORS[name] || "#8b5cf6",
  }));

  // Combined sensor comparison
  const sensorComparison = [
    ...data.sensor_performance.best_5.map((s) => ({ ...s, group: "Best" })),
    ...data.sensor_performance.worst_5.map((s) => ({ ...s, group: "Worst" })),
  ];

  return (
    <div className="space-y-6">
      <div>
        <h2 className="text-2xl font-bold tracking-tight text-foreground">Feature Attribution</h2>
        <p className="text-sm text-muted-foreground mt-1">
          Gradient-based importance analysis of input features on model predictions
        </p>
      </div>

      {/* Feature Importance Bar */}
      <Card>
        <CardHeader className="pb-2">
          <CardTitle className="text-base">Feature Importance Ranking</CardTitle>
          <CardDescription>Sorted by absolute gradient magnitude (higher = more influential)</CardDescription>
        </CardHeader>
        <CardContent>
          <ResponsiveContainer width="100%" height={400}>
            <BarChart data={features} layout="vertical" margin={{ left: 20 }}>
              <CartesianGrid strokeDasharray="3 3" stroke={gridStroke} />
              <XAxis type="number" tick={{ fill: axisStroke, fontSize: 11 }} stroke={gridStroke} />
              <YAxis type="category" dataKey="feature" tick={{ fill: axisStroke, fontSize: 11 }} stroke={gridStroke} width={80} />
              <Tooltip {...tooltipStyle} />
              <Bar dataKey="importance" radius={[0, 4, 4, 0]}>
                {features.map((f, i) => (
                  <Cell key={i} fill={f.color} fillOpacity={0.8} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
          <div className="flex items-center justify-center gap-6 mt-4">
            {Object.entries(CATEGORY_COLORS).map(([cat, color]) => (
              <div key={cat} className="flex items-center gap-2">
                <span className="inline-block w-3 h-3 rounded-sm" style={{ background: color }} />
                <span className="text-xs text-muted-foreground">{cat}</span>
              </div>
            ))}
          </div>
        </CardContent>
      </Card>

      {/* Category Pie + Table */}
      <div className="grid md:grid-cols-2 gap-6">
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-base">Attribution by Category</CardTitle>
            <CardDescription>Aggregate importance grouped by feature type</CardDescription>
          </CardHeader>
          <CardContent>
            <ResponsiveContainer width="100%" height={300}>
              <PieChart>
                <Pie
                  data={pieData}
                  cx="50%"
                  cy="50%"
                  innerRadius={60}
                  outerRadius={110}
                  dataKey="value"
                  nameKey="name"
                  label={({ name, percent }) => `${name} ${(percent * 100).toFixed(0)}%`}
                  labelLine={{ stroke: "#888" }}
                  stroke="none"
                >
                  {pieData.map((d, i) => (
                    <Cell key={i} fill={d.color} />
                  ))}
                </Pie>
                <Tooltip {...tooltipStyle} />
              </PieChart>
            </ResponsiveContainer>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-base">Feature Breakdown</CardTitle>
            <CardDescription>Detailed importance values per feature</CardDescription>
          </CardHeader>
          <CardContent>
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead>Feature</TableHead>
                  <TableHead>Category</TableHead>
                  <TableHead className="text-right">Importance</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {features.map((f) => (
                  <TableRow key={f.feature}>
                    <TableCell className="font-medium">{f.feature}</TableCell>
                    <TableCell>
                      <span className="inline-block w-2 h-2 rounded-full mr-2" style={{ background: f.color }} />
                      {f.category}
                    </TableCell>
                    <TableCell className="text-right font-mono text-sm">{f.importance.toFixed(4)}</TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </CardContent>
        </Card>
      </div>

      {/* Methodology Note */}
      <Card>
        <CardContent className="p-5">
          <p className="text-sm text-muted-foreground leading-relaxed">
            <strong className="text-foreground">Methodology.</strong> Feature importance is computed via
            gradient-based attribution — the absolute mean gradient of the loss with respect to each input
            feature, averaged across a random subset of validation samples. This approximates each feature&apos;s
            marginal contribution to the prediction. Traffic-derived features (lagged speeds, rolling statistics)
            dominate, while temporal encodings provide secondary context for diurnal patterns.
          </p>
        </CardContent>
      </Card>
    </div>
  );
}
