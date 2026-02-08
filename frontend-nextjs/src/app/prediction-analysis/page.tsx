"use client";

import { useEffect, useState, useMemo } from "react";
import {
  ScatterChart, Scatter, LineChart, Line, BarChart, Bar,
  XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Cell, Legend,
  ReferenceLine,
} from "recharts";
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";
import { C, tooltipStyle, gridStroke, axisStroke } from "@/lib/theme";

interface DashboardData {
  scatter: { actual: number; predicted: number }[];
  timeseries: Record<string, { actual: number[]; predicted: number[] }>;
  error_histogram: { bin_start: number; bin_end: number; count: number }[];
}

interface HeatmapData {
  sensor_ids: string[];
  hourly_actual: number[][];
  hourly_predicted: number[][];
  unit: string;
  steps_per_hour: number;
  start_index: number;
}

interface TimeSeriesData {
  sensor_ids: string[];
  series: Record<string, { actual: number[]; predicted: number[] }>;
  unit: string;
  start_index: number;
  steps: number;
}

export default function PredictionAnalysisPage() {
  const [data, setData] = useState<DashboardData | null>(null);
  const [selectedSensor, setSelectedSensor] = useState("");
  const [heatmapData, setHeatmapData] = useState<HeatmapData | null>(null);
  const [timeSeriesData, setTimeSeriesData] = useState<TimeSeriesData | null>(null);
  const [sensorInput, setSensorInput] = useState("");

  useEffect(() => {
    fetch("/dashboard-data.json").then((r) => r.json()).then((d) => {
      setData(d);
      const keys = Object.keys(d.timeseries);
      if (keys.length) setSelectedSensor(keys[0]);
    });
  }, []);

  useEffect(() => {
    fetch("/heatmap-data.json").then((r) => r.json()).then(setHeatmapData).catch(() => null);
  }, []);

  useEffect(() => {
    fetch("/timeseries-data.json").then((r) => r.json()).then(setTimeSeriesData).catch(() => null);
  }, []);

  const timeSeries = timeSeriesData?.series ?? data?.timeseries ?? {};
  const sensorIds = timeSeriesData?.sensor_ids ?? Object.keys(timeSeries);
  const raw = timeSeries[selectedSensor];
  const ts = raw ? raw.actual.slice(0, 200).map((a, i) => ({ t: i, actual: a, predicted: raw.predicted[i] })) : [];
  const unitLabel = timeSeriesData?.unit ? ` (${timeSeriesData.unit})` : "";

  useEffect(() => {
    if (!sensorIds.length) return;
    if (!selectedSensor || !timeSeries[selectedSensor]) {
      setSelectedSensor(sensorIds[0]);
      setSensorInput("1");
      return;
    }
    const idx = sensorIds.indexOf(selectedSensor);
    if (idx >= 0) setSensorInput(String(idx + 1));
  }, [sensorIds, selectedSensor, timeSeries]);

  if (!data || !selectedSensor) return <div className="flex items-center justify-center h-96 text-muted-foreground">Loading...</div>;

  // Residuals
  const residuals = data.scatter.map((d) => ({
    actual: d.actual,
    residual: d.predicted - d.actual,
  }));

  // Percentile errors
  const errors = data.scatter.map((d) => Math.abs(d.predicted - d.actual)).sort((a, b) => a - b);
  const getP = (p: number) => errors[Math.floor(p / 100 * errors.length)] || 0;
  const percentiles = [
    { name: "P50", value: getP(50) },
    { name: "P75", value: getP(75) },
    { name: "P90", value: getP(90) },
    { name: "P95", value: getP(95) },
    { name: "P99", value: getP(99) },
  ];

  const histData = data.error_histogram.map((d) => ({
    bin: `${d.bin_start.toFixed(2)}`,
    count: d.count,
  }));

  // Linear regression for scatter
  const n = data.scatter.length;
  const sumX = data.scatter.reduce((s, d) => s + d.actual, 0);
  const sumY = data.scatter.reduce((s, d) => s + d.predicted, 0);
  const sumXY = data.scatter.reduce((s, d) => s + d.actual * d.predicted, 0);
  const sumX2 = data.scatter.reduce((s, d) => s + d.actual * d.actual, 0);
  const regSlope = (n * sumXY - sumX * sumY) / (n * sumX2 - sumX * sumX);
  const regIntercept = (sumY - regSlope * sumX) / n;
  const xMin = Math.min(...data.scatter.map((d) => d.actual));
  const xMax = Math.max(...data.scatter.map((d) => d.actual));

  return (
    <div className="space-y-6">
      <div>
        <h2 className="text-2xl font-bold tracking-tight text-foreground">Prediction Analysis</h2>
        <p className="text-sm text-muted-foreground mt-1">
          Evaluation of model predictions against observed traffic measurements
        </p>
      </div>

      {/* Scatter + Residuals */}
      <div className="grid md:grid-cols-2 gap-6">
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-base">Actual vs. Predicted</CardTitle>
            <CardDescription>Ideal predictions follow the identity line</CardDescription>
          </CardHeader>
          <CardContent>
            <ResponsiveContainer width="100%" height={340}>
              <ScatterChart>
                <CartesianGrid strokeDasharray="3 3" stroke={gridStroke} />
                <XAxis type="number" dataKey="actual" name="Actual" tick={{ fill: axisStroke, fontSize: 11 }} stroke={gridStroke} label={{ value: "Actual", fill: axisStroke, fontSize: 12, position: "insideBottom", offset: -2 }} />
                <YAxis type="number" dataKey="predicted" name="Predicted" tick={{ fill: axisStroke, fontSize: 11 }} stroke={gridStroke} label={{ value: "Predicted", fill: axisStroke, fontSize: 12, angle: -90, position: "insideLeft" }} />
                <Tooltip {...tooltipStyle} />
                <ReferenceLine
                  segment={[{ x: xMin, y: xMin }, { x: xMax, y: xMax }]}
                  stroke={C.destructive}
                  strokeWidth={2}
                  strokeDasharray="6 3"
                  label={{ value: "y = x (ideal)", fill: C.destructive, fontSize: 11, position: "insideTopLeft" }}
                />
                <ReferenceLine
                  segment={[
                    { x: xMin, y: regSlope * xMin + regIntercept },
                    { x: xMax, y: regSlope * xMax + regIntercept },
                  ]}
                  stroke="#10b981"
                  strokeWidth={2}
                  label={{ value: "Regression fit", fill: "#10b981", fontSize: 11, position: "insideBottomRight" }}
                />
                <Scatter data={data.scatter} fill={C.chart2} fillOpacity={0.3} r={0.4} />
              </ScatterChart>
            </ResponsiveContainer>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-base">Residual Distribution</CardTitle>
            <CardDescription>Prediction error by actual value (should scatter around zero)</CardDescription>
          </CardHeader>
          <CardContent>
            <ResponsiveContainer width="100%" height={340}>
              <ScatterChart>
                <CartesianGrid strokeDasharray="3 3" stroke={gridStroke} />
                <XAxis type="number" dataKey="actual" name="Actual" tick={{ fill: axisStroke, fontSize: 11 }} stroke={gridStroke} />
                <YAxis type="number" dataKey="residual" name="Residual" tick={{ fill: axisStroke, fontSize: 11 }} stroke={gridStroke} />
                <Tooltip {...tooltipStyle} />
                <ReferenceLine y={0} stroke={C.muted} strokeDasharray="4 2" />
                <Scatter data={residuals} fill={C.chart3} fillOpacity={0.25} r={0.4} />
              </ScatterChart>
            </ResponsiveContainer>
          </CardContent>
        </Card>
      </div>

      {/* Time Series with Sensor Selector */}
      <Card>
        <CardHeader className="pb-2">
          <div className="flex items-center justify-between">
            <div>
              <CardTitle className="text-base">Forecast Time Series</CardTitle>
              <CardDescription>Actual vs. predicted over 200 time steps{unitLabel}</CardDescription>
            </div>
            <div className="flex items-center gap-2">
              <input
                type="number"
                min={1}
                max={sensorIds.length || 1}
                value={sensorInput}
                onChange={(e) => {
                  const value = e.target.value;
                  setSensorInput(value);
                  const num = Number(value);
                  if (!Number.isFinite(num)) return;
                  const idx = Math.min(Math.max(Math.floor(num), 1), sensorIds.length) - 1;
                  const id = sensorIds[idx];
                  if (id) setSelectedSensor(id);
                }}
                className="w-20 rounded-md border border-border bg-card text-foreground text-sm px-2 py-1.5 focus:outline-none focus:ring-1 focus:ring-ring"
              />
              <select
                className="rounded-md border border-border bg-card text-foreground text-sm px-3 py-1.5 focus:outline-none focus:ring-1 focus:ring-ring"
                value={selectedSensor}
                onChange={(e) => setSelectedSensor(e.target.value)}
              >
                {sensorIds.map((s, i) => (
                  <option key={s} value={s}>Sensor {i + 1}</option>
                ))}
              </select>
            </div>
          </div>
        </CardHeader>
        <CardContent>
          <ResponsiveContainer width="100%" height={340}>
            <LineChart data={ts}>
              <CartesianGrid strokeDasharray="3 3" stroke={gridStroke} />
              <XAxis dataKey="t" tick={{ fill: axisStroke, fontSize: 11 }} stroke={gridStroke} />
              <YAxis tick={{ fill: axisStroke, fontSize: 11 }} stroke={gridStroke} />
              <Tooltip {...tooltipStyle} />
              <Legend wrapperStyle={{ fontSize: 12, color: C.muted }} />
              <Line type="monotone" dataKey="actual" stroke={C.muted} strokeWidth={1.5} dot={false} name="Actual" />
              <Line type="monotone" dataKey="predicted" stroke={C.chart1} strokeWidth={2} dot={false} name="Predicted" />
            </LineChart>
          </ResponsiveContainer>
        </CardContent>
      </Card>

      {/* Error Histogram + Percentiles */}
      <div className="grid md:grid-cols-2 gap-6">
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-base">Error Histogram</CardTitle>
            <CardDescription>Distribution of absolute prediction errors</CardDescription>
          </CardHeader>
          <CardContent>
            <ResponsiveContainer width="100%" height={280}>
              <BarChart data={histData}>
                <CartesianGrid strokeDasharray="3 3" stroke={gridStroke} />
                <XAxis dataKey="bin" tick={{ fill: axisStroke, fontSize: 10 }} stroke={gridStroke} interval={4} />
                <YAxis tick={{ fill: axisStroke, fontSize: 11 }} stroke={gridStroke} />
                <Tooltip {...tooltipStyle} />
                <Bar dataKey="count" radius={[4, 4, 0, 0]} fill={C.chart2} fillOpacity={0.75} />
              </BarChart>
            </ResponsiveContainer>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-base">Error Percentiles</CardTitle>
            <CardDescription>Absolute error at key percentile thresholds</CardDescription>
          </CardHeader>
          <CardContent>
            <ResponsiveContainer width="100%" height={280}>
              <BarChart data={percentiles}>
                <CartesianGrid strokeDasharray="3 3" stroke={gridStroke} />
                <XAxis dataKey="name" tick={{ fill: axisStroke, fontSize: 11 }} stroke={gridStroke} />
                <YAxis tick={{ fill: axisStroke, fontSize: 11 }} stroke={gridStroke} />
                <Tooltip {...tooltipStyle} />
                <Bar dataKey="value" radius={[4, 4, 0, 0]}>
                  {percentiles.map((_, i) => (
                    <Cell key={i} fill={[C.chart1, C.chart2, C.chart3, C.chart4, C.chart5][i]} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </CardContent>
        </Card>
      </div>

      {/* Actual vs Predicted Heatmap */}
      <Card>
        <CardHeader className="pb-2">
          <CardTitle className="text-base">Actual vs Predicted Speed</CardTitle>
          <CardDescription>Sensor-by-hour comparison of observed and forecast speeds.</CardDescription>
        </CardHeader>
        <CardContent className="p-3">
          {heatmapData ? (
            <SensorHourHeatmap data={heatmapData} mode="compare" />
          ) : (
            <div className="h-64 flex items-center justify-center text-muted-foreground text-sm">Loading heatmap...</div>
          )}
        </CardContent>
      </Card>

      {/* Error Heatmap */}
      <Card>
        <CardHeader className="pb-2">
          <CardTitle className="text-base">Absolute Error</CardTitle>
          <CardDescription>Sensor-by-hour absolute prediction error.</CardDescription>
        </CardHeader>
        <CardContent className="p-3">
          {heatmapData ? (
            <SensorHourHeatmap data={heatmapData} mode="error" />
          ) : (
            <div className="h-64 flex items-center justify-center text-muted-foreground text-sm">Loading heatmap...</div>
          )}
        </CardContent>
      </Card>
    </div>
  );
}

/** Sensor × Hour heatmap for actual/predicted comparison or absolute error */
function SensorHourHeatmap({
  data,
  mode,
}: {
  data: HeatmapData;
  mode: "compare" | "error";
}) {
  const [hover, setHover] = useState<{ sensorId: string; sensorIndex: number; hour: number; actual: number; predicted: number; error: number } | null>(null);

  const { sensorKeys, hours, minVal, maxVal, maxError, errorGrid, valueGrid } = useMemo(() => {
    const hourCount = 24;
    const keys = data.sensor_ids;
    const actualRows = data.hourly_actual;
    const predictedRows = data.hourly_predicted;

    let lo = Infinity;
    let hi = -Infinity;
    let errHi = -Infinity;

    const valueRows: { actual: number; predicted: number }[][] = [];
    const errorRows: number[][] = [];

    for (let s = 0; s < keys.length; s++) {
      const row: { actual: number; predicted: number }[] = [];
      const errRow: number[] = [];
      for (let h = 0; h < hourCount; h++) {
        const avgA = actualRows[s]?.[h] ?? 0;
        const avgP = predictedRows[s]?.[h] ?? 0;
        const err = Math.abs(avgP - avgA);
        row.push({ actual: avgA, predicted: avgP });
        errRow.push(err);
        if (avgA < lo) lo = avgA;
        if (avgP < lo) lo = avgP;
        if (avgA > hi) hi = avgA;
        if (avgP > hi) hi = avgP;
        if (err > errHi) errHi = err;
      }
      valueRows.push(row);
      errorRows.push(errRow);
    }

    return {
      sensorKeys: keys,
      hours: Array.from({ length: hourCount }, (_, i) => i),
      minVal: lo,
      maxVal: hi,
      maxError: errHi,
      valueGrid: valueRows,
      errorGrid: errorRows,
    };
  }, [data]);

  const speedColor = (v: number): string => {
    const t = Math.max(0, Math.min(1, (v - minVal) / (maxVal - minVal || 1)));
    if (t < 0.25) {
      const s = t / 0.25;
      return `rgb(${Math.round(40 + 10 * s)}, ${Math.round(10 + 30 * s)}, ${Math.round(60 + 100 * s)})`;
    }
    if (t < 0.5) {
      const s = (t - 0.25) / 0.25;
      return `rgb(${Math.round(50 - 30 * s)}, ${Math.round(40 + 130 * s)}, ${Math.round(160 + 40 * s)})`;
    }
    if (t < 0.75) {
      const s = (t - 0.5) / 0.25;
      return `rgb(${Math.round(20 + 50 * s)}, ${Math.round(170 + 30 * s)}, ${Math.round(200 - 120 * s)})`;
    }
    const s = (t - 0.75) / 0.25;
    return `rgb(${Math.round(70 + 170 * s)}, ${Math.round(200 + 30 * s)}, ${Math.round(80 - 40 * s)})`;
  };

  const errorColor = (v: number): string => {
    const t = Math.max(0, Math.min(1, v / (maxError || 1)));
    if (t < 0.33) {
      const s = t / 0.33;
      return `rgb(${Math.round(15 + 20 * s)}, ${Math.round(25 + 60 * s)}, ${Math.round(80 + 100 * s)})`;
    }
    if (t < 0.66) {
      const s = (t - 0.33) / 0.33;
      return `rgb(${Math.round(35 + 190 * s)}, ${Math.round(85 + 100 * s)}, ${Math.round(180 - 80 * s)})`;
    }
    const s = (t - 0.66) / 0.34;
    return `rgb(${Math.round(225 + 14 * s)}, ${Math.round(185 - 120 * s)}, ${Math.round(100 - 60 * s)})`;
  };

  return (
    <div className="relative">
      <div className="overflow-x-auto">
        <div className="inline-flex flex-col gap-[2px] min-w-full">
          <div className="flex gap-[2px]">
            <div className="w-10 shrink-0" />
            {sensorKeys.map((s, i) => (
              <div key={s} className="text-[9px] text-muted-foreground text-center" style={{ width: 18 }}>
                {i % 5 === 0 ? `S${i + 1}` : ""}
              </div>
            ))}
          </div>

          {hours.map((h, r) => (
            <div key={h} className="flex gap-[2px]">
              <div className="w-10 shrink-0 text-[10px] text-muted-foreground font-medium flex items-center justify-end pr-1">
                {h}h
              </div>
              {sensorKeys.map((s, c) => {
                const cell = valueGrid[c]?.[r];
                const err = errorGrid[c]?.[r] ?? 0;
                const actual = cell?.actual ?? 0;
                const predicted = cell?.predicted ?? 0;
                const isHovered = hover?.sensorId === s && hover?.hour === h;
                return (
                  <div
                    key={`${s}-${h}`}
                    className="relative"
                    style={{ width: 18, height: 18 }}
                    onMouseEnter={() => setHover({ sensorId: s, sensorIndex: c + 1, hour: h, actual, predicted, error: err })}
                    onMouseLeave={() => setHover(null)}
                  >
                    {mode === "compare" ? (
                      <div className="flex w-full h-full overflow-hidden rounded-[2px]">
                        <div style={{ width: "50%", backgroundColor: speedColor(actual) }} />
                        <div style={{ width: "50%", backgroundColor: speedColor(predicted) }} />
                      </div>
                    ) : (
                      <div
                        className="w-full h-full rounded-[2px]"
                        style={{ backgroundColor: errorColor(err) }}
                      />
                    )}
                    {isHovered && (
                      <div className="absolute inset-0 ring-1 ring-white/90 rounded-[2px]" />
                    )}
                  </div>
                );
              })}
            </div>
          ))}
        </div>
      </div>

      {hover && (
        <div className="absolute top-2 right-2 rounded-md bg-black/90 text-white text-xs px-3 py-2 pointer-events-none backdrop-blur-sm z-20 space-y-0.5 shadow-lg">
          <div className="font-medium">Sensor {hover.sensorIndex} — {hover.hour}h</div>
          {mode === "compare" ? (
            <>
              <div className="text-white/70">Actual: <span className="text-white font-medium">{hover.actual.toFixed(2)} {data.unit}</span></div>
              <div className="text-white/70">Predicted: <span className="text-white">{hover.predicted.toFixed(2)} {data.unit}</span></div>
            </>
          ) : (
            <>
              <div className="text-white/70">Error: <span className="text-white font-medium">{hover.error.toFixed(2)} {data.unit}</span></div>
              <div className="text-white/70">Actual: <span className="text-white">{hover.actual.toFixed(2)} {data.unit}</span></div>
              <div className="text-white/70">Predicted: <span className="text-white">{hover.predicted.toFixed(2)} {data.unit}</span></div>
            </>
          )}
        </div>
      )}

      <div className="flex items-center gap-2 mt-3 text-[10px] text-muted-foreground">
        <span>{mode === "compare" ? "Slow" : "Low"}</span>
        <div className="flex gap-0 flex-1 max-w-[220px]">
          {Array.from({ length: 24 }, (_, i) => {
            const t = i / 23;
            const v = mode === "compare" ? minVal + t * (maxVal - minVal) : t * maxError;
            const color = mode === "compare" ? speedColor(v) : errorColor(v);
            return (
              <div key={i} className="flex-1" style={{ height: 7, backgroundColor: color }} />
            );
          })}
        </div>
        <span>{mode === "compare" ? "Fast" : "High"}</span>
      </div>
    </div>
  );
}