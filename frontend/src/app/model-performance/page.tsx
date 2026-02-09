"use client";

import { useEffect, useState, useRef, useCallback } from "react";
import Papa from "papaparse";
import * as Plot from "@observablehq/plot";
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";
import {
  Table, TableHeader, TableRow, TableHead, TableBody, TableCell,
} from "@/components/ui/table";

interface EpochRow {
  epoch: number;
  train_loss: number;
  val_loss: number;
  val_mae: number;
  val_rmse: number;
}

/* ---- Observable Plot wrapper ---- */
function PlotChart({ options, className }: { options: Plot.PlotOptions; className?: string }) {
  const ref = useRef<HTMLDivElement>(null);
  useEffect(() => {
    if (!ref.current) return;
    const plot = Plot.plot(options);
    ref.current.innerHTML = "";
    ref.current.appendChild(plot);
    return () => plot.remove();
  }, [options]);
  return <div ref={ref} className={className} />;
}

/* colors for dark mode */
const FG = "#e0e0e0";
const MUTED = "#888";
const GRID = "#333";
const TRAIN_C = "#c8c8c8";
const VAL_C = "#6b8de3";
const MAE_C = "#6b8de3";
const RMSE_C = "#e06060";

export default function ModelPerformancePage() {
  const [metrics, setMetrics] = useState<EpochRow[]>([]);
  const [r2Score, setR2Score] = useState<number | null>(null);

  useEffect(() => {
    fetch("/metrics.csv")
      .then((r) => r.text())
      .then((text) => {
        const result = Papa.parse(text, { header: true, dynamicTyping: true, skipEmptyLines: true });
        setMetrics(result.data as EpochRow[]);
      });
    // Compute R² from dashboard scatter data
    fetch("/dashboard-data.json")
      .then((r) => r.json())
      .then((d) => {
        if (d.scatter) {
          const n = d.scatter.length;
          const meanY = d.scatter.reduce((s: number, p: { predicted: number }) => s + p.predicted, 0) / n;
          const ssTot = d.scatter.reduce((s: number, p: { actual: number; predicted: number }) => s + (p.actual - meanY) ** 2, 0);
          const ssRes = d.scatter.reduce((s: number, p: { actual: number; predicted: number }) => s + (p.actual - p.predicted) ** 2, 0);
          setR2Score(1 - ssRes / ssTot);
        }
      });
  }, []);

  if (!metrics.length) return <div className="flex items-center justify-center h-96 text-muted-foreground">Loading...</div>;

  const lastEpoch = metrics[metrics.length - 1];
  const firstEpoch = metrics[0];

  /* ---- prepare tidy data for Plot ---- */
  const lossData = metrics.flatMap((m) => [
    { epoch: m.epoch, loss: m.train_loss, series: "Training" },
    { epoch: m.epoch, loss: m.val_loss, series: "Validation" },
  ]);

  const accData = metrics.flatMap((m) => [
    { epoch: m.epoch, value: m.val_mae, metric: "MAE" },
    { epoch: m.epoch, value: m.val_rmse, metric: "RMSE" },
  ]);

  const improvData = metrics.flatMap((m) => [
    { epoch: m.epoch, pct: ((firstEpoch.val_mae - m.val_mae) / firstEpoch.val_mae) * 100, metric: "MAE" },
    { epoch: m.epoch, pct: ((firstEpoch.val_rmse - m.val_rmse) / firstEpoch.val_rmse) * 100, metric: "RMSE" },
  ]);

  // Compute Y domain for loss (tight zoom)
  const allLoss = metrics.flatMap(m => [m.train_loss, m.val_loss]);
  const lossMin = Math.min(...allLoss);
  const lossMax = Math.max(...allLoss);
  const lossPad = (lossMax - lossMin) * 0.08;
  const lossDomain: [number, number] = [Math.max(0, lossMin - lossPad), lossMax + lossPad];

  // Compute Y domain for accuracy metrics with padding
  const allAcc = metrics.flatMap(m => [m.val_mae, m.val_rmse]);
  const accMin = Math.min(...allAcc);
  const accMax = Math.max(...allAcc);
  const accPad = (accMax - accMin) * 0.12;
  const accDomain: [number, number] = [Math.max(0, accMin - accPad), accMax + accPad];

  const plotBase = {
    style: { background: "transparent", color: FG, fontSize: "12px" },
  } as const;

  return (
    <div className="space-y-6">
      <div>
        <h2 className="text-2xl font-bold tracking-tight text-foreground">Model Performance</h2>
        <p className="text-sm text-muted-foreground mt-1">
          Training diagnostics and convergence analysis across {metrics.length} epochs
        </p>
      </div>

      {/* Summary Stats */}
      <div className="grid grid-cols-2 md:grid-cols-5 gap-4">
        <SummaryCard label="Final Train Loss" value={lastEpoch.train_loss.toFixed(4)} />
        <SummaryCard label="Final Val Loss" value={lastEpoch.val_loss.toFixed(4)} />
        <SummaryCard label="Final Val MAE" value={lastEpoch.val_mae.toFixed(4)} />
        <SummaryCard label="Final Val RMSE" value={lastEpoch.val_rmse.toFixed(4)} />
        <SummaryCard label="R² Score" value={r2Score !== null ? r2Score.toFixed(4) : "—"} />
      </div>

      {/* Training & Validation Loss */}
      <div className="grid md:grid-cols-2 gap-6">
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-base">Training &amp; Validation Loss</CardTitle>
            <CardDescription>Convergence curve — large initial drop then steady refinement</CardDescription>
          </CardHeader>
          <CardContent>
            <PlotChart options={{
              ...plotBase,
              width: 580,
              height: 380,
              marginLeft: 55,
              marginBottom: 40,
              x: { label: "Epoch", tickFormat: "d" },
              y: { label: "MSE Loss", grid: true, domain: lossDomain, tickFormat: ".4f" },
              color: { domain: ["Training", "Validation"], range: [TRAIN_C, VAL_C], legend: true },
              marks: [
                Plot.line(lossData, { x: "epoch", y: "loss", stroke: "series", strokeWidth: 2.5 }),
                Plot.dot(lossData, { x: "epoch", y: "loss", stroke: "series", r: 3 }),
                Plot.tip(lossData, Plot.pointer({ x: "epoch", y: "loss", stroke: "series",
                  format: { y: (d: number) => d.toFixed(4), x: (d: number) => `Epoch ${d}` }
                })),
              ],
            }} />
          </CardContent>
        </Card>

        {/* Validation MAE & RMSE */}
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-base">Validation MAE &amp; RMSE</CardTitle>
            <CardDescription>Error metrics per epoch</CardDescription>
          </CardHeader>
          <CardContent>
            <PlotChart options={{
              ...plotBase,
              width: 580,
              height: 380,
              marginLeft: 55,
              marginBottom: 40,
              x: { label: "Epoch", tickFormat: "d" },
              y: { label: "Error", grid: true, tickFormat: ".3f", ticks: 12, domain: accDomain },
              color: { domain: ["MAE", "RMSE"], range: [MAE_C, RMSE_C], legend: true },
              marks: [
                Plot.line(accData, { x: "epoch", y: "value", stroke: "metric", strokeWidth: 2.5 }),
                Plot.dot(accData, { x: "epoch", y: "value", stroke: "metric", r: 3 }),
                Plot.tip(accData, Plot.pointer({ x: "epoch", y: "value", stroke: "metric",
                  format: { y: (d: number) => d.toFixed(4), x: (d: number) => `Epoch ${d}` }
                })),
              ],
            }} />
          </CardContent>
        </Card>
      </div>

      {/* Improvement Chart */}
      <Card>
        <CardHeader className="pb-2">
          <CardTitle className="text-base">Improvement from Epoch 1 (%)</CardTitle>
          <CardDescription>Cumulative reduction in MAE &amp; RMSE relative to the first epoch</CardDescription>
        </CardHeader>
        <CardContent>
          <PlotChart options={{
            ...plotBase,
            width: 1180,
            height: 300,
            marginLeft: 55,
            marginBottom: 40,
            x: { label: "Epoch", tickFormat: "d" },
            y: { label: "Improvement (%)", grid: true, tickFormat: ".1f" },
            color: { domain: ["MAE", "RMSE"], range: [MAE_C, RMSE_C], legend: true },
            marks: [
              Plot.ruleY([0], { stroke: MUTED, strokeDasharray: "4 2" }),
              Plot.line(improvData, { x: "epoch", y: "pct", stroke: "metric", strokeWidth: 2 }),
              Plot.tip(improvData, Plot.pointer({ x: "epoch", y: "pct", stroke: "metric",
                format: { y: (d: number) => `${d.toFixed(1)}%`, x: (d: number) => `Epoch ${d}` }
              })),
            ],
          }} />
        </CardContent>
      </Card>

      {/* Epoch Table */}
      <Card>
        <CardHeader className="pb-2">
          <CardTitle className="text-base">Epoch-by-Epoch Results</CardTitle>
          <CardDescription>Complete training log</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="max-h-[420px] overflow-y-auto">
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead>Epoch</TableHead>
                  <TableHead className="text-right">Train Loss</TableHead>
                  <TableHead className="text-right">Val Loss</TableHead>
                  <TableHead className="text-right">Val MAE</TableHead>
                  <TableHead className="text-right">Val RMSE</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {metrics.map((m) => (
                  <TableRow key={m.epoch}>
                    <TableCell>{m.epoch}</TableCell>
                    <TableCell className="text-right font-mono text-sm">{m.train_loss.toFixed(4)}</TableCell>
                    <TableCell className="text-right font-mono text-sm">{m.val_loss.toFixed(4)}</TableCell>
                    <TableCell className="text-right font-mono text-sm">{m.val_mae.toFixed(4)}</TableCell>
                    <TableCell className="text-right font-mono text-sm">{m.val_rmse.toFixed(4)}</TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </div>
        </CardContent>
      </Card>

      {/* Training Notes */}
      <Card>
        <CardContent className="p-5">
          <p className="text-sm text-muted-foreground leading-relaxed">
            <strong className="text-foreground">Training configuration.</strong> SimpleSTGCN trained for {metrics.length} epochs
            using Adam optimizer (lr=0.001) with gradient clipping (max_norm=5.0). The loss function is MSE on
            z-score normalized speed values. An 80/20 temporal split ensures no data leakage. The sharp drop in
            training loss during the first 2 epochs reflects rapid learning of dominant traffic patterns, followed
            by gradual convergence on finer spatio-temporal dynamics.
          </p>
        </CardContent>
      </Card>
    </div>
  );
}

function SummaryCard({ label, value }: { label: string; value: string }) {
  return (
    <Card>
      <CardContent className="p-4 text-center">
        <p className="text-[11px] text-muted-foreground uppercase tracking-wide">{label}</p>
        <p className="text-xl font-bold text-foreground mt-1 font-mono">{value}</p>
      </CardContent>
    </Card>
  );
}
