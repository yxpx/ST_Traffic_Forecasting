"use client";

import { useEffect, useRef, useState } from "react";
import dynamic from "next/dynamic";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";

const MiniMap = dynamic(() => import("@/components/mini-map"), { ssr: false });

const API_BASE = process.env.NEXT_PUBLIC_API_BASE || "http://localhost:8000";

/* ── types ─────────────────────────────────────────────────────────── */

interface LiveStats {
  frames: number;
  elapsed: number;
  detections: number;
  avg: number;
  max: number;
  congestion: string;
  congestion_pct: number;
}

interface SensorCtx {
  sensor_id: string;
  sensor_idx: number;
  predicted_speed_mph: number;
  historical_avg_mph: number;
  network_avg_mph: number;
  congestion: string;
}

interface Sensor {
  id: number;
  lat: number;
  lng: number;
  avg_speed: number;
  congestion: number;
}

interface Weather {
  temp_f: number;
  humidity: number;
  visibility_mi: number;
  wind_mph: number;
}

/* ── page ──────────────────────────────────────────────────────────── */

export default function CctvAnalyticsPage() {
  /* detection state */
  const [source, setSource] = useState<"video" | "camera">("video");
  const [videoPath, setVideoPath] = useState("D:/OnTime/temp_video.mp4");
  const [maxSeconds, setMaxSeconds] = useState("30");
  const [status, setStatus] = useState("");
  const [isRunning, setIsRunning] = useState(false);
  const [liveFrame, setLiveFrame] = useState<string | null>(null);
  const [liveStats, setLiveStats] = useState<LiveStats | null>(null);
  const [summary, setSummary] = useState<{ frames: number; seconds: number; avg: number; max: number } | null>(null);
  const abortRef = useRef<AbortController | null>(null);

  /* sensor linkage */
  const [sensorIdx, setSensorIdx] = useState<number>(3);
  const [sensorCtx, setSensorCtx] = useState<SensorCtx | null>(null);
  const [sensors, setSensors] = useState<Sensor[]>([]);
  const [weather, setWeather] = useState<Weather | null>(null);

  /* load sensor locations + weather on mount */
  useEffect(() => {
    fetch("/sensor-locations.json")
      .then((r) => r.json())
      .then((d: { sensors: Sensor[] }) => setSensors(d.sensors))
      .catch(() => {});
    fetch(`${API_BASE}/dashboard/context`)
      .then((r) => (r.ok ? r.json() : null))
      .then((d) => { if (d?.weather) setWeather(d.weather); })
      .catch(() => {});
  }, []);

  /* fetch sensor model context when sensorIdx changes */
  useEffect(() => {
    if (sensorIdx < 0) return;
    fetch(`${API_BASE}/sensor/${sensorIdx}`)
      .then((r) => (r.ok ? r.json() : null))
      .then((d) => { if (d) setSensorCtx(d); })
      .catch(() => {});
  }, [sensorIdx]);

  /* push CCTV state so main dashboard can read it */
  useEffect(() => {
    if (!liveStats) return;
    fetch(`${API_BASE}/cv/state`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        active: isRunning,
        sensor_idx: sensorIdx,
        congestion: liveStats.congestion,
        congestion_pct: liveStats.congestion_pct,
        vehicles: liveStats.detections,
      }),
    }).catch(() => {});
  }, [liveStats, isRunning, sensorIdx]);

  /* push inactive on stop */
  useEffect(() => {
    if (!isRunning && summary) {
      fetch(`${API_BASE}/cv/state`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ active: false, sensor_idx: sensorIdx }),
      }).catch(() => {});
    }
  }, [isRunning, summary, sensorIdx]);

  /* ── stream handler ──────────────────────────────────────────────── */

  const runDetection = async () => {
    setIsRunning(true);
    setStatus("Connecting...");
    setSummary(null);
    setLiveFrame(null);
    setLiveStats(null);

    const controller = new AbortController();
    abortRef.current = controller;

    const params = new URLSearchParams({
      source,
      max_seconds: String(Math.min(Math.max(Number(maxSeconds) || 30, 5), 60)),
      frame_stride: "2",
    });
    if (source === "video") params.set("path", videoPath);

    try {
      const res = await fetch(`${API_BASE}/cv/stream?${params}`, { signal: controller.signal });
      if (!res.ok || !res.body) throw new Error(`Server error ${res.status}`);

      setStatus("Streaming...");
      const reader = res.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const parts = buffer.split("\n\n");
        buffer = parts.pop() || "";

        for (const part of parts) {
          const line = part.trim();
          if (!line.startsWith("data: ")) continue;
          const data = JSON.parse(line.slice(6));

          if (data.error) { setStatus(`Error: ${data.error}`); setIsRunning(false); return; }

          if (data.done) {
            setSummary({ frames: data.frames, seconds: data.seconds, avg: data.avg, max: data.max });
            setStatus(`Completed — ${data.frames} frames in ${data.seconds}s`);
            setIsRunning(false);
            return;
          }

          setLiveFrame(`data:image/jpeg;base64,${data.frame}`);
          setLiveStats({
            frames: data.n,
            elapsed: data.elapsed,
            detections: data.det,
            avg: data.avg,
            max: data.max,
            congestion: data.congestion || "—",
            congestion_pct: data.congestion_pct ?? 0,
          });
        }
      }
    } catch (err: unknown) {
      if (err instanceof DOMException && err.name === "AbortError") setStatus("Stopped");
      else setStatus(err instanceof Error ? err.message : "Stream failed");
    } finally {
      setIsRunning(false);
      abortRef.current = null;
    }
  };

  const stopDetection = () => abortRef.current?.abort();

  /* congestion colour helper */
  const congColor = (level?: string) =>
    level === "Severe" ? "text-red-500" :
    level === "Moderate" ? "text-orange-500" :
    level === "Light" ? "text-yellow-500" :
    level === "Free Flow" ? "text-green-500" :
    "text-muted-foreground";

  const barColor = (pct: number) =>
    pct >= 70 ? "bg-red-500" : pct >= 45 ? "bg-orange-500" : pct >= 20 ? "bg-yellow-500" : "bg-green-500";

  /* ── render ──────────────────────────────────────────────────────── */

  return (
    <div className="space-y-6">
      {/* Header */}
      <div>
        <h2 className="text-2xl font-bold tracking-tight text-foreground">CCTV Analytics</h2>
        <p className="text-sm text-muted-foreground mt-1">
          Real-time vehicle detection linked to the ST-GCN traffic prediction model
        </p>
      </div>

      {/* Main grid: feed | sidebar */}
      <div className="grid lg:grid-cols-[5fr_2fr] gap-6">

        {/* ── Left: Detection feed ────────────────────────────────── */}
        <Card className="overflow-hidden">
          <CardHeader className="pb-2">
            <CardTitle className="text-base">Detection Feed</CardTitle>
            <CardDescription>Frames streamed live via YOLOv8n tracker — vehicles only</CardDescription>
          </CardHeader>
          <CardContent className="p-4 space-y-4">
            {/* Controls row */}
            <div className="flex flex-wrap items-center gap-2">
              <div className="inline-flex rounded-md border border-border overflow-hidden">
                <button
                  className={`h-9 px-4 text-sm transition-colors ${source === "video" ? "bg-accent text-accent-foreground" : "text-muted-foreground hover:text-foreground"}`}
                  onClick={() => setSource("video")} disabled={isRunning}
                >Video</button>
                <button
                  className={`h-9 px-4 text-sm transition-colors ${source === "camera" ? "bg-accent text-accent-foreground" : "text-muted-foreground hover:text-foreground"}`}
                  onClick={() => setSource("camera")} disabled={isRunning}
                >Camera</button>
              </div>

              {!isRunning ? (
                <button className="h-9 rounded-md bg-primary text-primary-foreground text-sm font-medium px-4 hover:opacity-90" onClick={runDetection}>
                  Run Detection
                </button>
              ) : (
                <button className="h-9 rounded-md bg-destructive text-destructive-foreground text-sm font-medium px-4 hover:opacity-90" onClick={stopDetection}>
                  Stop
                </button>
              )}

              {/* Sensor picker */}
              <div className="flex items-center gap-1.5 ml-auto">
                <span className="text-xs text-muted-foreground">Linked sensor</span>
                <input
                  type="number" min={1} max={207}
                  value={sensorIdx + 1}
                  onChange={(e) => setSensorIdx(Math.min(206, Math.max(0, (Number(e.target.value) || 1) - 1)))}
                  disabled={isRunning}
                  className="h-9 w-20 rounded-md border border-border bg-card text-foreground text-sm px-2 text-center disabled:opacity-50"
                />
              </div>
            </div>

            {source === "video" && (
              <div className="grid gap-1.5">
                <label className="text-xs text-muted-foreground">Video path (server-side)</label>
                <input type="text" value={videoPath} onChange={(e) => setVideoPath(e.target.value)} disabled={isRunning}
                  className="h-9 rounded-md border border-border bg-card text-foreground text-sm px-3 disabled:opacity-50" />
              </div>
            )}

            <div className="grid gap-1.5">
              <label className="text-xs text-muted-foreground">Max seconds (5–60)</label>
              <input type="number" min={5} max={60} value={maxSeconds} onChange={(e) => setMaxSeconds(e.target.value)} disabled={isRunning}
                className="h-9 w-28 rounded-md border border-border bg-card text-foreground text-sm px-3 disabled:opacity-50" />
            </div>

            {status && <p className={`text-xs ${status.startsWith("Error") ? "text-destructive" : "text-muted-foreground"}`}>{status}</p>}

            {/* Live feed */}
            <div className="relative w-full bg-black rounded-md overflow-hidden" style={{ minHeight: 320 }}>
              {liveFrame ? (
                // eslint-disable-next-line @next/next/no-img-element
                <img src={liveFrame} alt="Detection" className="w-full h-auto" />
              ) : (
                <div className="flex items-center justify-center h-[320px] text-muted-foreground text-sm">
                  {isRunning ? "Waiting for first frame..." : "Click Run Detection to start"}
                </div>
              )}

              {/* Overlay HUD */}
              {isRunning && liveStats && (
                <div className="absolute top-2 left-2 bg-black/70 text-white text-xs px-2 py-1 rounded font-mono space-x-2">
                  <span>F{liveStats.frames}</span>
                  <span>&middot;</span>
                  <span>{liveStats.detections} vehicles</span>
                  <span>&middot;</span>
                  <span className={congColor(liveStats.congestion)}>{liveStats.congestion}</span>
                  <span>&middot;</span>
                  <span>{liveStats.elapsed}s</span>
                </div>
              )}
            </div>
          </CardContent>
        </Card>

        {/* ── Right: Sidebar ──────────────────────────────────────── */}
        <div className="space-y-4">

          {/* Video congestion (live from detection) */}
          <Card>
            <CardHeader className="pb-2">
              <CardTitle className="text-base">Video Congestion</CardTitle>
              <CardDescription>Derived from vehicle density &amp; movement</CardDescription>
            </CardHeader>
            <CardContent className="space-y-3 text-sm">
              <div className="flex items-center justify-between">
                <span className="text-muted-foreground">Level</span>
                <span className={`font-semibold ${congColor(liveStats?.congestion)}`}>{liveStats?.congestion ?? "—"}</span>
              </div>
              <div className="w-full h-2 bg-muted rounded-full overflow-hidden">
                <div className={`h-full rounded-full transition-all duration-500 ${barColor(liveStats?.congestion_pct ?? 0)}`}
                  style={{ width: `${liveStats?.congestion_pct ?? 0}%` }} />
              </div>
              <Row label="Vehicles in frame" value={liveStats ? String(liveStats.detections) : "—"} />
              <Row label="Avg vehicles" value={liveStats ? liveStats.avg.toFixed(1) : "—"} />
              <Row label="Peak vehicles" value={liveStats ? String(liveStats.max) : "—"} />
            </CardContent>
          </Card>

          {/* Model predictions for linked sensor */}
          <Card>
            <CardHeader className="pb-2">
              <CardTitle className="text-base">Model Prediction</CardTitle>
              <CardDescription>ST-GCN forecast — Sensor {sensorIdx + 1}{sensorCtx ? ` (${sensorCtx.sensor_id})` : ""}</CardDescription>
            </CardHeader>
            <CardContent className="space-y-3 text-sm">
              <Row label="Predicted speed" value={sensorCtx ? `${sensorCtx.predicted_speed_mph} mph` : "—"} />
              <Row label="Historical avg" value={sensorCtx ? `${sensorCtx.historical_avg_mph} mph` : "—"} />
              <Row label="Network avg" value={sensorCtx ? `${sensorCtx.network_avg_mph} mph` : "—"} />
              <div className="flex items-center justify-between">
                <span className="text-muted-foreground">Model congestion</span>
                <span className={`font-semibold ${congColor(sensorCtx?.congestion)}`}>{sensorCtx?.congestion ?? "—"}</span>
              </div>
              {/* Comparison badge */}
              {liveStats && sensorCtx && (
                <div className="pt-1 border-t border-border">
                  <p className="text-[11px] text-muted-foreground">
                    {liveStats.congestion === sensorCtx.congestion
                      ? "Video and model agree on congestion level"
                      : `Video: ${liveStats.congestion} vs Model: ${sensorCtx.congestion}`}
                  </p>
                </div>
              )}
            </CardContent>
          </Card>

          {/* Weather */}
          <Card>
            <CardHeader className="pb-2">
              <CardTitle className="text-base">Weather</CardTitle>
              <CardDescription>Los Angeles</CardDescription>
            </CardHeader>
            <CardContent className="space-y-3 text-sm">
              <Row label="Temperature" value={weather ? `${weather.temp_f}°F` : "—"} />
              <Row label="Humidity" value={weather ? `${weather.humidity}%` : "—"} />
              <Row label="Visibility" value={weather ? `${weather.visibility_mi} mi` : "—"} />
              <Row label="Wind" value={weather ? `${weather.wind_mph} mph` : "—"} />
            </CardContent>
          </Card>

          {/* Mini map */}
          {sensors.length > 0 && (
            <Card>
              <CardHeader className="pb-2">
                <CardTitle className="text-base">Sensor Location</CardTitle>
                <CardDescription>Linked sensor on the METR-LA network</CardDescription>
              </CardHeader>
              <CardContent className="p-0 overflow-hidden rounded-b-lg">
                <MiniMap sensors={sensors} highlightIdx={sensorIdx} height="180px" />
              </CardContent>
            </Card>
          )}
        </div>
      </div>

      {/* ── Bottom metrics ─────────────────────────────────────────── */}
      <div className="grid grid-cols-2 md:grid-cols-6 gap-3">
        <Metric label="Congestion" value={liveStats?.congestion ?? "-"} />
        <Metric label="Vehicles" value={liveStats ? String(liveStats.detections) : "-"} />
        <Metric label="Avg Vehicles" value={liveStats ? liveStats.avg.toFixed(1) : summary ? summary.avg.toFixed(1) : "-"} />
        <Metric label="Pred. Speed" value={sensorCtx ? `${sensorCtx.predicted_speed_mph} mph` : "-"} />
        <Metric label="Network Avg" value={sensorCtx ? `${sensorCtx.network_avg_mph} mph` : "-"} />
        <Metric label="Frames" value={liveStats ? String(liveStats.frames) : summary ? String(summary.frames) : "-"} />
      </div>
    </div>
  );
}

/* ── small components ──────────────────────────────────────────────── */

function Row({ label, value }: { label: string; value: string }) {
  return (
    <div className="flex items-center justify-between">
      <span className="text-muted-foreground">{label}</span>
      <span className="text-foreground font-medium">{value}</span>
    </div>
  );
}

function Metric({ label, value }: { label: string; value: string }) {
  return (
    <Card>
      <CardContent className="p-4">
        <p className="text-[11px] text-muted-foreground uppercase tracking-wide">{label}</p>
        <p className="text-lg font-semibold text-foreground mt-1">{value}</p>
      </CardContent>
    </Card>
  );
}
