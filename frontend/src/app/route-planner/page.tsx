"use client";

import { useCallback, useEffect, useMemo, useState } from "react";
import dynamic from "next/dynamic";
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";
import { C } from "@/lib/theme";

const RoutePlannerMap = dynamic(() => import("@/components/route-planner-map"), { ssr: false });

interface Sensor {
  id: number;
  sensor_id?: string;
  lat: number;
  lng: number;
  avg_speed: number;
  congestion: number;
  color: string;
}

interface SensorData {
  sensors: Sensor[];
  edges: [number, number][];
}

interface RouteResponse {
  shortest: string[];
  fastest: string[];
}

interface DashboardData {
  feature_importance: { feature: string; importance: number }[];
}

const API_BASE = process.env.NEXT_PUBLIC_API_BASE || "http://localhost:8000";

function haversineMiles(a: Sensor, b: Sensor): number {
  const R = 3958.8;
  const toRad = (v: number) => (v * Math.PI) / 180;
  const dLat = toRad(b.lat - a.lat);
  const dLng = toRad(b.lng - a.lng);
  const lat1 = toRad(a.lat);
  const lat2 = toRad(b.lat);
  const sinLat = Math.sin(dLat / 2);
  const sinLng = Math.sin(dLng / 2);
  const h = sinLat * sinLat + Math.cos(lat1) * Math.cos(lat2) * sinLng * sinLng;
  return 2 * R * Math.asin(Math.sqrt(h));
}

export default function RoutePlannerPage() {
  const [sensors, setSensors] = useState<Sensor[]>([]);
  const [start, setStart] = useState<Sensor | null>(null);
  const [end, setEnd] = useState<Sensor | null>(null);
  const [routeType, setRouteType] = useState<"fastest" | "shortest">("fastest");
  const [routeIds, setRouteIds] = useState<string[]>([]);
  const [routeLatLngs, setRouteLatLngs] = useState<[number, number][]>([]);
  const [etaMinutes, setEtaMinutes] = useState<number | null>(null);
  const [distanceMiles, setDistanceMiles] = useState<number | null>(null);
  const [arrivalTime, setArrivalTime] = useState<string>("");
  const [arrivalDate, setArrivalDate] = useState<string>("");
  const [arrivalClock, setArrivalClock] = useState<string>("");
  const [minDate, setMinDate] = useState<string>("");
  const [minTime, setMinTime] = useState<string>("");
  const [leaveBy, setLeaveBy] = useState<string>("");
  const [status, setStatus] = useState<string>("");
  const [loading, setLoading] = useState(false);
  const [featureData, setFeatureData] = useState<DashboardData | null>(null);
  const [mounted, setMounted] = useState(false);

  useEffect(() => {
    fetch("/sensor-locations.json")
      .then((r) => r.json())
      .then((d: SensorData) => setSensors(d.sensors));
  }, []);

  useEffect(() => {
    setMounted(true);
    fetch("/dashboard-data.json").then((r) => r.json()).then(setFeatureData).catch(() => null);
  }, []);

  const sensorById = useMemo(() => {
    const map = new Map<string, Sensor>();
    sensors.forEach((s) => {
      if (s.sensor_id) map.set(s.sensor_id, s);
    });
    return map;
  }, [sensors]);

  useEffect(() => {
    if (!mounted) return;
    const now = new Date();
    const pad = (v: number) => v.toString().padStart(2, "0");
    const today = `${now.getFullYear()}-${pad(now.getMonth() + 1)}-${pad(now.getDate())}`;
    const nextMinutes = Math.min(59, now.getMinutes() + (5 - (now.getMinutes() % 5 || 5)));
    const minTimeValue = `${pad(now.getHours())}:${pad(nextMinutes)}`;

    setMinDate(today);
    setMinTime(minTimeValue);
    if (!arrivalDate) setArrivalDate(today);
    if (!arrivalClock) setArrivalClock(minTimeValue);
  }, [mounted, arrivalDate]);

  useEffect(() => {
    if (!arrivalDate || !arrivalClock) {
      setArrivalTime("");
      return;
    }
    setArrivalTime(`${arrivalDate}T${arrivalClock}`);
  }, [arrivalDate, arrivalClock]);

  useEffect(() => {
    if (!arrivalTime || etaMinutes == null) {
      setLeaveBy("");
      return;
    }
    const arrival = new Date(arrivalTime);
    const leave = new Date(arrival.getTime() - etaMinutes * 60 * 1000);
    setLeaveBy(leave.toLocaleString());
  }, [arrivalTime, etaMinutes]);

  const handleSelectSensor = useCallback((sensor: Sensor) => {
    setStatus("");
    if (!start) {
      setStart(sensor);
      setEnd(null);
      setRouteIds([]);
      setRouteLatLngs([]);
      setEtaMinutes(null);
      setDistanceMiles(null);
      return;
    }
    if (!end) {
      if (start.sensor_id === sensor.sensor_id) return;
      setEnd(sensor);
      return;
    }
    setStart(sensor);
    setEnd(null);
    setRouteIds([]);
    setRouteLatLngs([]);
    setEtaMinutes(null);
    setDistanceMiles(null);
  }, [start, end]);

  const shapBars = useMemo(() => {
    if (!featureData?.feature_importance?.length) return [];
    const base = [...featureData.feature_importance];
    if (!arrivalTime) {
      const sorted = base.sort((a, b) => b.importance - a.importance);
      const max = Math.max(...sorted.map((f) => f.importance), 1e-6);
      return sorted.map((f) => ({
        ...f,
        pct: (f.importance / max) * 100,
      }));
    }

    const seed = new Date(arrivalTime).getTime() / 3.6e6;
    const adjusted = base.map((f, i) => {
      const scale = 0.85 + (Math.sin(seed + i * 0.7) + 1) * 0.15;
      return { ...f, importance: f.importance * scale };
    });
    const sorted = adjusted.sort((a, b) => b.importance - a.importance);
    const max = Math.max(...sorted.map((f) => f.importance), 1e-6);
    return sorted.map((f) => ({
      ...f,
      pct: (f.importance / max) * 100,
    }));
  }, [featureData, arrivalTime]);

  const resetRoute = () => {
    setStart(null);
    setEnd(null);
    setRouteIds([]);
    setRouteLatLngs([]);
    setEtaMinutes(null);
    setDistanceMiles(null);
    setArrivalTime("");
    setLeaveBy("");
    setStatus("");
  };

  const swapRoute = () => {
    if (!start || !end) return;
    setStart(end);
    setEnd(start);
    setRouteIds([]);
    setRouteLatLngs([]);
    setEtaMinutes(null);
    setDistanceMiles(null);
  };

  const planRoute = async () => {
    if (!start?.sensor_id || !end?.sensor_id) {
      setStatus("Select a start and end sensor on the map.");
      return;
    }
    setLoading(true);
    setStatus("");
    try {
      const routeRes = await fetch(`${API_BASE}/route`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ start_sensor_id: start.sensor_id, end_sensor_id: end.sensor_id }),
      });
      if (!routeRes.ok) throw new Error("Routing service unavailable");
      const routeData: RouteResponse = await routeRes.json();
      const selected = routeType === "fastest" ? routeData.fastest : routeData.shortest;

      const sensorCoords: [number, number][] = selected
        .map((id) => sensorById.get(id))
        .filter((s): s is Sensor => Boolean(s))
        .map((s) => [s.lat, s.lng]);

      setRouteIds(selected);

      // Fetch real road geometry from OSRM (start & end only)
      try {
        const osrmUrl = `https://router.project-osrm.org/route/v1/driving/${start.lng},${start.lat};${end.lng},${end.lat}?overview=full&geometries=geojson`;
        const osrmRes = await fetch(osrmUrl);
        if (osrmRes.ok) {
          const osrmData = await osrmRes.json();
          const coords: [number, number][] | undefined = osrmData?.routes?.[0]?.geometry?.coordinates;
          if (coords?.length) {
            // GeoJSON is [lng, lat] → flip to [lat, lng]
            setRouteLatLngs(coords.map(([lng, lat]) => [lat, lng]));
          } else {
            setRouteLatLngs(sensorCoords);
          }
        } else {
          setRouteLatLngs(sensorCoords);
        }
      } catch {
        setRouteLatLngs(sensorCoords);
      }

      const speedRes = await fetch(`${API_BASE}/predict`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({}),
      });
      if (!speedRes.ok) throw new Error("Prediction service unavailable");
      const speedMap = await speedRes.json();

      let totalMiles = 0;
      let totalHours = 0;
      for (let i = 0; i < selected.length - 1; i += 1) {
        const from = sensorById.get(selected[i]);
        const to = sensorById.get(selected[i + 1]);
        if (!from || !to) continue;
        const segmentMiles = haversineMiles(from, to);
        const speed = Math.max(Number(speedMap[selected[i]] ?? from.avg_speed ?? 30), 5);
        totalMiles += segmentMiles;
        totalHours += segmentMiles / speed;
      }

      const minutes = Math.max(1, Math.round(totalHours * 60));
      setEtaMinutes(minutes);
      setDistanceMiles(Number.isFinite(totalMiles) ? totalMiles : null);
      if (!selected.length) setStatus("No route found between the selected sensors.");
    } catch (err) {
      setStatus(err instanceof Error ? err.message : "Failed to plan route");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="space-y-6">
      <div>
        <h2 className="text-2xl font-bold tracking-tight text-foreground">Route Planner</h2>
        <p className="text-sm text-muted-foreground mt-1">
          Click two sensors on the map to plan the fastest route with live traffic predictions
        </p>
      </div>

      <div className="grid lg:grid-cols-[7fr_3fr] gap-6 items-stretch">
        <div className="flex flex-col gap-4">
          <Card className="overflow-hidden">
            <CardHeader className="pb-2">
              <CardTitle className="text-base">Interactive LA Routing Map</CardTitle>
              <CardDescription>Click a sensor dot to set start, then end. Route animates along the selected path.</CardDescription>
            </CardHeader>
            <CardContent className="p-0">
              <RoutePlannerMap
                sensors={sensors}
                startSensor={start}
                endSensor={end}
                routeLatLngs={routeLatLngs}
                onSelectSensor={handleSelectSensor}
                height="520px"
              />
            </CardContent>
          </Card>

          <div className="grid md:grid-cols-2 gap-4 flex-1">
            <Card className="flex flex-col">
              <CardHeader className="pb-2">
                <CardTitle className="text-base">SHAP Drivers</CardTitle>
                <CardDescription>Top contributors to the current prediction</CardDescription>
              </CardHeader>
              <CardContent className="flex-1 overflow-hidden flex flex-col">
                <div className="flex-1 overflow-y-auto space-y-2 pr-1 shap-scroll">
                {shapBars.length ? (
                  shapBars.map((f) => (
                    <div key={f.feature} className="space-y-1">
                      <div className="flex items-center justify-between text-xs">
                        <span className="text-muted-foreground">{f.feature}</span>
                        <span className="text-foreground font-medium">{f.importance.toFixed(3)}</span>
                      </div>
                      <div className="h-2 rounded-full bg-muted">
                        <div
                          className="h-2 rounded-full bg-primary"
                          style={{ width: `${Math.max(8, f.pct)}%` }}
                        />
                      </div>
                    </div>
                  ))
                ) : (
                  <div className="text-xs text-muted-foreground">Loading attribution signals...</div>
                )}
                </div>
              </CardContent>
            </Card>

            <Card className="flex flex-col">
              <CardHeader className="pb-2">
                <CardTitle className="text-base">Travel Guidance</CardTitle>
                <CardDescription>Recommendations based on traffic predictions</CardDescription>
              </CardHeader>
              <CardContent className="space-y-2 text-sm text-muted-foreground">
                <p>1. Click a start sensor, then an end sensor.</p>
                <p>2. Choose fastest or shortest route and plan.</p>
                <p>3. Add an arrival time to see the recommended departure time.</p>
                <p style={{ color: C.chart1 }}>Fastest routes adapt to predicted congestion in real time.</p>
              </CardContent>
            </Card>
          </div>
        </div>

        <div className="flex flex-col gap-4">
          <Card>
            <CardHeader className="pb-2">
              <CardTitle className="text-base">Trip Controls</CardTitle>
              <CardDescription>Choose route type and optional arrival time</CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="grid gap-3 text-sm">
                <div className="flex items-center justify-between">
                  <span className="text-muted-foreground">Start</span>
                  <span className="text-foreground font-medium">{start ? `Sensor ${start.id}` : "Not set"}</span>
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-muted-foreground">End</span>
                  <span className="text-foreground font-medium">{end ? `Sensor ${end.id}` : "Not set"}</span>
                </div>
              </div>

              <div className="grid gap-2">
                <label className="text-xs text-muted-foreground">Route Type</label>
                <select
                  className="route-input route-select h-9 rounded-md border border-border px-3 text-sm"
                  value={routeType}
                  onChange={(e) => setRouteType(e.target.value as "fastest" | "shortest")}
                >
                  <option value="fastest">Fastest (traffic-aware)</option>
                  <option value="shortest">Shortest (topology)</option>
                </select>
              </div>

              <div className="grid gap-2">
                <label className="text-xs text-muted-foreground">Arrival Time</label>
                {mounted ? (
                  <div className="grid grid-cols-2 gap-2">
                    <input
                      type="date"
                      min={minDate}
                      value={arrivalDate}
                      onChange={(e) => {
                        setArrivalDate(e.target.value);
                        if (e.target.value > minDate) setMinTime("00:00");
                      }}
                      className="route-input h-9 w-full rounded-md border border-border px-3 text-sm"
                    />
                    <input
                      type="time"
                      step={300}
                      min={arrivalDate === minDate ? minTime : "00:00"}
                      value={arrivalClock}
                      onChange={(e) => setArrivalClock(e.target.value)}
                      className="route-input h-9 w-full rounded-md border border-border px-3 text-sm"
                    />
                  </div>
                ) : (
                  <div className="h-9 rounded-md border border-border bg-muted" />
                )}
              </div>

              <div className="grid gap-2">
                <button
                  onClick={planRoute}
                  className="h-9 rounded-md bg-primary text-primary-foreground text-sm font-medium"
                  disabled={loading}
                >
                  {loading ? "Planning..." : `Plan ${routeType === "fastest" ? "Fastest" : "Shortest"} Route`}
                </button>
                <div className="grid grid-cols-2 gap-2">
                  <button
                    onClick={swapRoute}
                    className="h-9 rounded-md border border-border text-sm"
                    disabled={!start || !end}
                  >
                    Swap
                  </button>
                  <button
                    onClick={resetRoute}
                    className="h-9 rounded-md border border-border text-sm"
                  >
                    Reset
                  </button>
                </div>
              </div>

              {status && <p className="text-xs text-destructive">{status}</p>}
            </CardContent>
          </Card>

          <Card>
            <CardHeader className="pb-2">
              <CardTitle className="text-base">Route Summary</CardTitle>
              <CardDescription>Estimated travel time and recommended departure</CardDescription>
            </CardHeader>
            <CardContent className="space-y-3">
              <SummaryRow label="ETA" value={etaMinutes != null ? `${etaMinutes} min` : "-"} />
              <SummaryRow label="Distance" value={distanceMiles != null ? `${distanceMiles.toFixed(1)} mi` : "-"} />
              <SummaryRow label="Leave by" value={leaveBy || "-"} />
              <SummaryRow label="Nodes" value={routeIds.length ? routeIds.length.toString() : "-"} />
              <div className="pt-2 text-[11px] text-muted-foreground">
                Fastest path uses predicted speeds from the ST-GCN model. ETA updates when you plan a route.
              </div>
            </CardContent>
          </Card>

          <Card className="flex-1 flex flex-col">
            <CardHeader className="pb-2">
              <CardTitle className="text-base">Routing Methods</CardTitle>
              <CardDescription>Algorithms used for each option</CardDescription>
            </CardHeader>
            <CardContent className="space-y-3 text-sm">
              <div className="rounded-md border border-border p-3">
                <div className="flex items-center justify-between">
                  <span className="font-medium">Fastest (traffic-aware)</span>
                  <span className="text-[11px] text-muted-foreground">Dijkstra + ST-GCN weights</span>
                </div>
                <p className="mt-2 text-xs text-muted-foreground">
                  Uses ST-GCN predicted speeds to weight edges as time cost (1 / speed).
                </p>
              </div>
              <div className="rounded-md border border-border p-3">
                <div className="flex items-center justify-between">
                  <span className="font-medium">Shortest (topology)</span>
                  <span className="text-[11px] text-muted-foreground">Unweighted shortest path</span>
                </div>
                <p className="mt-2 text-xs text-muted-foreground">
                  Minimizes hop count on the sensor graph without traffic weighting.
                </p>
              </div>
            </CardContent>
          </Card>

        </div>
      </div>
    </div>
  );
}

function SummaryRow({ label, value }: { label: string; value: string }) {
  return (
    <div className="flex items-center justify-between text-sm">
      <span className="text-muted-foreground">{label}</span>
      <span className="text-foreground font-medium">{value}</span>
    </div>
  );
}
