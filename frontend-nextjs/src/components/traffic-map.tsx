"use client";

import { useEffect, useRef } from "react";
import L from "leaflet";
import "leaflet/dist/leaflet.css";

interface Sensor {
  id: number;
  sensor_id?: string;
  lat: number;
  lng: number;
  avg_speed: number;
  congestion: number;
  color: string;
}

interface TrafficMapProps {
  sensors: Sensor[];
  edges: [number, number][];
  height?: string;
  highlightSensorId?: number | null;
}

function getCongestionColor(c: number): string {
  if (c < 0.2) return "#22c55e";
  if (c < 0.35) return "#84cc16";
  if (c < 0.5) return "#eab308";
  if (c < 0.65) return "#f97316";
  if (c < 0.8) return "#ef4444";
  return "#dc2626";
}

export default function TrafficMap({ sensors, edges, height = "520px", highlightSensorId = null }: TrafficMapProps) {
  const mapRef = useRef<HTMLDivElement>(null);
  const mapInstanceRef = useRef<L.Map | null>(null);
  const highlightLayerRef = useRef<L.Layer | null>(null);

  useEffect(() => {
    if (!mapRef.current || mapInstanceRef.current) return;
    if (!sensors.length) return;

    // Compute center from actual sensor positions
    const avgLat = sensors.reduce((s, p) => s + p.lat, 0) / sensors.length;
    const avgLng = sensors.reduce((s, p) => s + p.lng, 0) / sensors.length;

    const map = L.map(mapRef.current, {
      center: [avgLat, avgLng],
      zoom: 12,
      zoomControl: true,
    });

    L.tileLayer("https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png", {
      attribution: '&copy; <a href="https://carto.com/">CARTO</a>',
      maxZoom: 19,
    }).addTo(map);

    // Sensor dots
    sensors.forEach((s) => {
      L.circleMarker([s.lat, s.lng], {
        radius: 4,
        fillColor: getCongestionColor(s.congestion),
        color: "rgba(0,0,0,0.25)",
        weight: 1,
        fillOpacity: 1,
      })
        .bindTooltip(
          `<div style="font-size:12px;line-height:1.5">
            <strong>Sensor ${s.id}</strong><br/>
            Speed: ${s.avg_speed.toFixed(1)} mph<br/>
            Congestion: ${(s.congestion * 100).toFixed(0)}%
          </div>`,
          { direction: "top", offset: [0, -6] }
        )
        .addTo(map);
    });

    // Legend
    const legend = new L.Control({ position: "bottomright" });
    legend.onAdd = () => {
      const div = L.DomUtil.create("div", "");
      div.style.cssText =
        "background:oklch(0.2686 0 0 / 0.9);padding:10px 14px;border-radius:8px;font-size:11px;color:oklch(0.9219 0 0);line-height:1.8;border:1px solid oklch(0.3715 0 0)";
      div.innerHTML = `
        <div style="font-weight:600;margin-bottom:2px;letter-spacing:0.05em;text-transform:uppercase;font-size:10px;color:oklch(0.7155 0 0)">Congestion</div>
        <div><span style="display:inline-block;width:20px;height:3px;border-radius:2px;background:#22c55e;margin-right:8px;vertical-align:middle"></span>Free flow</div>
        <div><span style="display:inline-block;width:20px;height:3px;border-radius:2px;background:#eab308;margin-right:8px;vertical-align:middle"></span>Moderate</div>
        <div><span style="display:inline-block;width:20px;height:3px;border-radius:2px;background:#f97316;margin-right:8px;vertical-align:middle"></span>Slow</div>
        <div><span style="display:inline-block;width:20px;height:3px;border-radius:2px;background:#ef4444;margin-right:8px;vertical-align:middle"></span>Congested</div>
      `;
      return div;
    };
    legend.addTo(map);

    mapInstanceRef.current = map;
    return () => { map.remove(); mapInstanceRef.current = null; };
  }, [sensors, edges]);

  useEffect(() => {
    const map = mapInstanceRef.current;
    if (!map) return;

    if (highlightLayerRef.current) {
      highlightLayerRef.current.remove();
      highlightLayerRef.current = null;
    }

    if (highlightSensorId == null) return;
    const sensor = sensors.find((s) => s.id === highlightSensorId);
    if (!sensor) return;

    const marker = L.circleMarker([sensor.lat, sensor.lng], {
      radius: 9,
      fillColor: "rgba(0,0,0,0)",
      color: "#ffffff",
      weight: 2,
      fillOpacity: 0.0,
    }).addTo(map);

    highlightLayerRef.current = marker;
    map.setView([sensor.lat, sensor.lng], Math.max(map.getZoom(), 12), { animate: true });
  }, [highlightSensorId, sensors]);

  return <div ref={mapRef} style={{ height, width: "100%", borderRadius: "var(--radius)" }} />;
}
