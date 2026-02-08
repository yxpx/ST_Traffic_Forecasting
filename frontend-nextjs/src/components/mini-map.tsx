"use client";

import { useEffect, useRef } from "react";
import L from "leaflet";
import "leaflet/dist/leaflet.css";

interface Sensor {
  id: number;
  lat: number;
  lng: number;
  avg_speed: number;
  congestion: number;
}

interface MiniMapProps {
  sensors: Sensor[];
  highlightIdx: number | null;
  height?: string;
}

export default function MiniMap({ sensors, highlightIdx, height = "200px" }: MiniMapProps) {
  const ref = useRef<HTMLDivElement>(null);
  const mapRef = useRef<L.Map | null>(null);
  const markerRef = useRef<L.CircleMarker | null>(null);

  // Create map once
  useEffect(() => {
    if (!ref.current || mapRef.current) return;
    if (!sensors.length) return;

    const avgLat = sensors.reduce((s, p) => s + p.lat, 0) / sensors.length;
    const avgLng = sensors.reduce((s, p) => s + p.lng, 0) / sensors.length;

    const map = L.map(ref.current, {
      center: [avgLat, avgLng],
      zoom: 11,
      zoomControl: false,
      attributionControl: false,
      dragging: false,
      scrollWheelZoom: false,
      doubleClickZoom: false,
    });

    L.tileLayer("https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png", {
      maxZoom: 19,
    }).addTo(map);

    // Light dots for all sensors
    sensors.forEach((s) => {
      L.circleMarker([s.lat, s.lng], {
        radius: 2,
        fillColor: "#4a5568",
        color: "transparent",
        fillOpacity: 0.5,
      }).addTo(map);
    });

    mapRef.current = map;
    // Let the container render, then tell Leaflet to recalculate size
    requestAnimationFrame(() => {
      if (mapRef.current) mapRef.current.invalidateSize();
    });
    return () => { map.remove(); mapRef.current = null; };
  }, [sensors]);

  // Highlight selected sensor
  useEffect(() => {
    const map = mapRef.current;
    if (!map) return;

    if (markerRef.current) {
      markerRef.current.remove();
      markerRef.current = null;
    }

    if (highlightIdx == null || !sensors[highlightIdx]) return;
    const s = sensors[highlightIdx];

    const m = L.circleMarker([s.lat, s.lng], {
      radius: 7,
      fillColor: "#3b82f6",
      color: "#ffffff",
      weight: 2,
      fillOpacity: 0.9,
    })
      .bindTooltip(`Sensor ${highlightIdx + 1}`, { permanent: true, direction: "right", offset: [8, 0], className: "mini-map-label" })
      .addTo(map);

    markerRef.current = m;
    // Use animate:false to avoid _leaflet_pos crash on zoom transitions
    map.setView([s.lat, s.lng], 13, { animate: false });
  }, [highlightIdx, sensors]);

  return <div ref={ref} style={{ height, width: "100%", borderRadius: "var(--radius)" }} />;
}
