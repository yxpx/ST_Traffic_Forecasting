import asyncio
import base64
import json
import os
import queue as queue_mod
import threading
import time
from typing import Dict, List, Optional

import cv2
import networkx as nx
import numpy as np
import pandas as pd
import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from ultralytics import YOLO

from models.stgcn_model import SimpleSTGCN
from backend.data_loader import create_feature_bundle, create_sliding_windows


class PredictRequest(BaseModel):
    timestamp: Optional[str] = None
    weather_override: Optional[Dict[str, float]] = None


class RouteRequest(BaseModel):
    start_sensor_id: str
    end_sensor_id: str


class CvRunRequest(BaseModel):
    source: str
    path: Optional[str] = None
    max_seconds: Optional[int] = None
    frame_stride: Optional[int] = None


class ModelService:
    def __init__(self):
        self.bundle = None
        self.model = None
        self.edge_index = None
        self.sensor_ids = []
        self.graph = None

    def load(self):
        h5_path = os.getenv("METR_LA_H5", "data/raw/METR-LA.h5")
        adj_path = os.getenv("METR_LA_ADJ", "data/raw/adj_METR-LA.pkl")
        weather_path = os.getenv("WEATHER_PATH", "data/raw/weather_CA_2019.csv")
        self.bundle = create_feature_bundle(h5_path, adj_path, weather_path)
        self.edge_index = torch.tensor(self.bundle.edge_index, dtype=torch.long)
        self.sensor_ids = self.bundle.sensor_ids
        self.graph = self._build_graph(self.bundle.edge_index)

        in_features = self.bundle.features.shape[-1]
        self.model = SimpleSTGCN(in_features=in_features, hidden_size=32, horizon=3)
        weights_path = os.getenv("MODEL_WEIGHTS")
        if weights_path and os.path.exists(weights_path):
            self.model.load_state_dict(torch.load(weights_path, map_location="cpu"))

    def _build_graph(self, edge_index: np.ndarray) -> nx.Graph:
        graph = nx.Graph()
        for src, dst in edge_index.T:
            graph.add_edge(int(src), int(dst), weight=1.0)
        return graph

    def predict_latest(self) -> Dict[str, float]:
        features = self.bundle.features
        window = min(12, features.shape[0])
        x, _ = create_sliding_windows(features, self.bundle.target, window=window, horizon=1)
        if len(x) == 0:
            raise ValueError("Not enough data for prediction")
        x_tensor = torch.tensor(x[-1:], dtype=torch.float32)
        pred = self.model(x_tensor, self.edge_index)
        pred = pred.squeeze(0).detach().cpu().numpy()
        if pred.ndim == 1:
            pred = pred[:, None]
        return {self.sensor_ids[i]: float(pred[i, 0]) for i in range(len(self.sensor_ids))}

    def route(self, start_id: str, end_id: str) -> Dict[str, List[str]]:
        if start_id not in self.sensor_ids or end_id not in self.sensor_ids:
            raise ValueError("Unknown sensor id")
        start = self.sensor_ids.index(start_id)
        end = self.sensor_ids.index(end_id)

        # Shortest on topology
        shortest = nx.shortest_path(self.graph, source=start, target=end)

        # Fastest uses inverse predicted speed as weight
        speeds = self.predict_latest()
        for src, dst, data in self.graph.edges(data=True):
            sid = self.sensor_ids[src]
            speed = max(speeds.get(sid, 5.0), 1.0)
            data["weight"] = 1.0 / speed
        fastest = nx.shortest_path(self.graph, source=start, target=end, weight="weight")

        return {
            "shortest": [self.sensor_ids[idx] for idx in shortest],
            "fastest": [self.sensor_ids[idx] for idx in fastest],
        }


# COCO class IDs for vehicles only
VEHICLE_CLASSES = {2, 3, 5, 7}  # car, motorcycle, bus, truck
CONF_THRESHOLD = 0.45  # higher threshold reduces flickering


def _compute_congestion(vehicle_count: int, avg_displacement: float) -> dict:
    """Derive congestion from vehicle count and avg pixel displacement between frames.

    avg_displacement: average centre-point movement (px) of tracked vehicles.
       - High displacement → vehicles moving → less congestion.
       - Low displacement + many vehicles → stuck → severe.
    """
    if vehicle_count == 0:
        return {"level": "Empty", "pct": 0}

    # Normalise displacement: <5px ≈ stopped, >40px ≈ free flow
    move_score = min(avg_displacement / 40.0, 1.0)  # 0=stuck 1=flowing
    density_score = min(vehicle_count / 15.0, 1.0)    # 0=empty 1=packed

    # Congestion = high density + low movement
    cong = density_score * (1 - move_score)
    pct = int(round(cong * 100))

    if pct >= 70:
        level = "Severe"
    elif pct >= 45:
        level = "Moderate"
    elif pct >= 20:
        level = "Light"
    else:
        level = "Free Flow"

    return {"level": level, "pct": pct}


class CvService:
    def __init__(self):
        self.model = None

    def load(self):
        if self.model is None:
            self.model = YOLO("models/yolov8n.pt")

    def _track_vehicles(self, frame):
        """Run tracking with vehicle filter. Returns (count, annotated_frame, boxes_xyxy)."""
        results = self.model.track(frame, persist=True, conf=CONF_THRESHOLD, verbose=False)
        r = results[0]
        if r.boxes is not None and len(r.boxes) > 0:
            cls_ids = r.boxes.cls.cpu().numpy().astype(int)
            mask = np.isin(cls_ids, list(VEHICLE_CLASSES))
            if mask.any():
                r.boxes = r.boxes[mask]
            else:
                r.boxes = None
        count = len(r.boxes) if r.boxes is not None else 0
        # Collect centre points for displacement calc
        centres = []
        if r.boxes is not None and count > 0:
            xyxy = r.boxes.xyxy.cpu().numpy()
            for box in xyxy:
                cx = (box[0] + box[2]) / 2
                cy = (box[1] + box[3]) / 2
                centres.append((cx, cy))
        annotated = r.plot()
        return count, annotated, centres

    def run(self, source: str, path: Optional[str] = None, max_seconds: Optional[int] = None, frame_stride: Optional[int] = None) -> Dict[str, float]:
        self.load()

        if source == "camera":
            cap = cv2.VideoCapture(0)
        else:
            video_path = path or os.getenv("CV_VIDEO_PATH", "assets/temp_video.mp4")
            if not os.path.exists(video_path):
                raise ValueError(f"Video not found: {video_path}")
            cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            raise ValueError("Unable to open video source")

        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 1280)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 720)

        out_path = os.path.join("frontend", "public", "cctv-detected.mp4")
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

        fallback_seconds = int(os.getenv("CV_MAX_SECONDS", "30"))
        max_seconds = max(5, min(int(max_seconds or fallback_seconds), 60))
        max_frames = int(max_seconds * fps)

        fallback_stride = int(os.getenv("CV_FRAME_STRIDE", "2"))
        frame_stride = max(1, min(int(frame_stride or fallback_stride), 5))

        frames = 0
        total = 0
        det_total = 0
        det_max = 0
        det_samples = 0
        start_time = time.time()
        while frames < max_frames:
            ok, frame = cap.read()
            if not ok:
                break
            total += 1
            if total % frame_stride != 0:
                continue
            results = self.model(frame, verbose=False)
            # Filter vehicles for the batch endpoint
            r = results[0]
            if r.boxes is not None and len(r.boxes) > 0:
                cls_ids = r.boxes.cls.cpu().numpy().astype(int)
                mask = np.isin(cls_ids, list(VEHICLE_CLASSES))
                r.boxes = r.boxes[mask] if mask.any() else None
            det_count = len(r.boxes) if r.boxes is not None else 0
            annotated = r.plot()
            det_total += det_count
            det_max = max(det_max, det_count)
            det_samples += 1
            writer.write(annotated)
            frames += 1

        cap.release()
        writer.release()
        elapsed = time.time() - start_time

        avg_det = det_total / det_samples if det_samples else 0.0
        return {
            "frames": frames,
            "seconds": int(elapsed),
            "avg_detections": round(avg_det, 2),
            "max_detections": int(det_max),
            "output": "/cctv-detected.mp4",
        }


app = FastAPI(title="TrafficFlow API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
service = ModelService()
cv_service = CvService()


@app.on_event("startup")
def _startup():
    service.load()


@app.post("/predict")
def predict(_: PredictRequest) -> Dict[str, float]:
    return service.predict_latest()


@app.post("/route")
def route(request: RouteRequest) -> Dict[str, List[str]]:
    return service.route(request.start_sensor_id, request.end_sensor_id)


@app.post("/cv/run")
def run_cv(request: CvRunRequest):
    source = request.source.lower().strip()
    if source not in {"video", "camera"}:
        raise HTTPException(status_code=400, detail="source must be 'video' or 'camera'")
    try:
        return cv_service.run(source, request.path, request.max_seconds, request.frame_stride)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/cv/stream")
async def stream_cv(
    source: str = "video",
    path: str = "",
    max_seconds: int = 30,
    frame_stride: int = 2,
):
    """SSE endpoint that streams annotated frames as base64 JPEG."""
    source = source.lower().strip()
    if source not in ("video", "camera"):
        raise HTTPException(status_code=400, detail="source must be 'video' or 'camera'")

    cv_service.load()
    q: queue_mod.Queue = queue_mod.Queue(maxsize=30)

    def _worker():
        try:
            if source == "camera":
                cap = cv2.VideoCapture(0)
            else:
                vp = path or os.getenv("CV_VIDEO_PATH", "assets/temp_video.mp4")
                if not os.path.exists(vp):
                    q.put({"error": f"Video not found: {vp}"})
                    q.put(None)
                    return
                cap = cv2.VideoCapture(vp)

            if not cap.isOpened():
                q.put({"error": "Cannot open video source"})
                q.put(None)
                return

            fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
            ms = max(5, min(max_seconds, 60))
            mf = int(ms * fps)
            stride = max(1, min(frame_stride, 5))

            n = 0
            raw = 0
            dt = 0
            dm = 0
            t0 = time.time()
            prev_centres: list = []  # previous frame centres for displacement
            disp_history: list = []  # rolling window of avg displacements
            DISP_WINDOW = 8  # frames to average for congestion smoothing

            while n < mf:
                ok, frame = cap.read()
                if not ok:
                    break
                raw += 1
                if raw % stride != 0:
                    continue

                nd, ann, centres = cv_service._track_vehicles(frame)
                dt += nd
                dm = max(dm, nd)
                n += 1

                # Compute displacement from previous frame
                avg_disp = 0.0
                if prev_centres and centres:
                    # Simple nearest-neighbour displacement
                    disps = []
                    for (cx, cy) in centres:
                        best = min(
                            (((cx - px) ** 2 + (cy - py) ** 2) ** 0.5 for (px, py) in prev_centres),
                            default=0.0,
                        )
                        disps.append(best)
                    avg_disp = sum(disps) / len(disps) if disps else 0.0
                prev_centres = centres

                disp_history.append(avg_disp)
                if len(disp_history) > DISP_WINDOW:
                    disp_history.pop(0)
                smoothed_disp = sum(disp_history) / len(disp_history)

                cong = _compute_congestion(nd, smoothed_disp)
                dt += nd
                dm = max(dm, nd)
                n += 1

                # Resize for streaming bandwidth
                h, w = ann.shape[:2]
                if w > 720:
                    scale = 720 / w
                    ann = cv2.resize(ann, (720, int(h * scale)))
                _, buf = cv2.imencode(".jpg", ann, [cv2.IMWRITE_JPEG_QUALITY, 70])
                b64 = base64.b64encode(buf).decode()

                elapsed = time.time() - t0
                q.put(
                    {
                        "frame": b64,
                        "n": n,
                        "det": nd,
                        "avg": round(dt / n, 2),
                        "max": dm,
                        "elapsed": round(elapsed, 1),
                        "congestion": cong["level"],
                        "congestion_pct": cong["pct"],
                    }
                )

            cap.release()
            elapsed = time.time() - t0
            q.put(
                {
                    "done": True,
                    "frames": n,
                    "seconds": round(elapsed, 1),
                    "avg": round(dt / n if n else 0, 2),
                    "max": dm,
                }
            )
        except Exception as e:
            q.put({"error": str(e)})
        finally:
            q.put(None)

    threading.Thread(target=_worker, daemon=True).start()

    async def _events():
        while True:
            try:
                item = q.get_nowait()
            except queue_mod.Empty:
                await asyncio.sleep(0.03)
                continue
            if item is None:
                break
            yield f"data: {json.dumps(item)}\n\n"

    return StreamingResponse(
        _events(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/dashboard/context")
def dashboard_context():
    """Return latest weather conditions."""
    # --- Weather (latest row) ---
    try:
        weather_path = os.getenv("WEATHER_PATH", "data/raw/weather_CA_2019.csv")
        wdf = pd.read_csv(weather_path)
        # Skip unit row
        while len(wdf) > 0 and (
            wdf.iloc[0].isna().all()
            or "Statute miles" in str(wdf.iloc[0].to_list())
            or "Millibars" in str(wdf.iloc[0].to_list())
        ):
            wdf = wdf.iloc[1:].copy()
        latest = wdf.iloc[-1]
        temp_f = float(latest.get("air_temp", 72))
        humidity = float(latest.get("relative_humidity", 50))
        visibility = float(latest.get("visibility", 10))
        wind = float(latest.get("wind_speed", 0))
    except Exception:
        temp_f, humidity, visibility, wind = 72.0, 50.0, 10.0, 0.0

    return {
        "weather": {
            "temp_f": round(temp_f, 1),
            "humidity": round(humidity, 1),
            "visibility_mi": round(visibility, 1),
            "wind_mph": round(wind, 1),
        },
    }


# ---------- Sensor context for CCTV integration ----------

@app.get("/sensor/{sensor_idx}")
def sensor_context(sensor_idx: int):
    """Return model predictions and metadata for a single sensor.

    sensor_idx: 0-based index into the 207 sensors.
    """
    if sensor_idx < 0 or sensor_idx >= len(service.sensor_ids):
        raise HTTPException(status_code=404, detail="Sensor index out of range")

    sid = service.sensor_ids[sensor_idx]

    # De-normalise predicted speed to mph using the raw mean/std stored during loading
    try:
        preds = service.predict_latest()
        mean_s = service.bundle.speed_mean   # ~53.7 mph
        std_s = service.bundle.speed_std     # ~20.3 mph

        pred_norm = preds.get(sid, 0.0)
        pred_mph = round(float(pred_norm * std_s + mean_s), 1)

        # All sensor avg for comparison
        all_preds = np.array(list(preds.values()))
        all_mph = all_preds * std_s + mean_s
        network_avg = round(float(np.mean(all_mph)), 1)

        # This sensor's historical avg from raw speeds
        col_idx = service.sensor_ids.index(sid)
        raw_norm = service.bundle.features[:, col_idx, 0]  # normalised values
        hist_mph = round(float(np.mean(raw_norm) * std_s + mean_s), 1)
    except Exception:
        pred_mph = 0.0
        network_avg = 0.0
        hist_mph = 0.0

    # Congestion classification
    if pred_mph < 20:
        congestion = "Severe"
    elif pred_mph < 35:
        congestion = "Moderate"
    elif pred_mph < 50:
        congestion = "Light"
    else:
        congestion = "Free Flow"

    return {
        "sensor_id": sid,
        "sensor_idx": sensor_idx,
        "predicted_speed_mph": pred_mph,
        "historical_avg_mph": hist_mph,
        "network_avg_mph": network_avg,
        "congestion": congestion,
    }


# ---------- CCTV live state (shared between endpoints and homepage) ----------

_cctv_state: dict = {
    "active": False,
    "sensor_idx": None,
    "congestion": None,
    "congestion_pct": 0,
    "vehicles": 0,
    "last_updated": None,
}


@app.post("/cv/state")
def update_cctv_state(payload: dict):
    """Frontend posts current detection state so main dashboard can read it."""
    _cctv_state.update({
        "active": payload.get("active", False),
        "sensor_idx": payload.get("sensor_idx"),
        "congestion": payload.get("congestion"),
        "congestion_pct": payload.get("congestion_pct", 0),
        "vehicles": payload.get("vehicles", 0),
        "last_updated": time.strftime("%H:%M:%S"),
    })
    return {"ok": True}


@app.get("/cv/state")
def get_cctv_state():
    """Main dashboard polls this to show CCTV status."""
    return _cctv_state
