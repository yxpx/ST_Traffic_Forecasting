"""
Generate full sensor time series (200 steps) in mph for the Next.js frontend.
Outputs frontend-nextjs/public/timeseries-data.json.
"""
import json
import os
import sys
import numpy as np
import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from models.stgcn_model import SimpleSTGCN
from src.data_loader import create_feature_bundle, load_metr_la

WINDOW = 12
HORIZON = 3
STEPS = 200
OUT_PATH = os.path.join(ROOT, "frontend-nextjs", "public", "timeseries-data.json")


def main() -> None:
    print("Loading data...", flush=True)
    h5_path = os.path.join(ROOT, "data/raw/METR-LA.h5")
    adj_path = os.path.join(ROOT, "data/raw/adj_METR-LA.pkl")
    weather_path = os.path.join(ROOT, "data/raw/weather_CA_2019.csv")

    # Raw speeds for de-normalization
    df = load_metr_la(h5_path)
    speeds = df.values.astype(np.float32)
    mean = float(np.nanmean(speeds))
    std = float(np.nanstd(speeds)) or 1.0

    bundle = create_feature_bundle(h5_path, adj_path, weather_path)
    n_features = bundle.features.shape[-1]

    model = SimpleSTGCN(in_features=n_features, hidden_size=32, horizon=HORIZON)
    model.load_state_dict(torch.load(os.path.join(ROOT, "models", "stgcn_weights.pt"), map_location="cpu"))
    model.eval()
    edge_index = torch.tensor(bundle.edge_index, dtype=torch.long)

    total_steps = bundle.features.shape[0]
    max_start = total_steps - WINDOW - HORIZON - STEPS
    start_idx = 5000 if max_start > 5000 else max(0, max_start)

    print(f"Generating time series window starting at index {start_idx}", flush=True)

    n_sensors = bundle.features.shape[1]
    actual_steps = np.zeros((STEPS, n_sensors), dtype=np.float32)
    pred_steps = np.zeros((STEPS, n_sensors), dtype=np.float32)

    for t in range(STEPS):
        idx = start_idx + t
        x = torch.from_numpy(bundle.features[idx : idx + WINDOW][None]).float()
        with torch.no_grad():
            pred = model(x, edge_index)  # (1, N, H)
        pred_norm = pred[0, :, 0].cpu().numpy()
        actual_norm = bundle.target[idx + WINDOW, :]

        pred_steps[t] = pred_norm * std + mean
        actual_steps[t] = actual_norm * std + mean

    series = {}
    for i, sid in enumerate(bundle.sensor_ids):
        series[str(sid)] = {
            "actual": actual_steps[:, i].tolist(),
            "predicted": pred_steps[:, i].tolist(),
        }

    payload = {
        "sensor_ids": [str(s) for s in bundle.sensor_ids],
        "series": series,
        "unit": "mph",
        "start_index": int(start_idx),
        "steps": STEPS,
    }

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(payload, f)
    print(f"Wrote {OUT_PATH} ({os.path.getsize(OUT_PATH) // 1024} KB)", flush=True)


if __name__ == "__main__":
    main()
