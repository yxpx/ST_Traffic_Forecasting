"""
Generate dashboard data: prediction samples, feature importance, error analysis.
Outputs JSON files for the Observable frontend.
"""
import json, os, sys
import numpy as np
import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from models.stgcn_model import SimpleSTGCN
from src.data_loader import create_feature_bundle

WINDOW = 12
HORIZON = 3
OUT_DIR = os.path.join(ROOT, "frontend-observable", "src")

def main():
    print("Loading data...", flush=True)
    bundle = create_feature_bundle(
        os.path.join(ROOT, "data/raw/METR-LA.h5"),
        os.path.join(ROOT, "data/raw/adj_METR-LA.pkl"),
        os.path.join(ROOT, "data/raw/weather_CA_2019.csv"),
    )

    n_features = bundle.features.shape[-1]
    model = SimpleSTGCN(in_features=n_features, hidden_size=32, horizon=HORIZON)
    model.load_state_dict(torch.load(os.path.join(ROOT, "models/stgcn_weights.pt"), map_location="cpu"))
    model.eval()
    edge_index = torch.tensor(bundle.edge_index, dtype=torch.long)

    # Feature names
    feature_names = [
        "Speed (z-scored)", "Time of Day (sin)", "Time of Day (cos)",
        "Visibility", "Sea-Level Pressure", "Relative Humidity",
        "Wind Speed", "Pressure (alt)", "Air Temperature",
        "Wind Direction", "Feature 11"
    ][:n_features]

    # --- 1. Prediction samples across different time periods ---
    print("Generating predictions...", flush=True)
    T_total = bundle.features.shape[0]
    N_sensors = bundle.features.shape[1]

    # Pick strategic sample indices across the dataset
    sample_indices = list(range(1000, min(T_total - WINDOW - HORIZON, 30000), 500))
    
    # Collect predictions for all sensors at many time points
    all_preds = []
    all_actuals = []
    for idx in sample_indices:
        x = torch.from_numpy(bundle.features[idx:idx+WINDOW][None]).float()
        with torch.no_grad():
            pred = model(x, edge_index)  # (1, N, H)
        pred_np = pred[0].numpy()  # (N, H)
        actual_np = bundle.target[idx + WINDOW : idx + WINDOW + HORIZON]  # (H, N)
        all_preds.append(pred_np)  # (N, H)
        all_actuals.append(actual_np.T)  # (N, H)

    all_preds = np.array(all_preds)    # (S, N, H)
    all_actuals = np.array(all_actuals) # (S, N, H)

    # --- 2. Scatter plot data: predicted vs actual (horizon=1, all sensors, subset of times) ---
    scatter_idx = list(range(0, len(sample_indices), 3))  # every 3rd sample
    scatter_data = []
    for si in scatter_idx:
        for sensor in range(0, N_sensors, 5):  # every 5th sensor
            scatter_data.append({
                "actual": float(all_actuals[si, sensor, 0]),
                "predicted": float(all_preds[si, sensor, 0]),
            })
    print(f"  Scatter points: {len(scatter_data)}", flush=True)

    # --- 3. Time series for specific sensors ---
    demo_sensors = [0, 50, 100, 150, 200]
    ts_range = range(5000, min(5500, T_total - WINDOW - HORIZON))
    timeseries = {}
    for s in demo_sensors:
        ts_actual = []
        ts_pred = []
        for idx in ts_range:
            x = torch.from_numpy(bundle.features[idx:idx + WINDOW][None]).float()
            with torch.no_grad():
                pred = model(x, edge_index)
            ts_actual.append(float(bundle.target[idx + WINDOW, s]))
            ts_pred.append(float(pred[0, s, 0]))
        timeseries[str(s)] = {"actual": ts_actual, "predicted": ts_pred}
    print(f"  Time series: {len(demo_sensors)} sensors × {len(ts_range)} steps", flush=True)

    # --- 4. Error distribution ---
    errors = (all_preds[:, :, 0] - all_actuals[:, :, 0]).flatten()
    abs_errors = np.abs(errors)
    error_stats = {
        "mean_error": float(np.mean(errors)),
        "std_error": float(np.std(errors)),
        "mae": float(np.mean(abs_errors)),
        "rmse": float(np.sqrt(np.mean(errors ** 2))),
        "p50": float(np.percentile(abs_errors, 50)),
        "p90": float(np.percentile(abs_errors, 90)),
        "p95": float(np.percentile(abs_errors, 95)),
    }
    # Histogram bins
    hist_counts, hist_edges = np.histogram(errors, bins=50)
    error_hist = [{"bin_start": float(hist_edges[i]), "bin_end": float(hist_edges[i+1]),
                   "count": int(hist_counts[i])} for i in range(len(hist_counts))]

    # --- 5. Feature importance via input gradient ---
    print("Computing feature importance...", flush=True)
    # Use gradient-based: average |dL/dx| across features
    importances = np.zeros(n_features)
    grad_samples = list(range(1000, min(T_total - WINDOW - HORIZON, 20000), 2000))
    for idx in grad_samples:
        x = torch.from_numpy(bundle.features[idx:idx + WINDOW][None]).float()
        x.requires_grad_(True)
        pred = model(x, edge_index)
        target = torch.from_numpy(bundle.target[idx + WINDOW:idx + WINDOW + HORIZON].T[None]).float()
        loss = ((pred - target) ** 2).mean()
        loss.backward()
        grad = x.grad.abs().mean(dim=(0, 1, 2)).numpy()  # (F,)
        importances += grad
    importances /= len(grad_samples)
    importances /= importances.sum()  # normalize to sum=1
    feature_importance = [{"feature": feature_names[i], "importance": float(importances[i])}
                          for i in range(n_features)]
    feature_importance.sort(key=lambda x: x["importance"], reverse=True)

    # --- 6. Per-sensor performance ---
    sensor_mae = np.mean(np.abs(all_preds[:, :, 0] - all_actuals[:, :, 0]), axis=0)  # (N,)
    sensor_perf = [{"sensor": i, "mae": float(sensor_mae[i])} for i in range(N_sensors)]
    sensor_perf.sort(key=lambda x: x["mae"])

    # --- Write outputs ---
    dashboard_data = {
        "scatter": scatter_data,
        "timeseries": timeseries,
        "error_stats": error_stats,
        "error_histogram": error_hist,
        "feature_importance": feature_importance,
        "sensor_performance": {
            "best_5": sensor_perf[:5],
            "worst_5": sensor_perf[-5:],
        },
        "n_sensors": N_sensors,
        "n_timesteps": T_total,
        "n_features": n_features,
        "feature_names": feature_names,
    }

    out_path = os.path.join(OUT_DIR, "dashboard-data.json")
    with open(out_path, "w") as f:
        json.dump(dashboard_data, f, indent=2)
    print(f"Wrote {out_path} ({os.path.getsize(out_path) // 1024} KB)", flush=True)
    print("Done.", flush=True)


if __name__ == "__main__":
    main()
