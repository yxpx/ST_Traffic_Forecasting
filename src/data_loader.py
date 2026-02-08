import json
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


@dataclass
class MetrLAData:
    speeds: np.ndarray
    timestamps: pd.DatetimeIndex
    sensor_ids: List[str]
    adjacency_matrix: np.ndarray


@dataclass
class FeatureBundle:
    features: np.ndarray
    target: np.ndarray
    timestamps: pd.DatetimeIndex
    sensor_ids: List[str]
    edge_index: np.ndarray
    speed_mean: float = 0.0
    speed_std: float = 1.0


def load_metr_la(h5_path: str) -> pd.DataFrame:
    if not os.path.exists(h5_path):
        raise FileNotFoundError(h5_path)

    # Try h5py first (no pytables dependency)
    import h5py

    with h5py.File(h5_path, "r") as f:
        keys = list(f.keys())
        if not keys:
            raise ValueError(f"No datasets found in {h5_path}")

        # METR-LA .h5 files are either a single dataset or a group with axis/block
        obj = f[keys[0]]
        if isinstance(obj, h5py.Dataset):
            data = obj[:]
            df = pd.DataFrame(data)
        else:
            # PyTables-style: look for block0_values inside the group
            def _find_datasets(group, prefix=""):
                datasets = {}
                for k in group.keys():
                    item = group[k]
                    if isinstance(item, h5py.Dataset):
                        datasets[f"{prefix}/{k}"] = item
                    elif isinstance(item, h5py.Group):
                        datasets.update(_find_datasets(item, f"{prefix}/{k}"))
                return datasets

            datasets = _find_datasets(obj)
            block_key = [k for k in datasets if "block0_values" in k]
            axis_key = [k for k in datasets if "axis1" in k]
            axis0_key = [k for k in datasets if "axis0" in k]

            if block_key:
                data = datasets[block_key[0]][:]
                columns = None
                index = None
                # In PyTables HDF5: axis0 = columns, axis1 = index
                if axis0_key:
                    raw_ax0 = datasets[axis0_key[0]][:]
                    columns = [
                        c.decode() if isinstance(c, bytes) else str(c)
                        for c in raw_ax0
                    ]
                if axis_key:
                    raw_ax1 = datasets[axis_key[0]][:]
                    try:
                        index = pd.to_datetime(raw_ax1)
                    except Exception:
                        index = None
                # If columns length doesn't match data width, swap
                if columns is not None and len(columns) != data.shape[1]:
                    columns, index = index, columns
                    if isinstance(columns, list) and len(columns) != data.shape[1]:
                        columns = None
                df = pd.DataFrame(data, columns=columns, index=index)
            else:
                # Fallback: grab the largest dataset
                largest = max(datasets.values(), key=lambda d: d.size)
                df = pd.DataFrame(largest[:])

    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.date_range("2019-01-01", periods=len(df), freq="5min")
    return df


def load_adj_mx(pkl_path: str) -> Tuple[List[str], np.ndarray]:
    if not os.path.exists(pkl_path):
        raise FileNotFoundError(pkl_path)
    with open(pkl_path, "rb") as file:
        obj = pickle_load(file)
    if isinstance(obj, (list, tuple)) and len(obj) >= 2:
        sensor_ids = obj[0]
        # METR-LA format: (sensor_ids, sensor_id_to_ind, adj_mx)
        # obj[1] might be a dict (sensor_id_to_ind) — skip it
        adjacency_matrix = None
        for item in obj[1:]:
            if isinstance(item, np.ndarray) and item.ndim == 2:
                adjacency_matrix = item
                break
            elif isinstance(item, (list, tuple)):
                for sub in item:
                    if isinstance(sub, np.ndarray) and sub.ndim == 2:
                        adjacency_matrix = sub
                        break
                if adjacency_matrix is not None:
                    break
        if adjacency_matrix is None and isinstance(obj[-1], np.ndarray):
            adjacency_matrix = obj[-1]
    elif isinstance(obj, dict):
        sensor_ids = obj.get("sensor_ids") or obj.get("sensor_id")
        adjacency_matrix = obj.get("adj_mx") or obj.get("adjacency_matrix")
    else:
        raise ValueError("Unexpected adj_mx.pkl structure")
    if sensor_ids is None or adjacency_matrix is None:
        raise ValueError("adj_mx.pkl missing sensor_ids or adjacency_matrix")
    return list(map(str, sensor_ids)), np.array(adjacency_matrix, dtype=np.float32)


def build_edge_index(adjacency_matrix: np.ndarray) -> np.ndarray:
    rows, cols = np.nonzero(adjacency_matrix)
    return np.stack([rows, cols], axis=0).astype(np.int64)


def _parse_weather_csv(weather_path: str) -> pd.DataFrame:
    if not os.path.exists(weather_path):
        raise FileNotFoundError(weather_path)
    df = pd.read_csv(weather_path)
    # Drop unnamed/empty columns
    df = df.loc[:, [col for col in df.columns if col.strip() and not col.startswith("Unnamed")]]
    # Remove unit row(s) if present
    while len(df) > 0 and (df.iloc[0].isna().all() or "Statute miles" in str(df.iloc[0].to_list()) or "Millibars" in str(df.iloc[0].to_list())):
        df = df.iloc[1:].copy()
    # Ensure date_time column exists
    if "date_time" not in df.columns:
        if "Date" in df.columns and "Time" in df.columns:
            df["date_time"] = df["Date"] + "-" + df["Time"]
        else:
            raise ValueError("Weather CSV missing date_time column")
    df["date_time"] = pd.to_datetime(df["date_time"], format="%m/%d/%y-%I:%M%p", errors="coerce")
    df = df.dropna(subset=["date_time"]).set_index("date_time")
    for col in df.columns:
        if col == "date_time":
            continue
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.sort_index()
    return df


def _resample_weather(weather_df: pd.DataFrame, freq: str = "5min") -> pd.DataFrame:
    numeric_cols = weather_df.select_dtypes(include=["number"]).columns
    return weather_df[numeric_cols].resample(freq).mean().ffill()


def _build_time_features(timestamps: pd.DatetimeIndex) -> np.ndarray:
    minutes = timestamps.hour * 60 + timestamps.minute
    frac = minutes / (24 * 60)
    sin_t = np.sin(2 * np.pi * frac)
    cos_t = np.cos(2 * np.pi * frac)
    return np.stack([sin_t, cos_t], axis=-1)


def align_weather_to_timestamps(weather_df: pd.DataFrame, timestamps: pd.DatetimeIndex) -> pd.DataFrame:
    weather_df = _resample_weather(weather_df)
    weather_df = weather_df.copy()
    weather_df["key"] = weather_df.index.strftime("%m-%d-%H:%M")
    weather_map = weather_df.groupby("key").mean()
    keys = timestamps.strftime("%m-%d-%H:%M")
    aligned = weather_map.reindex(keys).ffill().bfill()
    aligned.index = timestamps
    # Fill any remaining NaN with column means
    aligned = aligned.fillna(aligned.mean())
    aligned = aligned.fillna(0.0)
    return aligned


def normalize_zscore(values: np.ndarray) -> Tuple[np.ndarray, float, float]:
    mean = np.nanmean(values)
    std = np.nanstd(values)
    if std == 0:
        std = 1.0
    return (values - mean) / std, mean, std


def create_feature_bundle(
    h5_path: str,
    adj_path: str,
    weather_path: Optional[str] = None,
) -> FeatureBundle:
    df = load_metr_la(h5_path)
    sensor_ids, adjacency_matrix = load_adj_mx(adj_path)
    speeds = df.values.astype(np.float32)
    # Fill NaN speeds with 0 (missing sensor readings)
    speeds = np.nan_to_num(speeds, nan=0.0)
    timestamps = df.index

    speeds_norm, speed_mean, speed_std = normalize_zscore(speeds)
    speeds_norm = np.nan_to_num(speeds_norm, nan=0.0)
    time_features = _build_time_features(timestamps)

    weather_features = None
    if weather_path:
        weather_df = _parse_weather_csv(weather_path)
        aligned_weather = align_weather_to_timestamps(weather_df, timestamps)
        wf = aligned_weather.values.astype(np.float32)
        # Z-score normalize each weather column independently
        for c in range(wf.shape[1]):
            col = wf[:, c]
            m, s = np.nanmean(col), np.nanstd(col)
            if s == 0:
                s = 1.0
            wf[:, c] = (col - m) / s
        wf = np.nan_to_num(wf, nan=0.0)
        weather_features = wf

    # Build feature tensor: (time, nodes, features)
    base_feat = speeds_norm[:, :, None]
    time_feat = np.repeat(time_features[:, None, :], speeds.shape[1], axis=1)
    if weather_features is not None:
        weather_feat = np.repeat(weather_features[:, None, :], speeds.shape[1], axis=1)
        features = np.concatenate([base_feat, time_feat, weather_feat], axis=-1)
    else:
        features = np.concatenate([base_feat, time_feat], axis=-1)

    # Final NaN guard
    features = np.nan_to_num(features, nan=0.0).astype(np.float32)

    edge_index = build_edge_index(adjacency_matrix)
    target = np.nan_to_num(speeds_norm, nan=0.0).astype(np.float32)
    return FeatureBundle(
        features=features,
        target=target,
        timestamps=timestamps,
        sensor_ids=sensor_ids,
        edge_index=edge_index,
        speed_mean=float(speed_mean),
        speed_std=float(speed_std),
    )


def create_sliding_windows(features: np.ndarray, target: np.ndarray, window: int, horizon: int) -> Tuple[np.ndarray, np.ndarray]:
    total = features.shape[0]
    xs, ys = [], []
    for i in range(total - window - horizon + 1):
        xs.append(features[i : i + window])
        ys.append(target[i + window : i + window + horizon])
    return np.stack(xs, axis=0), np.stack(ys, axis=0)


def load_config(config_path: str) -> Dict:
    with open(config_path, "r", encoding="utf-8") as file:
        return json.load(file)


def pickle_load(file_obj):
    import pickle

    try:
        return pickle.load(file_obj)
    except UnicodeDecodeError:
        file_obj.seek(0)
        return pickle.load(file_obj, encoding="latin1")
