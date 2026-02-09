import json
from typing import Dict, List

import numpy as np


def load_camera_mapping(path: str) -> Dict[str, List[Dict]]:
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def split_counts(camera_counts: Dict[str, float], mapping: Dict[str, List[Dict]]) -> Dict[str, float]:
    sensor_updates: Dict[str, float] = {}
    for zone, count in camera_counts.items():
        if zone not in mapping:
            continue
        for rule in mapping[zone]:
            sensor_id = str(rule["sensor_id"])
            weight = float(rule.get("weight", 1.0))
            sensor_updates[sensor_id] = sensor_updates.get(sensor_id, 0.0) + count * weight
    return sensor_updates


def density_to_speed_impact(density: float, free_flow_speed: float = 65.0, jam_density: float = 200.0) -> float:
    # Simple inverse linear proxy based on Greenshields' model
    impact = free_flow_speed * (1.0 - min(density / jam_density, 1.0))
    return max(5.0, impact)


def inject_camera_effect(
    features: np.ndarray,
    sensor_ids: List[str],
    sensor_updates: Dict[str, float],
    time_index: int,
    speed_feature_index: int = 0,
) -> np.ndarray:
    updated = features.copy()
    sensor_index = {sid: idx for idx, sid in enumerate(sensor_ids)}
    for sensor_id, density in sensor_updates.items():
        if sensor_id not in sensor_index:
            continue
        idx = sensor_index[sensor_id]
        speed_impact = density_to_speed_impact(density)
        updated[time_index, idx, speed_feature_index] = speed_impact
    return updated
