"""
=============================================================================
TVSI TRAFFIC SYSTEM - FIXED VEHICLE DETECTION & LOGIC
=============================================================================
CRITICAL FIXES:
1. Strict vehicle-only detection (no humans/pedestrians)
2. Enhanced ROI-based vehicle counting
3. Better TVSI computation for actual traffic conditions
4. Realistic baseline values for vehicle traffic
5. Improved density and flow calculations
=============================================================================
"""

import cv2
import time
import numpy as np
import pandas as pd
import supervision as sv
from ultralytics import YOLO
from collections import defaultdict, deque
from datetime import datetime
from typing import List, Tuple, Dict, Optional, Any
import subprocess
import os

cv2.setUseOptimized(True)
cv2.setNumThreads(4)


# =============================================================================
# FIXED TVSI ENGINE - VEHICLE TRAFFIC ONLY
# =============================================================================

class TrafficVitalStabilityIndex:
    """
    Production-Grade TVSI Engine for VEHICLE TRAFFIC ONLY
    
    FIX #1: ROI Clarification
    - 'density' = vehicles in monitored zone / effective_monitored_area_m2
    - Monitored zone = full visible frame (no explicit polygon ROI)
    - roi_area_m2 is a scaling constant for the effective visible area
    
    Key Features:
    - Realistic baseline values for vehicle traffic
    - Flow = unique vehicle crossings per window (not cumulative)
    - Amber Alert = rate-of-change early warning (not severity level)
    """
    
    def __init__(
        self,
        baseline_window_size: int = 5,
        roi_area_m2: float = 500.0,
        speed_smoothing_window: int = 3,
        congestion_density_threshold: float = 0.03  # 15 vehicles in 500m² ROI
    ):
        self.baseline_window_size = baseline_window_size
        self.roi_area_m2 = roi_area_m2
        self.speed_smoothing_window = speed_smoothing_window
        self.congestion_density_threshold = congestion_density_threshold
        
        # Baseline storage
        self.flow_baseline_buffer = []
        self.density_baseline_buffer = []
        self.speed_var_baseline_buffer = []
        
        # REALISTIC DEFAULTS for vehicle traffic
        self.flow_baseline_mean = 8.0  # 8 vehicles per 5-sec window = reasonable flow
        self.flow_baseline_std = 3.0
        self.density_baseline_mean = 0.015  # ~7-8 vehicles in 500m² ROI
        self.density_baseline_std = 0.008
        self.speed_var_baseline_mean = 80.0  # Higher variance for mixed traffic
        self.speed_var_baseline_std = 30.0
        
        # State
        self.baseline_initialized = False
        self.window_count = 0
        self.speed_var_history = deque(maxlen=speed_smoothing_window)
        self.tvsi_history = deque(maxlen=100)
        self.state_history = deque(maxlen=100)
        
        # Congestion tracking
        self.congestion_events = []
        self.last_congestion_alert = None
        
        # AMBER ALERT: Rate-of-change tracking (CRITICAL FIX)
        self.tvsi_derivative_history = deque(maxlen=10)
        self.amber_alert_active = False
        self.amber_alert_timestamp = None
        self.time_to_congestion_sec = None
        
        # ST-GCN simulation (coordination loss detector)
        self.coordination_score_history = deque(maxlen=20)
        self.last_stgcn_update = 0
    
    def compute_tvsi(
        self,
        flow: int,
        vehicle_count_in_roi: int,
        speeds: List[float],
        stgcn_anomaly: float = 0.0
    ) -> Dict[str, Any]:
        """
        FIXED: Compute TVSI for VEHICLE traffic only.
        
        Returns immediate, realistic feedback for vehicle traffic patterns.
        """
        
        # Compute density: vehicles in monitored zone / effective monitored area (m²)
        # ROI is abstracted as full visible frame; roi_area_m2 is a dataset-agnostic scaling constant
        density = vehicle_count_in_roi / self.roi_area_m2 if self.roi_area_m2 > 0 else 0
        
        # Compute speed variance with defensive handling for edge cases
        if len(speeds) > 2:
            speed_variance = float(np.var(speeds))  # Explicit float for consistency
        elif len(speeds) == 2:
            speed_variance = (speeds[0] - speeds[1]) ** 2 / 2
        elif len(speeds) == 1:
            speed_variance = 0.0  # Single vehicle = no variance by definition
        else:
            speed_variance = 0.0  # No speeds = assume stable (conservative default)
        
        self.speed_var_history.append(speed_variance)
        smoothed_speed_var = np.mean(self.speed_var_history)
        
        # Calculate average speed (0.0 if no samples - conservative for congestion detection)
        avg_speed = float(np.mean(speeds)) if len(speeds) > 0 else 0.0
        
        # FIX #3: BASELINE COLLECTION - Only collect during healthy traffic conditions
        # SAFEGUARD: Congested samples are excluded because learning from congestion
        # would inflate baselines, causing future congestion to appear "normal".
        if not self.baseline_initialized:
            is_healthy_sample = (
                flow >= 2 and  # Minimum healthy flow
                vehicle_count_in_roi < 15 and  # Below congestion threshold
                density < self.congestion_density_threshold * 0.8  # Not approaching congestion
            )
            if is_healthy_sample:
                self.flow_baseline_buffer.append(flow)
                self.density_baseline_buffer.append(density)
                self.speed_var_baseline_buffer.append(smoothed_speed_var)
                self.window_count += 1
            
            if self.window_count >= self.baseline_window_size:
                self._compute_baselines()
        
        # IMPROVED NORMALIZATION
        norm_flow = self._safe_normalize(flow, self.flow_baseline_mean, self.flow_baseline_std)
        norm_density = self._safe_normalize(density, self.density_baseline_mean, self.density_baseline_std)
        norm_speed_var = self._safe_normalize(smoothed_speed_var, self.speed_var_baseline_mean, self.speed_var_baseline_std)
        stgcn_anomaly = np.clip(stgcn_anomaly, 0.0, 1.0)
        
        # ENHANCED TFSI - Traffic Flow Stability Index
        # Higher flow = good, higher density = bad
        tfsi = norm_flow - 2.0 * norm_density
        
        # TVSI CALCULATION with improved weights
        tvsi_raw = (
            0.5 * tfsi -                    # Flow vs Density (50%)
            0.25 * norm_speed_var -         # Speed Stability (25%)
            0.25 * stgcn_anomaly            # Pattern Anomaly (25%)
        )
        
        # DENSITY PENALTY: Strong penalty for high vehicle density
        if density > self.congestion_density_threshold:
            density_ratio = density / self.congestion_density_threshold
            congestion_penalty = (density_ratio - 1.0) ** 1.5 * 0.6
            tvsi_raw -= congestion_penalty
        
        # FLOW REWARD: Bonus for good vehicle flow
        if flow > self.flow_baseline_mean and density < self.congestion_density_threshold:
            flow_bonus = 0.15
            tvsi_raw += flow_bonus
        
        # SPEED CONSISTENCY REWARD
        if avg_speed > 40 and speed_variance < 100 and density < self.congestion_density_threshold / 2:
            consistency_bonus = 0.2
            tvsi_raw += consistency_bonus
        
        tvsi = np.clip(tvsi_raw, -1.0, 1.0)
        
        # CLASSIFY STATE
        state, explanation, severity = self._classify_traffic_state(
            tvsi, norm_flow, norm_density, norm_speed_var, 
            stgcn_anomaly, density, avg_speed, flow, vehicle_count_in_roi
        )
        
        # CONGESTION EVENT LOGGING
        if severity in ["CRITICAL", "SEVERE"]:
            self._log_congestion_event(flow, density, avg_speed, state, vehicle_count_in_roi)
        
        # Store history
        self.tvsi_history.append(tvsi)
        self.state_history.append(state)
        
        # Calculate trend AND derivative (CRITICAL FOR AMBER ALERT)
        trend = self._calculate_trend()
        tvsi_derivative = self._calculate_derivative()
        self.tvsi_derivative_history.append(tvsi_derivative)
        
        # AMBER ALERT: Detect rapid decline BEFORE congestion hits
        amber_alert, amber_reason, time_to_congestion = self._check_amber_alert(
            tvsi, tvsi_derivative, density, avg_speed, flow
        )
        self.time_to_congestion_sec = time_to_congestion
        
        # Generate suggested action
        suggested_action = self._get_suggested_action(state, severity, amber_alert, density, flow)
        
        return {
            'tvsi': float(tvsi),
            'state': state,
            'severity': severity,
            'explanation': explanation,
            'trend': trend,
            'components': {
                'tfsi': float(tfsi),
                'norm_flow': float(norm_flow),
                'norm_density': float(norm_density),
                'norm_speed_var': float(norm_speed_var),
                'stgcn_anomaly': float(stgcn_anomaly),
                'raw_density': float(density),
                'avg_speed': float(avg_speed),
                'vehicle_count': vehicle_count_in_roi
            },
            'baseline_ready': self.baseline_initialized,
            'baseline_progress': f"{self.window_count}/{self.baseline_window_size}",
            'congestion_detected': severity in ["CRITICAL", "SEVERE"],
            # NEW: Amber Alert outputs
            'amber_alert': amber_alert,
            'amber_reason': amber_reason if amber_alert else None,
            'time_to_congestion_sec': time_to_congestion,
            'tvsi_derivative': float(tvsi_derivative),
            'suggested_action': suggested_action
        }
    
    def _compute_baselines(self):
        """Compute baseline statistics from actual vehicle traffic."""
        if len(self.flow_baseline_buffer) > 0:
            self.flow_baseline_mean = max(np.mean(self.flow_baseline_buffer), 3.0)  # Min 3 vehicles/window
            self.flow_baseline_std = max(np.std(self.flow_baseline_buffer), 2.0)
        
        if len(self.density_baseline_buffer) > 0:
            self.density_baseline_mean = max(np.mean(self.density_baseline_buffer), 0.01)
            self.density_baseline_std = max(np.std(self.density_baseline_buffer), 0.005)
        
        if len(self.speed_var_baseline_buffer) > 0:
            self.speed_var_baseline_mean = np.mean(self.speed_var_baseline_buffer)
            self.speed_var_baseline_std = max(np.std(self.speed_var_baseline_buffer), 20.0)
        
        self.baseline_initialized = True
        print(f"\n✅ BASELINE CALIBRATED (Vehicle Traffic):")
        print(f"   Flow: {self.flow_baseline_mean:.1f}±{self.flow_baseline_std:.1f} vehicles/window")
        print(f"   Density: {self.density_baseline_mean:.4f}±{self.density_baseline_std:.4f} vehicles/m²")
        print(f"   Speed Variance: {self.speed_var_baseline_mean:.1f}±{self.speed_var_baseline_std:.1f}")
    
    def _safe_normalize(self, value: float, mean: float, std: float) -> float:
        """Z-score normalization with safety checks."""
        if mean is None or std is None or std < 1e-6:
            return 0.0
        normalized = (value - mean) / std
        return np.clip(normalized, -3.0, 3.0)
    
    def _classify_traffic_state(
        self,
        tvsi: float,
        norm_flow: float,
        norm_density: float,
        norm_speed_var: float,
        stgcn_anomaly: float,
        raw_density: float,
        avg_speed: float,
        flow: int,
        vehicle_count: int
    ) -> Tuple[str, str, str]:
        """Enhanced classification for vehicle traffic."""
        
        # CRITICAL GRIDLOCK
        if vehicle_count > 25 or raw_density > self.congestion_density_threshold * 2:
            state = "Severe Congestion"
            severity = "CRITICAL"
            explanation = f"🚨 GRIDLOCK: {vehicle_count} vehicles in ROI, density {raw_density:.4f}/m²"
        
        elif tvsi < -0.5:
            state = "Critical Failure"
            severity = "CRITICAL"
            reasons = []
            if vehicle_count > 18:
                reasons.append(f"{vehicle_count} vehicles (overcrowded)")
            if flow < 3:
                reasons.append(f"flow collapse ({flow} veh/window)")
            if avg_speed < 15:
                reasons.append(f"gridlock speed ({avg_speed:.0f} km/h)")
            explanation = f"🚨 CRITICAL: {', '.join(reasons) if reasons else 'immediate intervention needed'}"
        
        # HEAVY CONGESTION
        elif tvsi < -0.2:
            state = "Moderate Congestion"
            severity = "SEVERE" if tvsi < -0.35 else "WARNING"
            reasons = []
            if vehicle_count > 12:
                reasons.append(f"{vehicle_count} vehicles")
            if flow < 5:
                reasons.append(f"low flow ({flow} veh/window)")
            if avg_speed < 30:
                reasons.append(f"slow ({avg_speed:.0f} km/h)")
            explanation = f"⚠️ CONGESTION: {', '.join(reasons) if reasons else 'traffic degrading'}"
        
        # LIGHT CONGESTION
        elif tvsi < 0.0:
            state = "Light Congestion"
            severity = "CAUTION"
            reasons = []
            if vehicle_count > 8:
                reasons.append(f"{vehicle_count} vehicles")
            if avg_speed < 40:
                reasons.append(f"slower ({avg_speed:.0f} km/h)")
            explanation = f"⚡ CAUTION: {', '.join(reasons) if reasons else 'minor slowdown'}"
        
        # STABLE FLOW
        elif tvsi < 0.3:
            state = "Stable Flow"
            severity = "NORMAL"
            explanation = f"✓ STABLE: {vehicle_count} vehicles, flow {flow}, {avg_speed:.0f} km/h"
        
        # EXCELLENT
        else:
            state = "Excellent"
            severity = "OPTIMAL"
            reasons = []
            if flow >= self.flow_baseline_mean:
                reasons.append(f"good flow ({flow} veh/window)")
            if avg_speed > 50:
                reasons.append(f"free flow ({avg_speed:.0f} km/h)")
            if vehicle_count < 8:
                reasons.append(f"low density ({vehicle_count} vehicles)")
            explanation = f"✓✓ OPTIMAL: {', '.join(reasons) if reasons else 'excellent conditions'}"
        
        return state, explanation, severity
    
    def _calculate_trend(self) -> str:
        """Calculate TVSI trend."""
        if len(self.tvsi_history) < 3:
            return "STABLE"
        
        recent = list(self.tvsi_history)[-3:]
        diff = recent[-1] - recent[0]
        
        if diff > 0.1:
            return "IMPROVING"
        elif diff < -0.1:
            return "DEGRADING"
        else:
            return "STABLE"
    
    def _calculate_derivative(self) -> float:
        """Calculate instantaneous TVSI rate of change (units per window)."""
        if len(self.tvsi_history) < 2:
            return 0.0
        return self.tvsi_history[-1] - self.tvsi_history[-2]
    
    def _check_amber_alert(
        self,
        tvsi: float,
        tvsi_derivative: float,
        density: float,
        avg_speed: float,
        flow: int
    ) -> Tuple[bool, str, Optional[float]]:
        """
        AMBER ALERT: Rate-of-change early warning system.
        
        CORE DEFINITION: "Rapid degradation of traffic stability while recovery is still possible."
        
        KEY DISTINCTION:
        - Amber is triggered by TRAJECTORY (derivative), NOT absolute severity
        - Amber fires when TVSI is declining rapidly BUT still above critical
        - This gives operators a window to intervene BEFORE congestion locks in
        
        Triggers:
        1. Rapid TVSI decline (derivative < -0.15) while in warning zone
        2. Predictive: projected to hit critical within 30 seconds
        3. Leading indicators (density↑ + speed↓) while in warning zone
        """
        amber_active = False
        reason = ""
        time_to_congestion = None
        
        # Condition 1: Rapid decline detected (CORE AMBER TRIGGER)
        rapid_decline = tvsi_derivative < -0.15
        
        # Condition 2: TVSI in warning zone but NOT yet critical (still recoverable)
        in_warning_zone = -0.3 < tvsi < 0.2
        
        # Condition 3: Leading indicators (supporting evidence, not primary trigger)
        density_rising = density > self.congestion_density_threshold * 0.7
        speed_dropping = avg_speed < 35
        flow_reducing = flow < self.flow_baseline_mean * 0.6
        
        # Calculate time to congestion (linear extrapolation)
        if tvsi_derivative < -0.05 and tvsi > -0.5:
            # How many windows until TVSI hits -0.5 (critical)?
            windows_to_critical = (tvsi - (-0.5)) / abs(tvsi_derivative)
            time_to_congestion = windows_to_critical * 5.0  # 5 sec per window
        
        # AMBER ALERT CONDITIONS
        if rapid_decline and in_warning_zone:
            amber_active = True
            reason = f"Rapid TVSI decline ({tvsi_derivative:.2f}/window)"
        elif in_warning_zone and density_rising and speed_dropping:
            amber_active = True
            reason = f"Density↑ ({density:.4f}) + Speed↓ ({avg_speed:.0f}km/h)"
        elif in_warning_zone and flow_reducing and density_rising:
            amber_active = True
            reason = f"Flow collapse ({flow} veh) + Density rising"
        elif time_to_congestion is not None and time_to_congestion < 30:
            amber_active = True
            reason = f"Congestion predicted in {time_to_congestion:.0f}s"
        
        # Track amber state
        if amber_active and not self.amber_alert_active:
            self.amber_alert_active = True
            self.amber_alert_timestamp = time.time()
            print(f"\n🟠 AMBER ALERT ACTIVATED: {reason}")
        elif not amber_active and self.amber_alert_active:
            self.amber_alert_active = False
            print("\n✅ Amber alert cleared")
        
        return amber_active, reason, time_to_congestion
    
    def _get_suggested_action(
        self,
        state: str,
        severity: str,
        amber_alert: bool,
        density: float,
        flow: int
    ) -> str:
        """Generate actionable intervention suggestion."""
        
        if severity == "CRITICAL":
            return "🚨 IMMEDIATE: Activate signal preemption, deploy traffic control"
        elif severity == "SEVERE":
            return "⚠️ URGENT: Extend green phase +30s on main corridor"
        elif amber_alert:
            if density > self.congestion_density_threshold * 0.8:
                return "🟠 RECOMMENDED: Reduce inflow - shorten feeder green by 15s"
            elif flow < self.flow_baseline_mean * 0.5:
                return "🟠 RECOMMENDED: Check for incidents, prepare alternate routing"
            else:
                return "🟠 MONITOR: Prepare intervention, congestion developing"
        elif severity == "WARNING":
            return "⚡ ADVISORY: Monitor closely, prepare signal timing adjustment"
        elif severity == "CAUTION":
            return "→ WATCH: Minor slowdown, no action needed yet"
        else:
            return "✓ OPTIMAL: No intervention required"
    
    def _log_congestion_event(self, flow: int, density: float, avg_speed: float, state: str, vehicle_count: int):
        """Log congestion event."""
        now = time.time()
        
        if self.last_congestion_alert and (now - self.last_congestion_alert) < 30:
            return
        
        self.last_congestion_alert = now
        event = {
            'timestamp': datetime.now(),
            'flow': flow,
            'density': density,
            'avg_speed': avg_speed,
            'state': state,
            'vehicle_count': vehicle_count
        }
        self.congestion_events.append(event)
        
        print(f"\n🚨 CONGESTION: {state} | {vehicle_count} vehicles | Density: {density:.4f}/m² | Speed: {avg_speed:.1f} km/h")
    
    def simulate_stgcn_anomaly(self, speeds: List[float], density: float) -> float:
        """
        Simulate ST-GCN coordination loss signal.
        
        In production, this would come from actual ST-GCN model.
        For demo, we simulate based on observable patterns:
        - High speed variance = loss of coordination (primary signal)
        - Bi-modal speed distribution = lane blocking
        - Density contributes minimally to avoid feedback loop
        """
        if len(speeds) < 3:
            return 0.0
        
        # Speed variance component (normalized) - PRIMARY SIGNAL
        speed_std = np.std(speeds)
        variance_signal = min(speed_std / 30.0, 1.0)  # Cap at 1.0
        
        # FIX #4: Density spike component - CAPPED to prevent density→stgcn→tvsi feedback
        # Density already affects TVSI directly via norm_density and congestion_penalty.
        # Here we only allow density to contribute to anomaly at EXTREME spikes (3x threshold),
        # and hard-cap at 0.4 to prevent double-penalization through stgcn_anomaly.
        density_signal = min(density / (self.congestion_density_threshold * 3 + 1e-9), 0.4)  # +1e-9 prevents div/0
        
        # Bi-modality detection (simple: check if speeds cluster in 2 groups)
        if len(speeds) >= 5:
            sorted_speeds = np.sort(speeds)
            mid = len(sorted_speeds) // 2
            gap = sorted_speeds[mid] - sorted_speeds[mid-1] if mid > 0 else 0
            bimodal_signal = min(gap / 20.0, 1.0) if gap > 10 else 0.0
        else:
            bimodal_signal = 0.0
        
        # Combine signals - speed variance dominates to reduce density coupling
        stgcn_anomaly = 0.5 * variance_signal + 0.25 * density_signal + 0.25 * bimodal_signal
        
        # Add temporal smoothing
        self.coordination_score_history.append(stgcn_anomaly)
        smoothed = np.mean(self.coordination_score_history)
        
        return float(np.clip(smoothed, 0.0, 1.0))


# =============================================================================
# TVSI INTEGRATION WRAPPER
# =============================================================================

class TVSIIntegration:
    """Handles windowing and aggregation for TVSI computation."""
    
    def __init__(
        self,
        window_duration_sec: float = 5.0,
        fps: float = 30.0,
        roi_area_m2: float = 500.0
    ):
        self.window_duration_sec = window_duration_sec
        self.fps = fps
        self.frames_per_window = int(window_duration_sec * fps)
        
        self.tvsi_engine = TrafficVitalStabilityIndex(
            baseline_window_size=5,
            roi_area_m2=roi_area_m2,
            speed_smoothing_window=3,
            congestion_density_threshold=0.03
        )
        
        # Window state
        self.current_window_frames = 0
        self.window_crossed_ids = set()  # Vehicles that crossed DURING this window only
        self.counted_before_window = set()  # FIX #2: Snapshot of counted_ids at window start
        self.window_speeds = []
        self.window_vehicle_counts = []
        self.latest_tvsi_result = None
    
    def update(
        self,
        current_detections: Any,
        counted_ids: set,
        track_history: dict
    ) -> Optional[Dict[str, Any]]:
        """Update with current frame data. ST-GCN anomaly is computed internally."""
        
        self.current_window_frames += 1
        
        # FIX #2: Initialize window snapshot on first frame of window
        if self.current_window_frames == 1:
            self.counted_before_window = counted_ids.copy()  # Snapshot at window start
        
        # FIX #1: Count vehicles in monitored zone (full frame acts as ROI)
        # Density = vehicles visible / effective_monitored_area
        vehicles_in_monitored_zone = len(current_detections) if current_detections is not None and len(current_detections) > 0 else 0
        self.window_vehicle_counts.append(vehicles_in_monitored_zone)
        
        # FIX #5: Collect trend-based speeds from vehicle tracks
        # Speed is computed from first-to-last position in track history (smoothed trend, not instantaneous)
        if current_detections is not None and len(current_detections) > 0:
            # DEFENSIVE: Ensure tracker_id array exists and is iterable
            tracker_ids = current_detections.tracker_id if current_detections.tracker_id is not None else []
            for track_id in tracker_ids:
                if track_id is None:
                    continue  # Skip invalid track IDs
                if track_id in track_history and len(track_history[track_id]) >= 2:
                    x_prev, y_prev, t_prev = track_history[track_id][0]  # First recorded position
                    x_curr, y_curr, t_curr = track_history[track_id][-1]  # Latest position
                    
                    dist_pixels = np.hypot(x_curr - x_prev, y_curr - y_prev)
                    dist_meters = dist_pixels * 0.05
                    time_sec = t_curr - t_prev
                    
                    # DEFENSIVE: Avoid division by zero on stationary or same-frame detections
                    if time_sec > 0.01:  # Minimum 10ms between samples
                        speed_kmh = (dist_meters / time_sec) * 3.6
                        if 5 < speed_kmh < 150:  # Reasonable vehicle speed range
                            self.window_speeds.append(speed_kmh)
        
        # FIX #2: Track crossings ONLY within this window (not cumulative)
        # FLOW SEMANTICS: A vehicle can contribute to flow in EXACTLY ONE window.
        # Once in counted_before_window, it's excluded from all future windows.
        new_crossings = counted_ids - self.counted_before_window - self.window_crossed_ids
        self.window_crossed_ids.update(new_crossings)
        
        # Check if window complete
        if self.current_window_frames >= self.frames_per_window:
            flow = len(self.window_crossed_ids)
            avg_vehicle_count = int(np.mean(self.window_vehicle_counts)) if self.window_vehicle_counts else 0
            
            # Simulate ST-GCN anomaly signal (replaces hardcoded 0.0)
            # Uses same roi_area as TVSI engine for consistency
            density_estimate = avg_vehicle_count / 500.0 if avg_vehicle_count > 0 else 0.0
            simulated_stgcn = self.tvsi_engine.simulate_stgcn_anomaly(
                self.window_speeds if self.window_speeds else [],
                density_estimate
            )
            
            result = self.tvsi_engine.compute_tvsi(
                flow=flow,
                vehicle_count_in_roi=avg_vehicle_count,
                speeds=self.window_speeds if self.window_speeds else [],
                stgcn_anomaly=simulated_stgcn
            )
            
            result['window_data'] = {
                'flow': flow,
                'avg_density': avg_vehicle_count,
                'speed_samples': len(self.window_speeds),
                'avg_speed': float(np.mean(self.window_speeds)) if self.window_speeds else 0.0
            }
            
            self.latest_tvsi_result = result
            
            # Reset window
            self.current_window_frames = 0
            self.window_crossed_ids.clear()
            self.window_speeds.clear()
            self.window_vehicle_counts.clear()
            
            return result
        
        return None
    
    def get_latest_result(self) -> Optional[Dict[str, Any]]:
        """Get most recent TVSI result."""
        return self.latest_tvsi_result


# =============================================================================
# VISUALIZATION
# =============================================================================

def draw_tvsi_panel(panel, tvsi_result, x_offset=750, y_start=20):
    """Draw TVSI information on dashboard panel."""
    
    if tvsi_result is None:
        cv2.putText(panel, "TVSI: Waiting for vehicles...", (x_offset, y_start),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)
        return
    
    # Color based on severity
    severity = tvsi_result.get('severity', 'NORMAL')
    severity_colors = {
        'OPTIMAL': (0, 255, 0),
        'NORMAL': (0, 255, 255),
        'CAUTION': (0, 200, 255),
        'WARNING': (0, 165, 255),
        'SEVERE': (0, 100, 255),
        'CRITICAL': (0, 0, 255)
    }
    color = severity_colors.get(severity, (200, 200, 200))
    
    # Draw TVSI value
    tvsi_value = tvsi_result['tvsi']
    cv2.putText(panel, f"TVSI: {tvsi_value:.3f}", (x_offset, y_start),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    
    # Draw state with badge
    state = tvsi_result['state']
    severity_badge = {
        'OPTIMAL': '✓✓',
        'NORMAL': '✓',
        'CAUTION': '⚡',
        'WARNING': '⚠',
        'SEVERE': '⚠⚠',
        'CRITICAL': '🚨'
    }
    badge = severity_badge.get(severity, '•')
    cv2.putText(panel, f"{badge} {state}", (x_offset, y_start + 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    
    # Draw trend
    trend = tvsi_result.get('trend', 'STABLE')
    trend_symbols = {'IMPROVING': '↑', 'DEGRADING': '↓', 'STABLE': '→'}
    trend_colors = {'IMPROVING': (0, 255, 0), 'DEGRADING': (0, 0, 255), 'STABLE': (200, 200, 200)}
    trend_symbol = trend_symbols.get(trend, '→')
    trend_color = trend_colors.get(trend, (200, 200, 200))
    cv2.putText(panel, f"{trend_symbol} {trend}", (x_offset, y_start + 55),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, trend_color, 1)
    
    # Draw explanation
    explanation = tvsi_result['explanation']
    if len(explanation) > 50:
        explanation = explanation[:47] + "..."
    cv2.putText(panel, explanation, (x_offset, y_start + 80),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
    
    # Draw window metrics
    if 'window_data' in tvsi_result:
        wd = tvsi_result['window_data']
        metrics_text = f"F:{wd['flow']} D:{wd['avg_density']} S:{wd['avg_speed']:.0f}km/h"
        cv2.putText(panel, metrics_text, (x_offset, y_start + 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (150, 150, 150), 1)
    
    # Draw baseline progress
    if not tvsi_result.get('baseline_ready', False):
        progress = tvsi_result.get('baseline_progress', '0/5')
        cv2.putText(panel, f"Calibrating: {progress}", (x_offset, y_start + 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 150, 255), 1)
    
    # AMBER ALERT BANNER (NEW: Early warning)
    if tvsi_result.get('amber_alert', False):
        cv2.rectangle(panel, (x_offset - 10, y_start + 125), 
                     (x_offset + 380, y_start + 155), (0, 165, 255), -1)
        ttc = tvsi_result.get('time_to_congestion_sec')
        ttc_str = f" | ETA: {ttc:.0f}s" if ttc else ""
        cv2.putText(panel, f"AMBER ALERT: INTERVENE NOW{ttc_str}", (x_offset, y_start + 145),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 2)
    # RED CONGESTION ALERT BANNER
    elif tvsi_result.get('congestion_detected', False):
        cv2.rectangle(panel, (x_offset - 10, y_start + 125), 
                     (x_offset + 350, y_start + 155), (0, 0, 255), -1)
        cv2.putText(panel, "RED ALERT: CONGESTION ACTIVE", (x_offset, y_start + 145),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    
    # SUGGESTED ACTION (NEW)
    action = tvsi_result.get('suggested_action', '')
    if action:
        cv2.putText(panel, action[:60], (x_offset - 10, y_start + 172),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (200, 200, 200), 1)


# =============================================================================
# MAIN SYSTEM
# =============================================================================

def get_video_source():
    """Interactive video source selection."""
    print("\n" + "="*60)
    print("🚦 TVSI TRAFFIC ANALYTICS - VEHICLE DETECTION SYSTEM")
    print("="*60)
    print("\nSelect Video Source:")
    print("1. 📹 Real-time Laptop Camera")
    print("2. 🎥 YouTube Video URL")
    print("3. 📁 Local Video File")
    print("="*60)
    
    while True:
        choice = input("\nEnter your choice (1/2/3): ").strip()
        
        if choice == "1":
            print("\n✅ Using laptop camera")
            print("⚠️  WARNING: Make sure to point camera at VEHICLES, not people!")
            return 0
        
        elif choice == "2":
            print("\n📺 YouTube Video Mode")
            url = input("Enter YouTube URL (or press Enter for default): ").strip()
            if not url:
                url = "https://www.youtube.com/watch?v=MNn9qKG2UFI"
                print(f"Using default: {url}")
            return url
        
        elif choice == "3":
            filepath = input("Enter video file path: ").strip()
            if os.path.exists(filepath):
                return filepath
            else:
                print("❌ File not found. Try again.")
        
        else:
            print("❌ Invalid choice. Please enter 1, 2, or 3.")


def get_video_capture(source):
    """Get video capture from various sources."""
    if isinstance(source, str) and ("youtube.com" in source or "youtu.be" in source):
        print("📺 Downloading YouTube video...")
        temp_file = "temp_video.mp4"
        
        if os.path.exists(temp_file):
            print("♻️  Using cached video...")
        else:
            print("⬇️  Downloading new video...")
            subprocess.run([
                "yt-dlp",
                "-f", "best[height<=720]",
                "-o", temp_file,
                "--quiet",
                "--no-warnings",
                source
            ])
        
        return cv2.VideoCapture(temp_file)
    else:
        return cv2.VideoCapture(source)


def main():
    """Main execution."""
    
    # Configuration
    VIDEO_SOURCE = get_video_source()
    
    OUTPUT_VIDEO = "tvsi_traffic_analysis.mp4"
    OUTPUT_EXCEL = "tvsi_traffic_data.xlsx"
    OUTPUT_ALERTS = "tvsi_alerts.txt"
    OUTPUT_TVSI = "tvsi_results.csv"
    
    MODEL_PATH = "models/yolov8n.pt"
    CONFIDENCE = 0.40  # Increased for better vehicle detection
    FRAME_SIZE = (1280, 720)
    PANEL_HEIGHT = 180
    WRITER_SIZE = (FRAME_SIZE[0], FRAME_SIZE[1] + PANEL_HEIGHT)
    
    # STRICT VEHICLE CLASSES ONLY
    VEHICLE_CLASSES = {
        2: "Car",
        3: "Motorcycle", 
        5: "Bus",
        7: "Truck"
    }
    VEHICLE_CLASS_IDS = list(VEHICLE_CLASSES.keys())
    
    LINE_Y = 400
    SPEED_LIMIT = 60
    PIXEL_TO_METER = 0.05
    
    # Initialize TVSI
    print("🧠 Initializing TVSI Engine for Vehicle Traffic...")
    tvsi_integration = TVSIIntegration(
        window_duration_sec=5.0,
        fps=30.0,
        roi_area_m2=500.0
    )
    
    tvsi_results_log = []
    alert_log = []
    congestion_events = []
    
    def send_alert(message):
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        alert_msg = f"[{timestamp}] {message}"
        alert_log.append(alert_msg)
        print(f"🚨 ALERT: {message}")
        with open(OUTPUT_ALERTS, 'a', encoding='utf-8') as f:
            f.write(alert_msg + '\n')
    
    # Load Model
    print("🤖 Loading YOLOv8 (Vehicle Detection Only)...")
    model = YOLO(MODEL_PATH)
    model.fuse()
    
    # Tracker
    tracker = sv.ByteTrack(
        track_activation_threshold=0.45,
        lost_track_buffer=20,
        minimum_matching_threshold=0.85
    )
    
    # Video Setup
    cap = get_video_capture(VIDEO_SOURCE)
    if not cap.isOpened():
        raise RuntimeError("❌ Cannot open video source")
    
    fps = 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    IS_CAMERA = (VIDEO_SOURCE == 0)
    if IS_CAMERA:
        print("📹 Real-time camera mode")
        total_frames = 0
    
    ret, test_frame = cap.read()
    if not ret:
        raise RuntimeError("❌ Cannot read video")
    
    test_frame = cv2.resize(test_frame, FRAME_SIZE)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
    writer = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, fps, WRITER_SIZE, True)
    
    if not writer.isOpened():
        raise RuntimeError("❌ VideoWriter failed")
    
    print(f"✅ System ready: {OUTPUT_VIDEO} @ {fps} FPS")
    
    # State
    frame_id = 0
    counted_ids = set()
    vehicle_count = 0
    track_history = defaultdict(list)
    vehicle_data = []
    alerted_ids = set()
    last_detections = None
    
    type_counts = {v: 0 for v in VEHICLE_CLASSES.values()}
    
    fps_start_time = time.time()
    fps_counter = 0
    current_fps = 0
    
    heatmap = np.zeros((FRAME_SIZE[1], FRAME_SIZE[0]), dtype=np.float32)
    
    print(f"🚀 Starting TVSI processing...")
    print("⚠️  IMPORTANT: System will only track VEHICLES (cars, buses, trucks, motorcycles)")
    print("Press 'Q' to quit, 'P' to pause, 'S' to screenshot, 'T' for TVSI stats")
    
    paused = False
    start_time = time.time()
    
    # MAIN LOOP
    try:
        while True:
            if not paused:
                ret, frame = cap.read()
                if not ret:
                    print("\n✅ Processing complete!")
                    break
    
                frame = cv2.resize(frame, FRAME_SIZE)
                frame_id += 1
    
                if not IS_CAMERA and frame_id % 100 == 0:
                    progress = (frame_id / total_frames) * 100 if total_frames > 0 else 0
                    print(f"📊 Progress: {progress:.1f}% ({frame_id}/{total_frames})")
    
                fps_counter += 1
                if time.time() - fps_start_time > 1:
                    current_fps = fps_counter / (time.time() - fps_start_time)
                    fps_counter = 0
                    fps_start_time = time.time()
    
                # Run detection every 2nd frame
                if frame_id % 2 == 0:
                    results = model(
                        frame,
                        conf=CONFIDENCE,
                        iou=0.5,
                        classes=VEHICLE_CLASS_IDS,  # STRICT: Only vehicles
                        verbose=False
                    )[0]
    
                    detections = sv.Detections.from_ultralytics(results)
                    detections = tracker.update_with_detections(detections)
                    last_detections = detections
                elif last_detections is not None:
                    detections = last_detections
                else:
                    detections = sv.Detections.empty()
            
            overlay = frame.copy()
    
            # Draw counting line
            cv2.line(overlay, (0, LINE_Y), (frame.shape[1], LINE_Y), (0, 0, 255), 3)
            cv2.putText(overlay, "COUNTING LINE", (10, LINE_Y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
            now = time.time()
    
            # Process VEHICLE detections
            if len(detections) > 0:
                for box, track_id, class_id in zip(
                    detections.xyxy, 
                    detections.tracker_id,
                    detections.class_id
                ):
                    x1, y1, x2, y2 = map(int, box)
                    cx = (x1 + x2) // 2
                    cy = y2
    
                    vehicle_type = VEHICLE_CLASSES.get(class_id, "Unknown")
    
                    cv2.circle(heatmap, (cx, cy), 20, 1, -1)
    
                    track_history[track_id].append((cx, cy, now))
                    if len(track_history[track_id]) > 10:
                        track_history[track_id].pop(0)
    
                    speed_kmh = 0
                    if len(track_history[track_id]) >= 2:
                        x_prev, y_prev, t_prev = track_history[track_id][0]
                        dist_pixels = np.hypot(cx - x_prev, cy - y_prev)
                        dist_meters = dist_pixels * PIXEL_TO_METER
                        time_sec = now - t_prev
    
                        if time_sec > 0:
                            speed_kmh = (dist_meters / time_sec) * 3.6
    
                    # Count vehicle crossing line
                    if cy > LINE_Y and track_id not in counted_ids:
                        counted_ids.add(track_id)
                        vehicle_count += 1
                        type_counts[vehicle_type] += 1
                        
                        vehicle_data.append({
                            'Timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                            'Vehicle_ID': int(track_id),
                            'Type': vehicle_type,
                            'Speed_kmh': round(speed_kmh, 2),
                            'Frame': frame_id
                        })
    
                    color = (0, 0, 255) if speed_kmh > SPEED_LIMIT else (0, 255, 0)
                    cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2)
    
                    label = f"ID:{track_id} {vehicle_type}"
                    speed_label = f"{int(speed_kmh)} km/h"
                    
                    cv2.putText(overlay, label, (x1, y1 - 25),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                    cv2.putText(overlay, speed_label, (x1, y1 - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    
                    if speed_kmh > SPEED_LIMIT and track_id not in alerted_ids:
                        cv2.putText(overlay, "⚠ OVERSPEED", (x1, y2 + 20),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                        send_alert(f"{vehicle_type} ID:{track_id} overspeeding at {int(speed_kmh)} km/h")
                        alerted_ids.add(track_id)
            
            # TVSI UPDATE
            if not paused:
                tvsi_result = tvsi_integration.update(
                    current_detections=detections,
                    counted_ids=counted_ids,
                    track_history=track_history
                )
                
                if tvsi_result is not None:
                    log_entry = {
                        'Timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                        'Frame': frame_id,
                        'TVSI': tvsi_result['tvsi'],
                        'State': tvsi_result['state'],
                        'Severity': tvsi_result['severity'],
                        'Trend': tvsi_result['trend'],
                        'Explanation': tvsi_result['explanation'],
                        'Flow': tvsi_result['window_data']['flow'],
                        'Density': tvsi_result['window_data']['avg_density'],
                        'Avg_Speed': tvsi_result['window_data']['avg_speed'],
                        'Congestion_Detected': tvsi_result['congestion_detected'],
                        # NEW fields
                        'Amber_Alert': tvsi_result.get('amber_alert', False),
                        'Time_To_Congestion': tvsi_result.get('time_to_congestion_sec'),
                        'TVSI_Derivative': tvsi_result.get('tvsi_derivative', 0),
                        'Suggested_Action': tvsi_result.get('suggested_action', '')
                    }
                    tvsi_results_log.append(log_entry)
                    
                    severity = tvsi_result['severity']
                    emoji = {
                        'OPTIMAL': '✓✓',
                        'NORMAL': '✓',
                        'CAUTION': '⚡',
                        'WARNING': '⚠',
                        'SEVERE': '⚠⚠',
                        'CRITICAL': '🚨'
                    }.get(severity, '•')
                    
                    print(f"\n{emoji} TVSI: {tvsi_result['tvsi']:.3f} | {tvsi_result['state']} | {tvsi_result['trend']}")
                    print(f"   {tvsi_result['explanation']}")
                    
                    # AMBER ALERT logging (NEW)
                    if tvsi_result.get('amber_alert', False):
                        ttc = tvsi_result.get('time_to_congestion_sec')
                        ttc_str = f" (ETA: {ttc:.0f}s)" if ttc else ""
                        amber_msg = f"🟠 AMBER: {tvsi_result.get('amber_reason', 'Intervene now')}{ttc_str}"
                        print(f"   {amber_msg}")
                        print(f"   ACTION: {tvsi_result.get('suggested_action', '')}")
                    
                    if tvsi_result['congestion_detected']:
                        alert_msg = f"🔴 RED ALERT: {tvsi_result['state']} - {tvsi_result['explanation']}"
                        send_alert(alert_msg)
                        congestion_events.append(log_entry)
            
            heatmap_normalized = cv2.normalize(heatmap, None, 0, 255, cv2.NORM_MINMAX)
            heatmap_colored = cv2.applyColorMap(heatmap_normalized.astype(np.uint8), cv2.COLORMAP_JET)
            overlay = cv2.addWeighted(overlay, 0.85, heatmap_colored, 0.15, 0)
    
            # Dashboard panel
            panel = np.zeros((PANEL_HEIGHT, FRAME_SIZE[0], 3), dtype=np.uint8)
            panel[:] = (40, 40, 40)
    
            mode_text = "CAMERA" if IS_CAMERA else "VIDEO"
            cv2.putText(panel, f"Mode: {mode_text}", (FRAME_SIZE[0] - 200, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 200, 255), 2)
    
            info_y = 30
            cv2.putText(panel, f"Total Vehicles: {vehicle_count}", (20, info_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            info_y += 30
            cv2.putText(panel, f"FPS: {current_fps:.1f}", (20, info_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
            info_y += 30
            if IS_CAMERA:
                cv2.putText(panel, f"Frame: {frame_id}", (20, info_y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            else:
                cv2.putText(panel, f"Frame: {frame_id}/{total_frames}", (20, info_y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
            info_y += 30
            cv2.putText(panel, f"Alerts: {len(alert_log)}", (20, info_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 100, 100), 2)
    
            type_x = 350
            type_y = 30
            for vtype, count in type_counts.items():
                cv2.putText(panel, f"{vtype}: {count}", (type_x, type_y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)
                type_y += 30
            
            draw_tvsi_panel(panel, tvsi_integration.get_latest_result())
    
            if paused:
                cv2.putText(panel, "⏸ PAUSED", (FRAME_SIZE[0] - 400, 70),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
    
            final_frame = np.vstack([overlay, panel])
    
            if not paused:
                writer.write(final_frame)
            
            cv2.imshow("TVSI Traffic Analytics System", final_frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                print("\n⚠️  Stopped by user")
                break
            elif key == ord("p"):
                paused = not paused
                print("⏸️  Paused" if paused else "▶️  Resumed")
            elif key == ord("s"):
                screenshot_name = f"tvsi_screenshot_{frame_id}.jpg"
                cv2.imwrite(screenshot_name, final_frame)
                print(f"📸 Screenshot saved: {screenshot_name}")
            elif key == ord("t"):
                latest = tvsi_integration.get_latest_result()
                if latest:
                    print("\n" + "="*70)
                    print("📊 TVSI STATUS")
                    print("="*70)
                    print(f"TVSI: {latest['tvsi']:.3f}")
                    print(f"State: {latest['state']}")
                    print(f"Severity: {latest['severity']}")
                    print(f"Trend: {latest['trend']}")
                    print(f"Explanation: {latest['explanation']}")
                    if 'components' in latest:
                        print(f"\nVehicles in ROI: {latest['components']['vehicle_count']}")
                        print(f"Flow: {latest['window_data']['flow']} vehicles/window")
                        print(f"Avg Speed: {latest['window_data']['avg_speed']:.1f} km/h")
                    print("="*70)
    
    except KeyboardInterrupt:
        print("\n⚠️  Stopped by user")
    
    finally:
        cap.release()
        writer.release()
        cv2.destroyAllWindows()
    
        elapsed_time = time.time() - start_time
    
        # Save data
        if vehicle_data:
            df = pd.DataFrame(vehicle_data)
            df.to_excel(OUTPUT_EXCEL, index=False)
            print(f"\n✅ Data exported to: {OUTPUT_EXCEL}")
        
        if tvsi_results_log:
            tvsi_df = pd.DataFrame(tvsi_results_log)
            tvsi_df.to_csv(OUTPUT_TVSI, index=False)
            print(f"✅ TVSI results exported to: {OUTPUT_TVSI}")
            
            if congestion_events:
                congestion_df = pd.DataFrame(congestion_events)
                congestion_df.to_csv("congestion_events.csv", index=False)
                print(f"✅ Congestion events exported to: congestion_events.csv")
        
        # Summary
        print("\n" + "="*70)
        print("📊 PROCESSING SUMMARY")
        print("="*70)
        print(f"⏱️  Total time: {elapsed_time:.1f}s")
        print(f"🎬 Frames: {frame_id}")
        if elapsed_time > 0:
            print(f"⚡ Average FPS: {frame_id/elapsed_time:.1f}")
        print(f"✅ Video: {OUTPUT_VIDEO}")
        print(f"🚗 Vehicles counted: {vehicle_count}")
        print(f"⚠️  Alerts: {len(alert_log)}")
        
        print("\n📊 Vehicle Types:")
        for vtype, count in type_counts.items():
            print(f"   {vtype}: {count}")
        
        if tvsi_results_log:
            print("\n🧠 TVSI SUMMARY:")
            avg_tvsi = np.mean([r['TVSI'] for r in tvsi_results_log])
            print(f"   Average TVSI: {avg_tvsi:.3f}")
            print(f"   Congestion Events: {len(congestion_events)}")
        
        print("="*70)


if __name__ == "__main__":
    main()