"""
Configuration dataclasses.

This module has zero internal dependencies (no imports from engine.py,
worker.py, or app.py), which is what lets every other module import
config.py freely without risking a circular import.
"""
from dataclasses import dataclass


@dataclass
class AppConfig:
    camera_index: int = 0
    inference_max_dim: int = 480
    target_fps: float = 30.0
    queue_max_size: int = 1


@dataclass
class BaseExerciseConfig:
    error_display_min_seconds: float = 1.5
    calib_extension_seconds: float = 3.0
    calib_transition_seconds: float = 1.5
    calib_flex_seconds: float = 2.0
    calib_flex_timeout_seconds: float = 15.0
    min_visibility_tracking: float = 0.4
    min_visibility_calibration: float = 0.7
    smoothing_time_constant_s: float = 0.15


@dataclass
class ElbowFlexionConfig(BaseExerciseConfig):
    side_profile_max_shoulder_ratio: float = 0.35
    wrist_z_max: float = 0.40
    elbow_z_max: float = 0.30
    pinned_elbow_max_angle_deg: float = 20.0
    trunk_sway_max_ratio: float = 0.20
    extension_angle_min_deg: float = 140.0
    min_rom_deg: float = 40.0
    z_stability_ratio: float = 0.75
    stability_ratio: float = 0.75
    ext_buffer_deg: float = 15.0
    flex_buffer_deg: float = 15.0


@dataclass
class ShoulderAbductionConfig(BaseExerciseConfig):
    frontal_profile_min_shoulder_ratio: float = 0.40
    elbow_z_max_diff: float = 0.30
    wrist_z_max_diff: float = 0.40
    trunk_sway_max_ratio: float = 0.08
    extension_angle_max_deg: float = 30.0
    shrug_max_ratio: float = 0.12
    min_rom_deg: float = 20.0
    flex_buffer_deg: float = 10.0
    stability_ratio: float = 0.75
    ext_buffer_deg: float = 15.0


@dataclass
class SquatConfig(BaseExerciseConfig):
    """Configuration parameters for the scale-invariant Squat protocol."""
    side_profile_max_shoulder_ratio: float = 0.35
    trunk_lean_max_ratio: float = 0.80
    knee_tracking_max_ratio: float = 0.15
    extension_angle_min_deg: float = 160.0
    valid_squat_min_rom: float = 30.0
    flex_buffer_deg: float = 15.0
