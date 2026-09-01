"""
The kinematics engine.

Contains the MediaPipe Pose bindings, the BaseExercise state machine
("the physics engine"), the three concrete exercise protocols, and the
EXERCISE_REGISTRY that maps a slug to its class.

Depends only on config.py (for the *Config dataclasses) plus third-party
libraries -- it does not import worker.py or app.py, so those two are
free to import EXERCISE_REGISTRY from here without any risk of a cycle.
"""
import time

import mediapipe as mp
import numpy as np

from config import (
    BaseExerciseConfig,
    ElbowFlexionConfig,
    ShoulderAbductionConfig,
    SquatConfig,
)

mp_pose = mp.solutions.pose
mp_draw = mp.solutions.drawing_utils


# ==========================================
# BASE EXERCISE CLASS (THE ENGINE)
# ==========================================
class BaseExercise:
    button_labels = ("Right Arm", "Left Arm")

    def __init__(self, side="right", config: BaseExerciseConfig = None):
        self.cfg = config
        self.side = side.lower()
        self.counter = 0
        self.stage = "calibration_start"
        self.smoothed_angle = None
        self._last_update_time = None
        self.error_counts = {}

        # Universal Timers & Bookkeeping
        self.error_timer_start = None
        self.active_error_msg = ""
        self._calib_start_time = None
        self._calib_transition_start_time = None
        self._calib_flex_start_time = None
        self._calib_flex_hold_start_time = None
        self._calib_sample_count = 0

        # Flexion Hold Accumulators
        self._calib_flex_sums = 0.0
        self._calib_flex_count = 0

        # Baselines
        self.target_ext = 0.0
        self.target_flex = 180.0

    def calculate_angle(self, a, b, c):
        a, b, c = np.array(a), np.array(b), np.array(c)
        ba, bc = a - b, c - b
        cosine_angle = np.dot(ba, bc) / (max(np.linalg.norm(ba), 1e-6) * max(np.linalg.norm(bc), 1e-6))
        return np.degrees(np.arccos(np.clip(cosine_angle, -1.0, 1.0)))

    def _update_smoothed_angle(self, raw_angle):
        now = time.monotonic()
        if self.smoothed_angle is None or self._last_update_time is None:
            self.smoothed_angle = raw_angle
        else:
            dt = max(now - self._last_update_time, 0.0)
            tau = max(self.cfg.smoothing_time_constant_s, 1e-6)
            alpha = 1.0 - np.exp(-dt / tau)
            self.smoothed_angle = (alpha * raw_angle) + ((1 - alpha) * self.smoothed_angle)
        self._last_update_time = now
        return self.smoothed_angle

    def process_frame(self, landmarks, h, w):
        raw_angle, joint_pos = self._get_primary_kinematics(landmarks, h, w)
        if raw_angle is None:
            if not self.stage.startswith("calibration"):
                self.stage = "error"
            return None, None, False, "STAND FULLY IN FRAME!"

        self._update_smoothed_angle(raw_angle)
        is_good_form, current_error = self._evaluate_form(landmarks, h, w)

        if self.stage.startswith("calibration"):
            feedback_msg = self._route_calibration(is_good_form, current_error, landmarks, h, w)
            return self.smoothed_angle, joint_pos, True, feedback_msg

        if not is_good_form:
            if self.stage != "error":
                self.stage = "error"
                self.error_counts[current_error] = self.error_counts.get(current_error, 0) + 1

            self.active_error_msg = current_error
            self.error_timer_start = None
        elif self.stage == "error":
            self.stage = "recovering"
            self.error_timer_start = time.monotonic()

        feedback_msg = ""
        ui_is_good = is_good_form

        if self.stage == "error":
            feedback_msg = f"{self.active_error_msg} (VOIDED)"
            ui_is_good = False
        elif self.stage == "recovering":
            ui_is_good = False
            if self.error_timer_start is not None and (
                    time.monotonic() - self.error_timer_start) < self.cfg.error_display_min_seconds:
                feedback_msg = f"{self.active_error_msg} (VOIDED)"
            else:
                feedback_msg = "RETURN TO START TO RESET"

        # Let the physics engine run so the state machine can transition out of 'recovering'
        if is_good_form:
            self._update_rep_counter()

        return self.smoothed_angle, joint_pos, ui_is_good, feedback_msg

    def _get_primary_kinematics(self, landmarks, h, w):
        raise NotImplementedError

    def _evaluate_form(self, landmarks, h, w):
        raise NotImplementedError

    def _route_calibration(self, is_good_form, current_error, landmarks, h, w):
        raise NotImplementedError

    def _update_rep_counter(self):
        raise NotImplementedError


# ==========================================
# ELBOW FLEXION PROTOCOL
# ==========================================
class ElbowFlexion(BaseExercise):
    display_name = "Elbow Flexion"
    slug = "elbow_flexion"

    def __init__(self, side="right", config: ElbowFlexionConfig = None):
        super().__init__(side, config or ElbowFlexionConfig())
        self.baselines = {'wrist_z': 0.0, 'elbow_z': 0.0, 'upper_ratio': 0.0, 'forearm_ratio': 0.0}
        self._calib_sums = {'wrist_z': 0.0, 'elbow_z': 0.0, 'upper_ratio': 0.0, 'forearm_ratio': 0.0, 'angle': 0.0}

        # New smoothing structure for secondary kinematic signals
        self.smoothed = {'wrist_z': None, 'elbow_z': None, 'upper_ratio': None, 'forearm_ratio': None}
        self._last_smooth_time = None

    def _get_primary_kinematics(self, landmarks, h, w):
        req_vis = self.cfg.min_visibility_calibration if self.stage.startswith(
            "calib") else self.cfg.min_visibility_tracking
        idx = mp_pose.PoseLandmark
        s_idx, e_idx, w_idx = (idx.RIGHT_SHOULDER, idx.RIGHT_ELBOW, idx.RIGHT_WRIST) if self.side == "right" else (
            idx.LEFT_SHOULDER, idx.LEFT_ELBOW, idx.LEFT_WRIST)

        if min(landmarks[s_idx.value].visibility, landmarks[e_idx.value].visibility,
               landmarks[w_idx.value].visibility) < req_vis:
            return None, None

        shoulder = [landmarks[s_idx.value].x * w, landmarks[s_idx.value].y * h]
        elbow = [landmarks[e_idx.value].x * w, landmarks[e_idx.value].y * h]
        wrist = [landmarks[w_idx.value].x * w, landmarks[w_idx.value].y * h]
        return self.calculate_angle(shoulder, elbow, wrist), elbow

    def _evaluate_form(self, landmarks, h, w):
        idx = mp_pose.PoseLandmark
        s_idx, e_idx, w_idx, h_idx = (idx.RIGHT_SHOULDER, idx.RIGHT_ELBOW, idx.RIGHT_WRIST,
                                      idx.RIGHT_HIP) if self.side == "right" else (idx.LEFT_SHOULDER, idx.LEFT_ELBOW,
                                                                                   idx.LEFT_WRIST, idx.LEFT_HIP)
        opp_s_idx = idx.LEFT_SHOULDER if self.side == "right" else idx.RIGHT_SHOULDER
        t_shoulder, t_elbow, t_wrist, t_hip = landmarks[s_idx.value], landmarks[e_idx.value], landmarks[w_idx.value], \
            landmarks[h_idx.value]

        shoulder = [t_shoulder.x * w, t_shoulder.y * h]
        elbow = [t_elbow.x * w, t_elbow.y * h]
        wrist = [t_wrist.x * w, t_wrist.y * h]
        hip = [t_hip.x * w, t_hip.y * h]
        opp_shoulder = [landmarks[opp_s_idx.value].x * w, landmarks[opp_s_idx.value].y * h]

        torso_length = max(np.linalg.norm(np.array(shoulder) - np.array(hip)), 1e-6)
        shoulder_width = abs(shoulder[0] - opp_shoulder[0])

        upper_len = np.linalg.norm(np.array(shoulder) - np.array(elbow))
        forearm_len = np.linalg.norm(np.array(elbow) - np.array(wrist))

        # Apply EMA smoothing to noisy Z and Ratio constraints
        now = time.monotonic()
        raw_signals = {
            'wrist_z': abs(t_wrist.z - t_shoulder.z),
            'elbow_z': abs(t_elbow.z - t_shoulder.z),
            'upper_ratio': upper_len / torso_length,
            'forearm_ratio': forearm_len / torso_length
        }

        if self._last_smooth_time is None or any(v is None for v in self.smoothed.values()):
            self.smoothed.update(raw_signals)
        else:
            dt = max(now - self._last_smooth_time, 0.0)
            tau = max(self.cfg.smoothing_time_constant_s, 1e-6)
            alpha = 1.0 - np.exp(-dt / tau)
            for k in raw_signals:
                self.smoothed[k] = (alpha * raw_signals[k]) + ((1 - alpha) * self.smoothed[k])
        self._last_smooth_time = now

        # Priority 1: Profile Alignment
        if shoulder_width > (self.cfg.side_profile_max_shoulder_ratio * torso_length):
            return False, "TURN TO SIDE PROFILE!"

        if self.stage.startswith("calib"): return True, ""

        # Priority 2: Posture (Root Cause)
        if abs(shoulder[0] - hip[0]) > (self.cfg.trunk_sway_max_ratio * torso_length):
            return False, "TRUNK SWAY!"

        # Priority 3: Limb Vectors (Symptoms)
        if elbow[1] < shoulder[1]:
            return False, "KEEP ELBOW PINNED!"
        if np.degrees(np.arctan2(abs(elbow[0] - shoulder[0]),
                                 max(elbow[1] - shoulder[1], 1))) > self.cfg.pinned_elbow_max_angle_deg:
            return False, "KEEP ELBOW PINNED!"

        # Priority 4: Arm Swinging Out (Uses smoothed variables)
        is_upper_stable = self.smoothed['upper_ratio'] > (self.baselines['upper_ratio'] * self.cfg.stability_ratio)
        is_forearm_stable = self.smoothed['forearm_ratio'] > (
                    self.baselines['forearm_ratio'] * self.cfg.stability_ratio)

        z_ratio = self.cfg.z_stability_ratio
        w_lim = max(self.baselines['wrist_z'], self.cfg.wrist_z_max * z_ratio) / z_ratio if self.baselines[
                                                                                                'wrist_z'] > 0 else self.cfg.wrist_z_max
        e_lim = max(self.baselines['elbow_z'], self.cfg.elbow_z_max * z_ratio) / z_ratio if self.baselines[
                                                                                                'elbow_z'] > 0 else self.cfg.elbow_z_max

        if (self.smoothed['wrist_z'] > w_lim or self.smoothed['elbow_z'] > e_lim) or not (
                is_upper_stable and is_forearm_stable):
            return False, "ARM SWINGING OUT!"

        return True, ""

    def _route_calibration(self, is_good, err_msg, landmarks, h, w):
        now = time.monotonic()

        idx = mp_pose.PoseLandmark
        s_idx = idx.RIGHT_SHOULDER if self.side == "right" else idx.LEFT_SHOULDER
        current_shoulder = [landmarks[s_idx.value].x * w, landmarks[s_idx.value].y * h]

        if not hasattr(self, '_prev_shoulder'):
            self._prev_shoulder = current_shoulder

        motion = np.linalg.norm(np.array(current_shoulder) - np.array(self._prev_shoulder))
        self._prev_shoulder = current_shoulder

        if not is_good:
            self._calib_start_time = None
            self._calib_sums = {k: 0.0 for k in self._calib_sums}
            self._calib_sample_count = 0
            return err_msg

        if self.stage == "calibration_start":
            if self.smoothed_angle < self.cfg.extension_angle_min_deg:
                self._calib_start_time = None
                self._calib_sums = {k: 0.0 for k in self._calib_sums}
                self._calib_sample_count = 0
                return "DROP ARM STRAIGHT TO START"

            h_idx = idx.RIGHT_HIP if self.side == "right" else idx.LEFT_HIP
            hip = [landmarks[h_idx.value].x * w, landmarks[h_idx.value].y * h]
            torso_len = max(np.linalg.norm(np.array(current_shoulder) - np.array(hip)), 1e-6)

            if motion > (0.025 * torso_len):
                self._calib_start_time = now
                self._calib_sums = {k: 0.0 for k in self._calib_sums}
                self._calib_sample_count = 0
                return "STAND COMPLETELY STILL!"

            if self._calib_start_time is None: self._calib_start_time = now

            e_idx, w_idx = (idx.RIGHT_ELBOW, idx.RIGHT_WRIST) if self.side == "right" else (idx.LEFT_ELBOW,
                                                                                            idx.LEFT_WRIST)
            t_s, t_e, t_w = landmarks[s_idx.value], landmarks[e_idx.value], landmarks[w_idx.value]

            s_px = [t_s.x * w, t_s.y * h]
            e_px = [t_e.x * w, t_e.y * h]
            w_px = [t_w.x * w, t_w.y * h]

            upper_len = np.linalg.norm(np.array(s_px) - np.array(e_px))
            forearm_len = np.linalg.norm(np.array(e_px) - np.array(w_px))

            self._calib_sums['upper_ratio'] += upper_len / torso_len
            self._calib_sums['forearm_ratio'] += forearm_len / torso_len
            self._calib_sums['wrist_z'] += abs(t_w.z - t_s.z)
            self._calib_sums['elbow_z'] += abs(t_e.z - t_s.z)
            self._calib_sums['angle'] += self.smoothed_angle
            self._calib_sample_count += 1

            elapsed = now - self._calib_start_time
            if elapsed >= self.cfg.calib_extension_seconds:
                n = self._calib_sample_count

                if n < 15:
                    self._calib_start_time = None
                    self._calib_sums = {k: 0.0 for k in self._calib_sums}
                    self._calib_sample_count = 0
                    return "KEEP BODY FULLY VISIBLE!"

                self.baselines['wrist_z'] = self._calib_sums['wrist_z'] / n
                self.baselines['elbow_z'] = self._calib_sums['elbow_z'] / n
                self.baselines['upper_ratio'] = self._calib_sums['upper_ratio'] / n
                self.baselines['forearm_ratio'] = self._calib_sums['forearm_ratio'] / n
                self.target_ext = (self._calib_sums['angle'] / n) - self.cfg.ext_buffer_deg
                self.stage = "calibration_transition"
            return f"STAND STILL: {max(0, int(self.cfg.calib_extension_seconds - elapsed) + 1)}s"

        elif self.stage == "calibration_transition":
            if self._calib_transition_start_time is None: self._calib_transition_start_time = now
            if (now - self._calib_transition_start_time) > self.cfg.calib_transition_seconds:
                self.stage = "calibration_flex"
            return "SUCCESS! NOW BEND ARM..."

        elif self.stage == "calibration_flex":
            if self._calib_flex_start_time is None: self._calib_flex_start_time = now
            elapsed_total = now - self._calib_flex_start_time

            if self.smoothed_angle < (self.target_ext - self.cfg.min_rom_deg):
                if self._calib_flex_hold_start_time is None:
                    self._calib_flex_hold_start_time = now
                    self._calib_flex_sums = 0.0
                    self._calib_flex_count = 0

                self._calib_flex_sums += self.smoothed_angle
                self._calib_flex_count += 1
                elapsed_hold = now - self._calib_flex_hold_start_time

                if elapsed_hold >= self.cfg.calib_flex_seconds:
                    avg_flex = self._calib_flex_sums / self._calib_flex_count
                    self.target_flex = avg_flex + self.cfg.flex_buffer_deg
                    self.stage = "calibration_returning"
                    return "CALIBRATION COMPLETE!"

                seconds_left = max(0, int(self.cfg.calib_flex_seconds - elapsed_hold) + 1)
                return f"HOLD THIS FLEX: {seconds_left}s"
            else:
                self._calib_flex_hold_start_time = None
                self._calib_flex_sums = 0.0
                self._calib_flex_count = 0

                if elapsed_total >= self.cfg.calib_flex_timeout_seconds:
                    self.stage = "calibration_transition"
                    self._calib_transition_start_time = None
                    return "TIMEOUT: PLEASE TRY AGAIN"
                return "BEND ARM PAST MINIMUM LIMIT"

        elif self.stage == "calibration_returning":
            if self.smoothed_angle > self.target_ext:
                self.stage = "extended"
                return "START WORKOUT!"
            return "RETURN TO START..."

    def _update_rep_counter(self):
        if self.smoothed_angle > self.target_ext:
            if self.stage == 'flexed':
                self.counter += 1
            self.stage = "extended"
        elif self.smoothed_angle < self.target_flex and self.stage == 'extended':
            self.stage = "flexed"


# ==========================================
# SHOULDER ABDUCTION PROTOCOL
# ==========================================
class ShoulderAbduction(BaseExercise):
    display_name = "Shoulder Abduction"
    slug = "shoulder_abduction"

    def __init__(self, side="right", config: ShoulderAbductionConfig = None):
        super().__init__(side, config or ShoulderAbductionConfig())
        self.baselines = {'shrug_ratio': 0.0, 'trunk_x_ratio': 0.0, 'upper_ratio': 0.0}
        self._calib_sums = {'angle': 0.0, 'shrug_ratio': 0.0, 'trunk_x_ratio': 0.0, 'upper_ratio': 0.0}

        self.smoothed = {'trunk_x_ratio': None, 'upper_ratio': None, 'elbow_z': None, 'shrug_ratio': None}
        self._last_smooth_time = None

    def _get_primary_kinematics(self, landmarks, h, w):
        req_vis = self.cfg.min_visibility_calibration if self.stage.startswith(
            "calib") else self.cfg.min_visibility_tracking
        idx = mp_pose.PoseLandmark
        h_idx, s_idx, e_idx = (idx.RIGHT_HIP, idx.RIGHT_SHOULDER, idx.RIGHT_ELBOW) if self.side == "right" else (
            idx.LEFT_HIP, idx.LEFT_SHOULDER, idx.LEFT_ELBOW)

        if min(landmarks[s_idx.value].visibility, landmarks[e_idx.value].visibility,
               landmarks[h_idx.value].visibility) < req_vis:
            return None, None

        hip = [landmarks[h_idx.value].x * w, landmarks[h_idx.value].y * h]
        shoulder = [landmarks[s_idx.value].x * w, landmarks[s_idx.value].y * h]
        elbow = [landmarks[e_idx.value].x * w, landmarks[e_idx.value].y * h]
        return self.calculate_angle(hip, shoulder, elbow), shoulder

    def _evaluate_form(self, landmarks, h, w):
        idx = mp_pose.PoseLandmark
        s_idx, e_idx, h_idx = (idx.RIGHT_SHOULDER, idx.RIGHT_ELBOW, idx.RIGHT_HIP) if self.side == "right" else (
            idx.LEFT_SHOULDER, idx.LEFT_ELBOW, idx.LEFT_HIP)
        opp_s_idx = idx.LEFT_SHOULDER if self.side == "right" else idx.RIGHT_SHOULDER
        n_idx = mp_pose.PoseLandmark.NOSE

        t_shoulder, t_elbow, t_hip, t_nose = landmarks[s_idx.value], landmarks[e_idx.value], landmarks[h_idx.value], \
            landmarks[n_idx.value]

        shoulder = [t_shoulder.x * w, t_shoulder.y * h]
        elbow = [t_elbow.x * w, t_elbow.y * h]
        hip = [t_hip.x * w, t_hip.y * h]
        nose = [t_nose.x * w, t_nose.y * h]
        opp_shoulder = [landmarks[opp_s_idx.value].x * w, landmarks[opp_s_idx.value].y * h]

        torso_length = max(np.linalg.norm(np.array(shoulder) - np.array(hip)), 1e-6)
        shoulder_width = abs(shoulder[0] - opp_shoulder[0])
        upper_len = np.linalg.norm(np.array(shoulder) - np.array(elbow))

        now = time.monotonic()
        raw_signals = {
            'trunk_x_ratio': abs(nose[0] - hip[0]) / torso_length,
            'upper_ratio': upper_len / torso_length,
            'elbow_z': abs(t_elbow.z - t_shoulder.z),
            'shrug_ratio': abs(shoulder[1] - nose[1]) / torso_length
        }

        if self._last_smooth_time is None or any(v is None for v in self.smoothed.values()):
            self.smoothed.update(raw_signals)
        else:
            dt = max(now - self._last_smooth_time, 0.0)
            tau = max(self.cfg.smoothing_time_constant_s, 1e-6)
            alpha = 1.0 - np.exp(-dt / tau)
            for k in raw_signals:
                self.smoothed[k] = (alpha * raw_signals[k]) + ((1 - alpha) * self.smoothed[k])
        self._last_smooth_time = now

        # Priority 1: Camera Alignment
        if shoulder_width < (self.cfg.frontal_profile_min_shoulder_ratio * torso_length):
            return False, "FACE THE CAMERA DIRECTLY!"

        if self.stage.startswith("calib"): return True, ""

        # Priority 2: Posture (Sway)
        sway_limit = self.baselines['trunk_x_ratio'] + self.cfg.trunk_sway_max_ratio
        if self.smoothed['trunk_x_ratio'] > sway_limit:
            return False, "KEEP TORSO STRAIGHT!"

        # Priority 3: Arm Vectors (Foreshortening & Z-axis Guard)
        if self.smoothed['upper_ratio'] < (self.baselines.get('upper_ratio', 0) * self.cfg.stability_ratio):
            return False, "LIFT TO SIDE, NOT FORWARD!"

        if self.smoothed['elbow_z'] > self.cfg.elbow_z_max_diff:
            return False, "LIFT TO SIDE, NOT FORWARD!"

        # Priority 4: Posture (Shrug)
        shrug_limit = self.baselines['shrug_ratio'] - self.cfg.shrug_max_ratio
        if self.smoothed['shrug_ratio'] < shrug_limit:
            return False, "DO NOT SHRUG!"

        return True, ""

    def _route_calibration(self, is_good, err_msg, landmarks, h, w):
        now = time.monotonic()

        idx = mp_pose.PoseLandmark
        s_idx = idx.RIGHT_SHOULDER if self.side == "right" else idx.LEFT_SHOULDER
        current_shoulder = [landmarks[s_idx.value].x * w, landmarks[s_idx.value].y * h]

        if not hasattr(self, '_prev_shoulder'):
            self._prev_shoulder = current_shoulder

        motion = np.linalg.norm(np.array(current_shoulder) - np.array(self._prev_shoulder))
        self._prev_shoulder = current_shoulder

        if not is_good:
            self._calib_start_time = None
            self._calib_sums = {k: 0.0 for k in self._calib_sums}
            self._calib_sample_count = 0
            return err_msg

        if self.stage == "calibration_start":
            if self.smoothed_angle > self.cfg.extension_angle_max_deg:
                self._calib_start_time = None
                self._calib_sums = {k: 0.0 for k in self._calib_sums}
                self._calib_sample_count = 0
                return "DROP ARM STRAIGHT DOWN"

            h_idx = idx.RIGHT_HIP if self.side == "right" else idx.LEFT_HIP
            hip = [landmarks[h_idx.value].x * w, landmarks[h_idx.value].y * h]
            torso_length = max(np.linalg.norm(np.array(current_shoulder) - np.array(hip)), 1e-6)

            if motion > (0.025 * torso_length):
                self._calib_start_time = now
                self._calib_sums = {k: 0.0 for k in self._calib_sums}
                self._calib_sample_count = 0
                return "STAND COMPLETELY STILL!"

            if self._calib_start_time is None: self._calib_start_time = now

            e_idx = idx.RIGHT_ELBOW if self.side == "right" else idx.LEFT_ELBOW
            elbow = [landmarks[e_idx.value].x * w, landmarks[e_idx.value].y * h]

            n_idx = mp_pose.PoseLandmark.NOSE
            nose = [landmarks[n_idx.value].x * w, landmarks[n_idx.value].y * h]

            self._calib_sums['angle'] += self.smoothed_angle
            self._calib_sums['shrug_ratio'] += abs(current_shoulder[1] - nose[1]) / torso_length
            self._calib_sums['trunk_x_ratio'] += abs(nose[0] - hip[0]) / torso_length

            upper_len = np.linalg.norm(np.array(current_shoulder) - np.array(elbow))
            self._calib_sums['upper_ratio'] += upper_len / torso_length

            self._calib_sample_count += 1

            elapsed = now - self._calib_start_time
            if elapsed >= self.cfg.calib_extension_seconds:
                n = self._calib_sample_count

                if n < 15:
                    self._calib_start_time = None
                    self._calib_sums = {k: 0.0 for k in self._calib_sums}
                    self._calib_sample_count = 0
                    return "KEEP BODY FULLY VISIBLE!"

                self.target_ext = (self._calib_sums['angle'] / n) + self.cfg.ext_buffer_deg
                self.baselines['shrug_ratio'] = self._calib_sums['shrug_ratio'] / n
                self.baselines['trunk_x_ratio'] = self._calib_sums['trunk_x_ratio'] / n
                self.baselines['upper_ratio'] = self._calib_sums['upper_ratio'] / n
                self.stage = "calibration_transition"
            return f"REST ARM DOWN: {max(0, int(self.cfg.calib_extension_seconds - elapsed) + 1)}s"

        elif self.stage == "calibration_transition":
            if self._calib_transition_start_time is None: self._calib_transition_start_time = now
            if (now - self._calib_transition_start_time) > self.cfg.calib_transition_seconds:
                self.stage = "calibration_flex"
            return "SUCCESS! NOW RAISE ARM..."

        elif self.stage == "calibration_flex":
            if self._calib_flex_start_time is None: self._calib_flex_start_time = now
            elapsed_total = now - self._calib_flex_start_time

            if self.smoothed_angle > (self.target_ext + self.cfg.min_rom_deg):
                if self._calib_flex_hold_start_time is None:
                    self._calib_flex_hold_start_time = now
                    self._calib_flex_sums = 0.0
                    self._calib_flex_count = 0

                self._calib_flex_sums += self.smoothed_angle
                self._calib_flex_count += 1
                elapsed_hold = now - self._calib_flex_hold_start_time

                if elapsed_hold >= self.cfg.calib_flex_seconds:
                    avg_flex = self._calib_flex_sums / self._calib_flex_count
                    self.target_flex = avg_flex - self.cfg.flex_buffer_deg
                    self.stage = "calibration_returning"
                    return "CALIBRATION COMPLETE!"

                seconds_left = max(0, int(self.cfg.calib_flex_seconds - elapsed_hold) + 1)
                return f"HOLD ARM UP: {seconds_left}s"
            else:
                self._calib_flex_hold_start_time = None
                self._calib_flex_sums = 0.0
                self._calib_flex_count = 0

                if elapsed_total >= self.cfg.calib_flex_timeout_seconds:
                    self.stage = "calibration_transition"
                    self._calib_transition_start_time = None
                    return "TIMEOUT: PLEASE TRY AGAIN"
                return "RAISE ARM PAST MINIMUM LIMIT"

        elif self.stage == "calibration_returning":
            if self.smoothed_angle < self.target_ext:
                self.stage = "extended"
                return "START WORKOUT!"
            return "RETURN TO START..."

    def _update_rep_counter(self):
        if self.smoothed_angle < self.target_ext:
            if self.stage == 'flexed':
                self.counter += 1
            self.stage = "extended"
        elif self.smoothed_angle > self.target_flex and self.stage == 'extended':
            self.stage = "flexed"


# ==========================================
# SQUAT PROTOCOL
# ==========================================
class Squat(BaseExercise):
    display_name = "Squat"
    slug = "squat"

    # Override the labels specifically for sagittal tracking
    button_labels = ("Camera on Right Side", "Camera on Left Side")

    def __init__(self, side="right", config: SquatConfig = None):
        super().__init__(side, config or SquatConfig())

        self._calib_sums = {'angle': 0.0}
        self.smoothed = {'shoulder_width_ratio': None, 'trunk_x_ratio': None, 'knee_x_ratio': None}
        self._last_smooth_time = None

    def _get_primary_kinematics(self, landmarks, h, w):
        req_vis = self.cfg.min_visibility_calibration if self.stage.startswith(
            "calib") else self.cfg.min_visibility_tracking
        idx = mp_pose.PoseLandmark

        h_idx, k_idx, a_idx = (idx.RIGHT_HIP, idx.RIGHT_KNEE, idx.RIGHT_ANKLE) if self.side == "right" else \
            (idx.LEFT_HIP, idx.LEFT_KNEE, idx.LEFT_ANKLE)

        # Visibility filter to prevent tracking hallucinations
        if min(landmarks[h_idx.value].visibility, landmarks[k_idx.value].visibility,
               landmarks[a_idx.value].visibility) < req_vis:
            return None, None

        hip = [landmarks[h_idx.value].x * w, landmarks[h_idx.value].y * h]
        knee = [landmarks[k_idx.value].x * w, landmarks[k_idx.value].y * h]
        ankle = [landmarks[a_idx.value].x * w, landmarks[a_idx.value].y * h]

        return self.calculate_angle(hip, knee, ankle), knee

    def _evaluate_form(self, landmarks, h, w):
        idx = mp_pose.PoseLandmark
        s_idx, h_idx, k_idx = (idx.RIGHT_SHOULDER, idx.RIGHT_HIP, idx.RIGHT_KNEE) if self.side == "right" else \
            (idx.LEFT_SHOULDER, idx.LEFT_HIP, idx.LEFT_KNEE)

        t_idx = idx.RIGHT_FOOT_INDEX if self.side == "right" else idx.LEFT_FOOT_INDEX
        opp_s_idx = idx.LEFT_SHOULDER if self.side == "right" else idx.RIGHT_SHOULDER

        t_shoulder = landmarks[s_idx.value]
        t_hip = landmarks[h_idx.value]
        t_knee = landmarks[k_idx.value]
        t_toe = landmarks[t_idx.value]
        t_opp_shoulder = landmarks[opp_s_idx.value]

        # NEW: The Hidden Toe / Visibility Guard
        req_vis = self.cfg.min_visibility_calibration if self.stage.startswith(
            "calib") else self.cfg.min_visibility_tracking
        if min(t_shoulder.visibility, t_toe.visibility, t_opp_shoulder.visibility) < req_vis:
            return False, "FOOT OR SHOULDER HIDDEN!"

        shoulder = [t_shoulder.x * w, t_shoulder.y * h]
        hip = [t_hip.x * w, t_hip.y * h]
        knee = [t_knee.x * w, t_knee.y * h]
        toe = [t_toe.x * w, t_toe.y * h]
        opp_shoulder = [t_opp_shoulder.x * w, t_opp_shoulder.y * h]

        torso_length = max(np.linalg.norm(np.array(shoulder) - np.array(hip)), 1e-6)
        facing_dir = 1 if toe[0] > hip[0] else -1
        knee_past_toe_dist = (knee[0] - toe[0]) * facing_dir

        now = time.monotonic()
        raw_signals = {
            'shoulder_width_ratio': abs(shoulder[0] - opp_shoulder[0]) / torso_length,
            'trunk_x_ratio': abs(shoulder[0] - hip[0]) / torso_length,
            'knee_x_ratio': max(0, knee_past_toe_dist) / torso_length
        }

        # Apply time-constant EMA smoothing to eliminate GUI flickering
        if self._last_smooth_time is None or any(v is None for v in self.smoothed.values()):
            self.smoothed.update(raw_signals)
        else:
            dt = max(now - self._last_smooth_time, 0.0)
            tau = max(self.cfg.smoothing_time_constant_s, 1e-6)
            alpha = 1.0 - np.exp(-dt / tau)
            for k in raw_signals:
                self.smoothed[k] = (alpha * raw_signals[k]) + ((1 - alpha) * self.smoothed[k])
        self._last_smooth_time = now

        # Priority 1: Sagittal View Guard
        if self.smoothed['shoulder_width_ratio'] > self.cfg.side_profile_max_shoulder_ratio:
            return False, "TURN TO SIDE PROFILE!"

        if self.stage.startswith("calib"): return True, ""

        # REORDERED TO PREVENT ERROR SHADOWING:

        # Priority 2: Knee Tracking (Horizontal Past Toes)
        if self.smoothed['knee_x_ratio'] > self.cfg.knee_tracking_max_ratio:
            return False, "KNEES PAST TOES!"

        # Priority 3: Postural Compensation (Moved to last)
        if self.smoothed['trunk_x_ratio'] > self.cfg.trunk_lean_max_ratio:
            return False, "TRUNK LEAN TOO FAR FORWARD!"

        return True, ""

    def _route_calibration(self, is_good, err_msg, landmarks, h, w):
        now = time.monotonic()

        idx = mp_pose.PoseLandmark
        s_idx = idx.RIGHT_SHOULDER if self.side == "right" else idx.LEFT_SHOULDER
        current_shoulder = [landmarks[s_idx.value].x * w, landmarks[s_idx.value].y * h]

        if not hasattr(self, '_prev_shoulder'):
            self._prev_shoulder = current_shoulder

        motion = np.linalg.norm(np.array(current_shoulder) - np.array(self._prev_shoulder))
        self._prev_shoulder = current_shoulder

        if not is_good:
            self._calib_start_time = None
            self._calib_sums = {k: 0.0 for k in self._calib_sums}
            self._calib_sample_count = 0
            return err_msg

        # STAGE 1: STANDING CALIBRATION
        if self.stage == "calibration_start":
            if self.smoothed_angle < self.cfg.extension_angle_min_deg:
                self._calib_start_time = None
                self._calib_sums = {k: 0.0 for k in self._calib_sums}
                self._calib_sample_count = 0
                return "STAND STRAIGHT TO START"

            h_idx = idx.RIGHT_HIP if self.side == "right" else idx.LEFT_HIP
            hip = [landmarks[h_idx.value].x * w, landmarks[h_idx.value].y * h]
            torso_length = max(np.linalg.norm(np.array(current_shoulder) - np.array(hip)), 1e-6)

            # Velocity guard against calibration while walking into frame
            if motion > (0.025 * torso_length):
                self._calib_start_time = now
                self._calib_sums = {k: 0.0 for k in self._calib_sums}
                self._calib_sample_count = 0
                return "STAND COMPLETELY STILL!"

            if self._calib_start_time is None: self._calib_start_time = now

            self._calib_sums['angle'] += self.smoothed_angle
            self._calib_sample_count += 1

            elapsed = now - self._calib_start_time
            if elapsed >= self.cfg.calib_extension_seconds:
                n = self._calib_sample_count
                if n < 15:
                    self._calib_start_time = None
                    self._calib_sums = {k: 0.0 for k in self._calib_sums}
                    self._calib_sample_count = 0
                    return "KEEP BODY FULLY VISIBLE!"

                # Secure statistical baselines
                self.target_ext = (self._calib_sums['angle'] / n) - self.cfg.flex_buffer_deg
                self.stage = "calibration_transition"

            return f"STAND STILL: {max(0, int(self.cfg.calib_extension_seconds - elapsed) + 1)}s"

        # STAGE 2: TRANSITION TO DEEP SQUAT
        elif self.stage == "calibration_transition":
            if self._calib_transition_start_time is None: self._calib_transition_start_time = now
            if (now - self._calib_transition_start_time) > self.cfg.calib_transition_seconds:
                self.stage = "calibration_flex"
            return "SUCCESS! NOW SQUAT DOWN..."

        # STAGE 3: ADAPTIVE SQUAT CALIBRATION
        elif self.stage == "calibration_flex":
            if self._calib_flex_start_time is None: self._calib_flex_start_time = now
            elapsed_total = now - self._calib_flex_start_time

            # Guardrail: Ensure they have dropped at least the minimum ROM from standing
            if self.smoothed_angle < (self.target_ext - self.cfg.valid_squat_min_rom):

                # Re-calculate torso length for the velocity guard
                h_idx = idx.RIGHT_HIP if self.side == "right" else idx.LEFT_HIP
                hip = [landmarks[h_idx.value].x * w, landmarks[h_idx.value].y * h]
                torso_length = max(np.linalg.norm(np.array(current_shoulder) - np.array(hip)), 1e-6)

                # Velocity guard: Ensure they have reached the bottom and are holding still
                if motion < (0.025 * torso_length):
                    if self._calib_flex_hold_start_time is None:
                        self._calib_flex_hold_start_time = now
                        self._calib_flex_sums = 0.0
                        self._calib_flex_count = 0

                    self._calib_flex_sums += self.smoothed_angle
                    self._calib_flex_count += 1
                    elapsed_hold = now - self._calib_flex_hold_start_time

                    if elapsed_hold >= self.cfg.calib_flex_seconds:
                        avg_flex = self._calib_flex_sums / max(1, self._calib_flex_count)
                        self.target_flex = avg_flex + self.cfg.flex_buffer_deg
                        self.stage = "calibration_returning"
                        return "MAX DEPTH SAVED!"

                    seconds_left = max(0, int(self.cfg.calib_flex_seconds - elapsed_hold) + 1)
                    return f"HOLD SQUAT DEPTH: {seconds_left}s"
                else:
                    # Patient is still descending or wobbling
                    self._calib_flex_hold_start_time = None
                    self._calib_flex_sums = 0.0
                    self._calib_flex_count = 0
                    return "REACH MAX DEPTH AND HOLD STILL"
            else:
                self._calib_flex_hold_start_time = None
                self._calib_flex_sums = 0.0
                self._calib_flex_count = 0

                if elapsed_total >= self.cfg.calib_flex_timeout_seconds:
                    self.stage = "calibration_transition"
                    self._calib_transition_start_time = None
                    return "TIMEOUT: PLEASE TRY AGAIN"
                return "SQUAT DEEPER TO CALIBRATE"

        # STAGE 4: RETURN TO BASELINE
        elif self.stage == "calibration_returning":
            if self.smoothed_angle > self.target_ext:
                self.stage = "extended"
                return "START WORKOUT!"
            return "RETURN TO STANDING..."

    def _update_rep_counter(self):
        # Angle increases when standing (extended), decreases when squatting (flexed)
        if self.smoothed_angle > self.target_ext:
            if self.stage == 'flexed':
                self.counter += 1
            self.stage = "extended"
        elif self.smoothed_angle < self.target_flex and self.stage == 'extended':
            self.stage = "flexed"


# ==========================================
# REGISTRY
# ==========================================
EXERCISE_REGISTRY = {
    ElbowFlexion.slug: ElbowFlexion,
    ShoulderAbduction.slug: ShoulderAbduction,
    Squat.slug: Squat,
}
