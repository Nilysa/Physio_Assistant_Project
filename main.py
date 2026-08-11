import cv2
import csv
import datetime
import logging
import os
import queue
import threading
import time
import sys
import mediapipe as mp
import numpy as np
import tkinter as tk
from dataclasses import dataclass
from tkinter import font
from PIL import Image, ImageTk

# ==========================================
# GLOBALS & DEPENDENCIES
# ==========================================
mp_pose = mp.solutions.pose
mp_draw = mp.solutions.drawing_utils

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(threadName)s: %(message)s",
)
logger = logging.getLogger("physio_assistant")

COLOR_GOOD_RGB = (46, 204, 113)  # green
COLOR_BAD_RGB = (231, 76, 60)  # red

SESSIONS_DIR = "sessions"
CSV_FLUSH_EVERY_N_FRAMES = 15
TARGET_FPS = 30.0


# ==========================================
# 1. CONFIG
# ==========================================
@dataclass
class ElbowFlexionConfig:
    stability_ratio: float = 0.75
    side_profile_max_shoulder_ratio: float = 0.35
    wrist_z_max: float = 0.40
    elbow_z_max: float = 0.30
    pinned_elbow_max_angle_deg: float = 20.0
    trunk_sway_max_ratio: float = 0.20
    error_display_min_seconds: float = 1.5
    calib_extension_seconds: float = 3.0
    calib_transition_seconds: float = 1.5
    calib_flex_seconds: float = 2.0
    calib_flex_timeout_seconds: float = 15.0
    extension_angle_min_deg: float = 140.0
    flexion_angle_max_deg: float = 60.0
    min_visibility_tracking: float = 0.4
    min_visibility_calibration: float = 0.7
    smoothing_time_constant_s: float = 0.15
    z_stability_ratio: float = 0.75


# ==========================================
# 2. BASE EXERCISE CLASS
# ==========================================
class BaseExercise:
    def __init__(self):
        self.counter = 0
        self.stage = "extended"
        self.smoothed_angle = None
        self.smoothing_time_constant_s = 0.15
        self._last_update_time = None

    def calculate_angle(self, a, b, c):
        a = np.array(a)
        b = np.array(b)
        c = np.array(c)

        ba = a - b
        bc = c - b

        norm_ba = max(np.linalg.norm(ba), 1e-6)
        norm_bc = max(np.linalg.norm(bc), 1e-6)

        cosine_angle = np.dot(ba, bc) / (norm_ba * norm_bc)
        cosine_angle = np.clip(cosine_angle, -1.0, 1.0)

        angle = np.arccos(cosine_angle)
        return np.degrees(angle)

    def _update_smoothed_angle(self, raw_angle):
        now = time.monotonic()
        if self.smoothed_angle is None or self._last_update_time is None:
            self.smoothed_angle = raw_angle
        else:
            dt = max(now - self._last_update_time, 0.0)
            tau = max(self.smoothing_time_constant_s, 1e-6)
            alpha = 1.0 - np.exp(-dt / tau)
            self.smoothed_angle = (alpha * raw_angle) + ((1 - alpha) * self.smoothed_angle)
        self._last_update_time = now
        return self.smoothed_angle

    def process_frame(self, landmarks, h, w):
        raise NotImplementedError("Subclasses must implement this method")


# ==========================================
# 3. SPECIFIC EXERCISE CLASS (ELBOW FLEXION)
# ==========================================
class ElbowFlexion(BaseExercise):
    display_name = "Elbow Flexion"
    slug = "elbow_flexion"

    def __init__(self, side="right", config: ElbowFlexionConfig = None):
        super().__init__()
        self.stage = "calibration_start"
        self.side = side.lower()
        self.cfg = config or ElbowFlexionConfig()
        self.smoothing_time_constant_s = self.cfg.smoothing_time_constant_s

        self.baseline_upper_ratio = 0.0
        self.baseline_forearm_ratio = 0.0
        self.baseline_wrist_z_diff = 0.0
        self.baseline_elbow_z_diff = 0.0
        self.target_ext = 0.0
        self.target_flex = 180.0

        self._calib_ext_sum_upper_ratio = 0.0
        self._calib_ext_sum_forearm_ratio = 0.0
        self._calib_ext_sum_wrist_z_diff = 0.0
        self._calib_ext_sum_elbow_z_diff = 0.0
        self._calib_ext_sum_angle = 0.0
        self._calib_ext_sample_count = 0

        self._calib_start_time = None
        self._calib_transition_start_time = None
        self._calib_flex_start_time = None
        self._calib_flex_hold_start_time = None

        self.active_error_msg = ""
        self.error_timer_start = None

    def process_frame(self, landmarks, h, w):
        cfg = self.cfg

        if self.side == "right":
            shoulder_idx = mp_pose.PoseLandmark.RIGHT_SHOULDER.value
            elbow_idx = mp_pose.PoseLandmark.RIGHT_ELBOW.value
            wrist_idx = mp_pose.PoseLandmark.RIGHT_WRIST.value
            hip_idx = mp_pose.PoseLandmark.RIGHT_HIP.value
            opp_shoulder_idx = mp_pose.PoseLandmark.LEFT_SHOULDER.value
        else:
            shoulder_idx = mp_pose.PoseLandmark.LEFT_SHOULDER.value
            elbow_idx = mp_pose.PoseLandmark.LEFT_ELBOW.value
            wrist_idx = mp_pose.PoseLandmark.LEFT_WRIST.value
            hip_idx = mp_pose.PoseLandmark.LEFT_HIP.value
            opp_shoulder_idx = mp_pose.PoseLandmark.RIGHT_SHOULDER.value

        t_shoulder = landmarks[shoulder_idx]
        t_elbow = landmarks[elbow_idx]
        t_wrist = landmarks[wrist_idx]
        t_hip = landmarks[hip_idx]
        opp_shoulder = landmarks[opp_shoulder_idx]

        feedback_msg = ""
        good_form = True
        req_vis = cfg.min_visibility_calibration if self.stage.startswith("calib") else cfg.min_visibility_tracking

        if t_shoulder.visibility > req_vis and t_elbow.visibility > req_vis and t_wrist.visibility > req_vis and t_hip.visibility > req_vis:
            shoulder = [t_shoulder.x * w, t_shoulder.y * h]
            elbow = [t_elbow.x * w, t_elbow.y * h]
            wrist = [t_wrist.x * w, t_wrist.y * h]
            hip = [t_hip.x * w, t_hip.y * h]

            raw_angle = self.calculate_angle(shoulder, elbow, wrist)
            self._update_smoothed_angle(raw_angle)

            upper_arm_length = np.linalg.norm(np.array(shoulder) - np.array(elbow))
            forearm_length = np.linalg.norm(np.array(elbow) - np.array(wrist))
            torso_length = max(np.linalg.norm(np.array(shoulder) - np.array(hip)), 1e-6)

            upper_ratio = upper_arm_length / torso_length
            forearm_ratio = forearm_length / torso_length

            wrist_z_diff = abs(t_wrist.z - t_shoulder.z)
            elbow_z_diff = abs(t_elbow.z - t_shoulder.z)

            # ==========================================
            # PRE-CALIBRATION SIDE-PROFILE GUARD
            # ==========================================
            shoulder_width = abs(t_shoulder.x - opp_shoulder.x) * w
            is_side_profile = shoulder_width < (cfg.side_profile_max_shoulder_ratio * torso_length)

            # ==========================================
            # ROBUST CALIBRATION
            # ==========================================
            if self.stage.startswith("calibration"):
                if not is_side_profile:
                    self._calib_start_time = None
                    self._calib_flex_hold_start_time = None
                    self._calib_ext_sum_upper_ratio = 0.0
                    self._calib_ext_sum_forearm_ratio = 0.0
                    self._calib_ext_sum_wrist_z_diff = 0.0
                    self._calib_ext_sum_elbow_z_diff = 0.0
                    self._calib_ext_sum_angle = 0.0
                    self._calib_ext_sample_count = 0
                    return self.smoothed_angle, elbow, False, "TURN TO SIDE PROFILE!"

                now = time.monotonic()

                if self.stage == "calibration_start":
                    if self.smoothed_angle > cfg.extension_angle_min_deg and wrist[1] > elbow[1]:
                        if self._calib_start_time is None:
                            self._calib_start_time = now

                        self._calib_ext_sum_upper_ratio += upper_ratio
                        self._calib_ext_sum_forearm_ratio += forearm_ratio
                        self._calib_ext_sum_wrist_z_diff += wrist_z_diff
                        self._calib_ext_sum_elbow_z_diff += elbow_z_diff
                        self._calib_ext_sum_angle += self.smoothed_angle
                        self._calib_ext_sample_count += 1

                        elapsed = now - self._calib_start_time
                        seconds_left = max(0, int(cfg.calib_extension_seconds - elapsed) + 1)
                        feedback_msg = f"STAND STILL: {seconds_left}s"

                        if elapsed >= cfg.calib_extension_seconds and self._calib_ext_sample_count > 0:
                            n = self._calib_ext_sample_count
                            self.baseline_upper_ratio = self._calib_ext_sum_upper_ratio / n
                            self.baseline_forearm_ratio = self._calib_ext_sum_forearm_ratio / n
                            self.baseline_wrist_z_diff = self._calib_ext_sum_wrist_z_diff / n
                            self.baseline_elbow_z_diff = self._calib_ext_sum_elbow_z_diff / n
                            self.target_ext = (self._calib_ext_sum_angle / n) - 15
                            self.stage = "calibration_transition"
                            self._calib_start_time = None
                            self._calib_ext_sum_upper_ratio = 0.0
                            self._calib_ext_sum_forearm_ratio = 0.0
                            self._calib_ext_sum_wrist_z_diff = 0.0
                            self._calib_ext_sum_elbow_z_diff = 0.0
                            self._calib_ext_sum_angle = 0.0
                            self._calib_ext_sample_count = 0
                    else:
                        self._calib_start_time = None
                        self._calib_ext_sum_upper_ratio = 0.0
                        self._calib_ext_sum_forearm_ratio = 0.0
                        self._calib_ext_sum_wrist_z_diff = 0.0
                        self._calib_ext_sum_elbow_z_diff = 0.0
                        self._calib_ext_sum_angle = 0.0
                        self._calib_ext_sample_count = 0
                        feedback_msg = "DROP ARM STRAIGHT TO START"

                elif self.stage == "calibration_transition":
                    feedback_msg = "SUCCESS! NOW BEND ARM..."
                    if self._calib_transition_start_time is None:
                        self._calib_transition_start_time = now
                    if (now - self._calib_transition_start_time) > cfg.calib_transition_seconds:
                        self.stage = "calibration_flex"
                        self._calib_transition_start_time = None
                        self._calib_flex_start_time = now

                elif self.stage == "calibration_flex":
                    if self.smoothed_angle < self.target_flex:
                        self.target_flex = self.smoothed_angle

                    if self._calib_flex_start_time is None:
                        self._calib_flex_start_time = now
                    flex_elapsed = now - self._calib_flex_start_time

                    if self.smoothed_angle < cfg.flexion_angle_max_deg:
                        if self._calib_flex_hold_start_time is None:
                            self._calib_flex_hold_start_time = now
                        hold_elapsed = now - self._calib_flex_hold_start_time
                        seconds_left = max(0, int(cfg.calib_flex_seconds - hold_elapsed) + 1)
                        feedback_msg = f"HOLD THIS FLEX: {seconds_left}s"

                        if hold_elapsed >= cfg.calib_flex_seconds:
                            self.target_flex += 15
                            self.stage = "extended"
                            self._calib_flex_hold_start_time = None
                            self._calib_flex_start_time = None
                    else:
                        self._calib_flex_hold_start_time = None
                        feedback_msg = "BEND ARM FULLY"

                    if self.stage == "calibration_flex" and flex_elapsed >= cfg.calib_flex_timeout_seconds:
                        self.target_flex += 15
                        self.stage = "extended"
                        self._calib_flex_hold_start_time = None
                        self._calib_flex_start_time = None
                        logger.info(
                            "Flexion calibration timed out; accepting best observed angle (%.1f deg).",
                            self.target_flex,
                        )

                return self.smoothed_angle, elbow, True, feedback_msg

            # ==========================================
            # FORM ENFORCEMENT & Z-AXIS CHECK
            # ==========================================
            is_upper_arm_2d_stable = upper_ratio > (self.baseline_upper_ratio * cfg.stability_ratio)
            is_forearm_2d_stable = forearm_ratio > (self.baseline_forearm_ratio * cfg.stability_ratio)

            wrist_z_limit = max(self.baseline_wrist_z_diff,
                                cfg.wrist_z_max * cfg.z_stability_ratio) / cfg.z_stability_ratio \
                if self.baseline_wrist_z_diff > 0 else cfg.wrist_z_max
            elbow_z_limit = max(self.baseline_elbow_z_diff,
                                cfg.elbow_z_max * cfg.z_stability_ratio) / cfg.z_stability_ratio \
                if self.baseline_elbow_z_diff > 0 else cfg.elbow_z_max
            is_arm_in_z_plane = (wrist_z_diff < wrist_z_limit) and (elbow_z_diff < elbow_z_limit)

            is_in_plane = is_upper_arm_2d_stable and is_forearm_2d_stable and is_arm_in_z_plane

            dx = elbow[0] - shoulder[0]
            dy = elbow[1] - shoulder[1]
            vertical_angle = np.degrees(np.arctan2(abs(dx), max(dy, 1)))
            is_elbow_pinned = vertical_angle < cfg.pinned_elbow_max_angle_deg

            is_elbow_down = elbow[1] > shoulder[1]
            is_wrist_down = wrist[1] > elbow[1]

            is_trunk_stable = abs(shoulder[0] - hip[0]) < (cfg.trunk_sway_max_ratio * torso_length)

            good_form = is_elbow_down and is_in_plane and is_trunk_stable and is_side_profile and is_elbow_pinned

            if not good_form:
                if self.stage != "error":
                    # Form breaks -> Immediately void the rep
                    self.stage = "error"

                if not is_side_profile:
                    self.active_error_msg = "TURN TO SIDE!"
                elif not is_elbow_pinned:
                    self.active_error_msg = "KEEP ELBOW PINNED!"
                elif not is_in_plane:
                    self.active_error_msg = "ARM SWINGING OUT!"
                elif not is_trunk_stable:
                    self.active_error_msg = "TRUNK SWAY!"
                else:
                    self.active_error_msg = "INCORRECT FORM!"

                if self.error_timer_start is None:
                    self.error_timer_start = time.monotonic()

            elif self.stage == "error":
                # Good form returns, but the rep remains voided.
                # User must return to full extension to reset the cycle.
                self.stage = "recovering"

            if self.error_timer_start is not None:
                feedback_msg = self.active_error_msg
                if good_form and (time.monotonic() - self.error_timer_start) > cfg.error_display_min_seconds:
                    self.error_timer_start = None

            # ==========================================
            # REP COUNTING LOGIC
            # ==========================================
            if good_form:
                if self.smoothed_angle > self.target_ext and is_wrist_down:
                    if self.stage == 'flexed':
                        self.counter += 1
                        logger.info("Rep completed! Total: %d", self.counter)

                    # Both 'flexed' and 'recovering' stages resolve to 'extended' here
                    self.stage = "extended"

                elif self.smoothed_angle < self.target_flex and self.stage == 'extended':
                    self.stage = "flexed"

            return self.smoothed_angle, elbow, good_form, feedback_msg
        else:
            if not self.stage.startswith("calibration"):
                self.stage = "error"
            return None, None, False, "Stand fully in frame!"


# Registry
EXERCISE_REGISTRY = {
    ElbowFlexion.slug: ElbowFlexion,
}


# ==========================================
# 4. BACKGROUND VIDEO/POSE WORKER
# ==========================================
class VideoWorker(threading.Thread):
    INFERENCE_MAX_DIM = 480

    def __init__(self, side, csv_path, result_queue, config: ElbowFlexionConfig = None,
                 exercise_cls=ElbowFlexion, model_complexity=1):
        super().__init__(daemon=True)
        self.result_queue = result_queue
        self.exercise = exercise_cls(side=side, config=config)
        self.csv_path = csv_path
        self.model_complexity = model_complexity
        self._stop_event = threading.Event()
        self._camera_ready = threading.Event()
        self._error = None

    def stop(self):
        self._stop_event.set()

    def wait_until_stopped(self, timeout=None):
        self.join(timeout=timeout)
        return not self.is_alive()

    def run(self):
        cap = None
        pose = None
        csv_file = None
        try:
            if sys.platform.startswith("win"):
                cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
            elif sys.platform.startswith("linux"):
                cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
            else:
                cap = cv2.VideoCapture(0)

            if not cap.isOpened():
                self._push_error("Could not open camera.")
                return

            pose = mp_pose.Pose(
                model_complexity=self.model_complexity,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
            )

            os.makedirs(os.path.dirname(self.csv_path), exist_ok=True)
            csv_file = open(self.csv_path, mode='w', newline='')
            csv_writer = csv.writer(csv_file)
            csv_writer.writerow(["Timestamp_ISO", "Smoothed_Angle", "Stage", "Good_Form", "Feedback"])
            frames_since_flush = 0

            while not self._stop_event.is_set():
                loop_start = time.monotonic()

                success, img = cap.read()
                if not success:
                    time.sleep(0.05)
                    continue

                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                h, w, _ = img.shape

                scale = self.INFERENCE_MAX_DIM / float(max(h, w))
                if scale < 1.0:
                    small_w, small_h = int(w * scale), int(h * scale)
                    img_for_inference = cv2.resize(img_rgb, (small_w, small_h), interpolation=cv2.INTER_AREA)
                else:
                    img_for_inference = img_rgb

                results = pose.process(img_for_inference)

                angle = None
                is_good = False
                msg = ""
                stage = self.exercise.stage
                counter = self.exercise.counter

                if results.pose_landmarks:
                    self._mask_face(results.pose_landmarks.landmark, img_rgb, h, w)
                    mp_draw.draw_landmarks(img_rgb, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

                    angle, joint_pos, is_good, msg = self.exercise.process_frame(
                        results.pose_landmarks.landmark, h, w
                    )

                    nose = results.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE.value]
                    if nose.visibility > 0.5 and nose.y < 0.10:
                        is_good = False
                        msg = "STEP BACK! " + (msg if msg else "")

                    stage = self.exercise.stage
                    counter = self.exercise.counter

                    if angle is not None:
                        timestamp_iso = datetime.datetime.now().isoformat(timespec="milliseconds")
                        csv_writer.writerow([timestamp_iso, round(angle, 2), stage, is_good, msg])
                        frames_since_flush += 1
                        if frames_since_flush >= CSV_FLUSH_EVERY_N_FRAMES:
                            csv_file.flush()
                            frames_since_flush = 0

                        color = COLOR_GOOD_RGB if is_good else COLOR_BAD_RGB
                        cv2.putText(img_rgb, f"{int(angle)} deg", tuple(np.array(joint_pos).astype(int)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)

                    if msg:
                        cv2.putText(img_rgb, msg, (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, COLOR_BAD_RGB, 3, cv2.LINE_AA)

                self._push_result(img_rgb, counter, stage, msg, is_good)

                elapsed = time.monotonic() - loop_start
                remaining = (1.0 / TARGET_FPS) - elapsed
                if remaining > 0:
                    time.sleep(remaining)

        except Exception:
            logger.exception("Camera/pose worker crashed")
            self._push_error("Camera/pose error - check logs for details.")
        finally:
            if cap is not None:
                cap.release()
            if pose is not None:
                pose.close()
            if csv_file is not None:
                csv_file.close()

    def _mask_face(self, landmarks, img_rgb, h, w):
        nose = landmarks[mp_pose.PoseLandmark.NOSE.value]
        l_ear = landmarks[mp_pose.PoseLandmark.LEFT_EAR.value]
        r_ear = landmarks[mp_pose.PoseLandmark.RIGHT_EAR.value]

        if nose.visibility <= 0.5:
            return

        nx, ny = int(nose.x * w), int(nose.y * h)

        ears_visible = l_ear.visibility > 0.5 and r_ear.visibility > 0.5
        if ears_visible:
            ear_dist = abs(l_ear.x - r_ear.x) * w
            dynamic_radius = int(ear_dist * 1.5) if ear_dist > 0 else int(h * 0.08)
        else:
            dynamic_radius = int(h * 0.08)

        cv2.circle(img_rgb, (nx, ny), dynamic_radius, (25, 25, 25), -1)

    def _push_result(self, img_rgb, counter, stage, msg, is_good):
        try:
            while True:
                self.result_queue.get_nowait()
        except queue.Empty:
            pass
        self.result_queue.put_nowait({
            "type": "frame",
            "image": img_rgb,
            "counter": counter,
            "stage": stage,
            "msg": msg,
            "is_good": is_good,
        })

    def _push_error(self, message):
        try:
            self.result_queue.put_nowait({"type": "error", "message": message})
        except queue.Full:
            pass


# ==========================================
# 5. TKINTER APPLICATION CLASS
# ==========================================
class PhysioApp:
    POLL_INTERVAL_MS = 15
    WORKER_JOIN_TIMEOUT_S = 2.0

    def __init__(self, root):
        self.root = root
        self.root.title("Intelligent Physiotherapy Assistant")
        self.root.geometry("1000x650")
        self.root.configure(bg="#2C3E50")

        self.worker = None
        self.result_queue = queue.Queue(maxsize=2)
        self._poll_job = None
        self._stopping = False

        self.main_menu_frame = tk.Frame(self.root, bg="#2C3E50")
        self.tracking_frame = tk.Frame(self.root, bg="#2C3E50")

        self.build_main_menu()
        self.build_tracking_dashboard()

        self.main_menu_frame.pack(fill="both", expand=True)
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

    def build_main_menu(self):
        title_font = font.Font(family="Helvetica", size=24, weight="bold")
        btn_font = font.Font(family="Helvetica", size=14)

        tk.Label(self.main_menu_frame, text="Intelligent Physiotherapy Assistant", font=title_font, fg="white",
                 bg="#2C3E50").pack(pady=40)
        tk.Label(self.main_menu_frame, text="Select Exercise Protocol", font=btn_font, fg="#BDC3C7", bg="#2C3E50").pack(
            pady=10)

        self.status_label = tk.Label(self.main_menu_frame, text="", font=font.Font(family="Helvetica", size=11),
                                     fg="#E74C3C", bg="#2C3E50")
        self.status_label.pack(pady=(0, 10))

        tk.Button(self.main_menu_frame, text="Elbow Flexion (Right Arm)", font=btn_font, bg="#2980B9", fg="white",
                  width=30, height=2,
                  command=lambda: self.start_tracking("elbow_flexion", "right")).pack(pady=10)

        tk.Button(self.main_menu_frame, text="Elbow Flexion (Left Arm)", font=btn_font, bg="#2980B9", fg="white",
                  width=30, height=2,
                  command=lambda: self.start_tracking("elbow_flexion", "left")).pack(pady=10)

    def build_tracking_dashboard(self):
        self.video_label = tk.Label(self.tracking_frame, bg="black")
        self.video_label.pack(side="left", padx=20, pady=20)

        info_frame = tk.Frame(self.tracking_frame, bg="#2C3E50")
        info_frame.pack(side="right", fill="both", expand=True, padx=20, pady=20)

        info_font = font.Font(family="Helvetica", size=18, weight="bold")

        self.lbl_reps = tk.Label(info_frame, text="Reps: 0", font=info_font, fg="#2ECC71", bg="#2C3E50")
        self.lbl_reps.pack(pady=20)

        self.lbl_stage = tk.Label(info_frame, text="Stage: N/A", font=info_font, fg="#F1C40F", bg="#2C3E50")
        self.lbl_stage.pack(pady=20)

        self.lbl_feedback = tk.Label(info_frame, text="", font=font.Font(family="Helvetica", size=14, weight="bold"),
                                     fg="#E74C3C", bg="#2C3E50", wraplength=250)
        self.lbl_feedback.pack(pady=40)

        btn_stop = tk.Button(info_frame, text="End Session", font=font.Font(family="Helvetica", size=14), bg="#C0392B",
                             fg="white", width=15, command=self.stop_tracking)
        btn_stop.pack(side="bottom", pady=40)

    def start_tracking(self, exercise_slug, side):
        if self.worker is not None or self._stopping:
            self.status_label.config(text="Previous session is still shutting down, please wait...")
            return

        exercise_cls = EXERCISE_REGISTRY[exercise_slug]
        self.status_label.config(text="")

        self.main_menu_frame.pack_forget()
        self.tracking_frame.pack(fill="both", expand=True)

        session_id = int(time.time())
        csv_name = f"session_{exercise_slug}_{side}_{session_id}.csv"
        csv_path = os.path.join(SESSIONS_DIR, csv_name)

        while not self.result_queue.empty():
            try:
                self.result_queue.get_nowait()
            except queue.Empty:
                break

        self.worker = VideoWorker(side=side, csv_path=csv_path, result_queue=self.result_queue,
                                  exercise_cls=exercise_cls)
        self.worker.start()

        self._poll_job = self.root.after(self.POLL_INTERVAL_MS, self._poll_results)

    def _poll_results(self):
        try:
            result = self.result_queue.get_nowait()
        except queue.Empty:
            result = None

        if result is not None:
            if result["type"] == "error":
                self.lbl_feedback.config(text=result["message"], fg="#E74C3C")
            else:
                img_pil = Image.fromarray(result["image"])
                img_tk = ImageTk.PhotoImage(image=img_pil)
                self.video_label.imgtk = img_tk
                self.video_label.configure(image=img_tk)

                self.lbl_reps.config(text=f"Reps: {result['counter']}")
                self.lbl_stage.config(text=f"Stage: {result['stage'].replace('_', ' ').title()}")
                msg = result["msg"]
                self.lbl_feedback.config(text=msg if msg else "Form: Optimal",
                                         fg="#E74C3C" if msg else "#2ECC71")

        if self.worker is not None:
            self._poll_job = self.root.after(self.POLL_INTERVAL_MS, self._poll_results)

    def stop_tracking(self):
        if self._poll_job is not None:
            self.root.after_cancel(self._poll_job)
            self._poll_job = None

        if self.worker is not None:
            worker = self.worker
            self._stopping = True
            worker.stop()

            confirmed_dead = worker.wait_until_stopped(timeout=self.WORKER_JOIN_TIMEOUT_S)
            if confirmed_dead:
                self.worker = None
                self._stopping = False
            else:
                logger.warning("Worker thread did not stop within timeout; waiting in background.")
                self._await_worker_shutdown(worker)

        self.tracking_frame.pack_forget()
        self.main_menu_frame.pack(fill="both", expand=True)

    def _await_worker_shutdown(self, worker, attempt=0):
        if not worker.is_alive():
            if self.worker is worker:
                self.worker = None
            self._stopping = False
            self.status_label.config(text="")
            logger.info("Previous worker thread confirmed stopped.")
            return

        if attempt == 0:
            self.status_label.config(text="Releasing camera from previous session...")

        self.root.after(200, lambda: self._await_worker_shutdown(worker, attempt + 1))

    def on_close(self):
        self.stop_tracking()
        self.root.destroy()
        sys.exit(0)


if __name__ == "__main__":
    root = tk.Tk()
    app = PhysioApp(root)
    root.mainloop()