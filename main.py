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

# Colors are specified in RGB (the frame is converted to RGB before any
# drawing happens) even though OpenCV normally expects BGR. Naming them
# explicitly avoids relying on the "red/green are symmetric" coincidence.
COLOR_GOOD_RGB = (46, 204, 113)   # green
COLOR_BAD_RGB = (231, 76, 60)     # red

SESSIONS_DIR = "sessions"
CSV_FLUSH_EVERY_N_FRAMES = 15
TARGET_FPS = 30.0


# ==========================================
# 1. CONFIG (tunable thresholds, pulled out of the exercise logic)
# ==========================================
@dataclass
class ElbowFlexionConfig:
    stability_ratio: float = 0.75          # min fraction of baseline limb-segment length to still count as "in plane"
    side_profile_max_shoulder_ratio: float = 0.35  # shoulder-width / torso-length ceiling for "side profile"
    wrist_z_max: float = 0.40              # max |wrist.z - shoulder.z| before flagging arm swinging out of plane
    elbow_z_max: float = 0.30              # max |elbow.z - shoulder.z| before flagging arm swinging out of plane
    pinned_elbow_max_angle_deg: float = 20.0  # max deviation from vertical for the upper arm to count as "pinned"
    trunk_sway_max_ratio: float = 0.20     # max horizontal shoulder/hip offset, as a fraction of torso length
    error_display_min_seconds: float = 1.5  # minimum time an error message stays up once triggered, even if fixed sooner
    calib_extension_frames: int = 90       # ~3s at 30fps to lock in the extension baseline
    calib_transition_frames: int = 45      # ~1.5s pause between calibration phases
    calib_flex_frames: int = 60            # ~2s hold to lock in the flexion baseline
    extension_angle_min_deg: float = 140.0
    flexion_angle_max_deg: float = 60.0
    min_visibility_tracking: float = 0.4
    min_visibility_calibration: float = 0.7
    smoothing_time_constant_s: float = 0.15  # EMA time constant (tau); replaces a fixed per-frame alpha


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
        """
        Exponential moving average with a time-constant (tau) rather than a
        fixed per-frame alpha, so smoothing behavior stays consistent even
        if frames are dropped or the loop briefly lags (alpha would
        otherwise implicitly assume a fixed dt between updates).
        """
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
    # Human-readable label + CSV/session filename slug, used by the
    # exercise registry below so the GUI and file naming stay in sync
    # with whatever exercises are registered.
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
        self.target_ext = 0.0
        self.target_flex = 180.0

        self.calib_frames = 0
        self.flex_calib_frames = 0

        self.active_error_msg = ""
        self.error_timer_start = None

        # Preserves the in-rep stage ("extended"/"flexed") across a
        # transient form error so a good-form recovery doesn't force the
        # user to fully re-extend before a flex can register again.
        self.stage_before_error = None

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

            # ==========================================
            # PRE-CALIBRATION SIDE-PROFILE GUARD
            # ==========================================
            shoulder_width = abs(t_shoulder.x - opp_shoulder.x) * w
            is_side_profile = shoulder_width < (cfg.side_profile_max_shoulder_ratio * torso_length)

            # ==========================================
            # ROBUST CALIBRATION (Frame-Based Data Storage)
            # ==========================================
            if self.stage.startswith("calibration"):

                if not is_side_profile:
                    self.calib_frames = max(0, self.calib_frames - 2)
                    self.flex_calib_frames = max(0, self.flex_calib_frames - 2)
                    return self.smoothed_angle, elbow, False, "TURN TO SIDE PROFILE!"

                if self.stage == "calibration_start":
                    if self.smoothed_angle > cfg.extension_angle_min_deg and wrist[1] > elbow[1]:
                        self.calib_frames += 1
                        seconds_left = max(0, 3 - (self.calib_frames // 30))
                        feedback_msg = f"STAND STILL: {seconds_left}s"

                        if self.calib_frames >= cfg.calib_extension_frames:
                            self.baseline_upper_ratio = upper_ratio
                            self.baseline_forearm_ratio = forearm_ratio
                            self.target_ext = self.smoothed_angle - 15
                            self.stage = "calibration_transition"
                            self.calib_frames = 0
                    else:
                        self.calib_frames = max(0, self.calib_frames - 2)
                        feedback_msg = "DROP ARM STRAIGHT TO START"

                elif self.stage == "calibration_transition":
                    feedback_msg = "SUCCESS! NOW BEND ARM..."
                    self.calib_frames += 1
                    if self.calib_frames > cfg.calib_transition_frames:
                        self.stage = "calibration_flex"
                        self.calib_frames = 0

                elif self.stage == "calibration_flex":
                    if self.smoothed_angle < self.target_flex:
                        self.target_flex = self.smoothed_angle

                    if self.smoothed_angle < cfg.flexion_angle_max_deg:
                        self.flex_calib_frames += 1
                        seconds_left = max(0, 2 - (self.flex_calib_frames // 30))
                        feedback_msg = f"HOLD THIS FLEX: {seconds_left}s"

                        if self.flex_calib_frames >= cfg.calib_flex_frames:
                            self.target_flex += 15
                            self.stage = "extended"
                            self.flex_calib_frames = 0
                    else:
                        self.flex_calib_frames = max(0, self.flex_calib_frames - 2)
                        feedback_msg = "BEND ARM FULLY"

                return self.smoothed_angle, elbow, True, feedback_msg

            # ==========================================
            # FORM ENFORCEMENT & Z-AXIS CHECK
            # ==========================================
            is_upper_arm_2d_stable = upper_ratio > (self.baseline_upper_ratio * cfg.stability_ratio)
            is_forearm_2d_stable = forearm_ratio > (self.baseline_forearm_ratio * cfg.stability_ratio)

            wrist_depth_diff = abs(t_wrist.z - t_shoulder.z)
            elbow_depth_diff = abs(t_elbow.z - t_shoulder.z)
            is_arm_in_z_plane = (wrist_depth_diff < cfg.wrist_z_max) and (elbow_depth_diff < cfg.elbow_z_max)

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
                    # Entering an error state for the first time this
                    # episode -- remember what the rep stage was so it
                    # can be restored once good form returns.
                    self.stage_before_error = self.stage
                    self.stage = "error"

                # Re-evaluate the specific cause every frame while form is
                # bad, so the message reflects the *current* problem
                # instead of freezing on whatever tripped first.
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
                # Good form has returned. Keep showing the error banner
                # for at least error_display_min_seconds so it doesn't
                # flicker, but resume rep tracking from wherever the user
                # was before the error instead of requiring a full
                # extension first.
                self.stage = self.stage_before_error or "extended"
                self.stage_before_error = None

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
                    self.stage = "extended"

                elif self.smoothed_angle < self.target_flex and self.stage == 'extended':
                    self.stage = "flexed"

            return self.smoothed_angle, elbow, good_form, feedback_msg
        else:
            if not self.stage.startswith("calibration"):
                if self.stage != "error":
                    self.stage_before_error = self.stage
                self.stage = "error"
            return None, None, False, "Stand fully in frame!"


# Registry mapping a stable slug -> exercise class, so the GUI and file
# naming can stay data-driven instead of hardcoding a single exercise.
# Add new BaseExercise subclasses here to extend the app.
EXERCISE_REGISTRY = {
    ElbowFlexion.slug: ElbowFlexion,
}


# ==========================================
# 4. BACKGROUND VIDEO/POSE WORKER
# ==========================================
class VideoWorker(threading.Thread):
    """
    Owns the camera, the MediaPipe Pose instance, and the active exercise.
    Runs capture + inference + exercise logic off the Tk main thread so a
    slow inference frame never stalls the GUI event loop. Only this thread
    mutates `exercise` state; the main thread only reads the packaged
    results it puts on `result_queue`.
    """

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
        """
        Blocks until the thread has actually finished (camera released,
        pose model closed, CSV file closed) rather than just until
        `join()` times out. Returns True if the thread is confirmed dead.
        """
        self.join(timeout=timeout)
        return not self.is_alive()

    def run(self):
        cap = None
        pose = None
        csv_file = None
        try:
            # Backend hints speed up camera init on Windows/Linux; falls
            # back to the default backend automatically if unsupported.
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
                    # Camera hiccup -- back off briefly and retry rather
                    # than spinning or crashing the thread.
                    time.sleep(0.05)
                    continue

                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                results = pose.process(img_rgb)
                h, w, _ = img.shape

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

                # Pace the loop to roughly TARGET_FPS without busy-waiting.
                # monotonic() is immune to wall-clock jumps (NTP sync, DST)
                # that time.time() is exposed to.
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

        # Only trust the ear-derived radius if both ears are actually
        # visible -- otherwise MediaPipe can still return a stale/low-
        # confidence coordinate for an occluded ear, which would silently
        # produce a wrongly-sized (or misplaced) mask and risk exposing
        # part of the face. Fall back to a fixed radius in that case.
        ears_visible = l_ear.visibility > 0.5 and r_ear.visibility > 0.5
        if ears_visible:
            ear_dist = abs(l_ear.x - r_ear.x) * w
            dynamic_radius = int(ear_dist * 1.5) if ear_dist > 0 else int(h * 0.08)
        else:
            dynamic_radius = int(h * 0.08)

        cv2.circle(img_rgb, (nx, ny), dynamic_radius, (25, 25, 25), -1)

    def _push_result(self, img_rgb, counter, stage, msg, is_good):
        # Keep only the freshest frame -- if the UI hasn't consumed the
        # previous one yet, drop it rather than letting the queue (and
        # therefore latency) grow unbounded.
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
        # True while a worker thread is being shut down but hasn't been
        # confirmed dead yet -- blocks starting a new session so two
        # threads never race for the same camera device.
        self._stopping = False

        # Build UI Frames
        self.main_menu_frame = tk.Frame(self.root, bg="#2C3E50")
        self.tracking_frame = tk.Frame(self.root, bg="#2C3E50")

        self.build_main_menu()
        self.build_tracking_dashboard()

        # Start by showing the main menu
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

        # Drain any stale results from a previous session.
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
                # Thread didn't die in time (camera/pose call stuck). Keep
                # a background watcher polling instead of discarding the
                # reference -- discarding it here would let a user start a
                # new session while the old thread still holds the camera
                # device, causing an open failure or corrupted frames.
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

        # Keep checking without blocking the GUI thread.
        self.root.after(200, lambda: self._await_worker_shutdown(worker, attempt + 1))

    def on_close(self):
        self.stop_tracking()
        self.root.destroy()
        sys.exit(0)


if __name__ == "__main__":
    root = tk.Tk()
    app = PhysioApp(root)
    root.mainloop()