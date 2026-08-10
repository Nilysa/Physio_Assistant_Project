import cv2
import csv
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

# Colors are specified in RGB (the frame is converted to RGB before any
# drawing happens) even though OpenCV normally expects BGR. Naming them
# explicitly avoids relying on the "red/green are symmetric" coincidence.
COLOR_GOOD_RGB = (46, 204, 113)   # green
COLOR_BAD_RGB = (231, 76, 60)     # red

SESSIONS_DIR = "sessions"
CSV_FLUSH_EVERY_N_FRAMES = 15


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
    error_display_seconds: float = 1.5     # how long an error message stays on screen once triggered
    calib_extension_frames: int = 90       # ~3s at 30fps to lock in the extension baseline
    calib_transition_frames: int = 45      # ~1.5s pause between calibration phases
    calib_flex_frames: int = 60            # ~2s hold to lock in the flexion baseline
    extension_angle_min_deg: float = 140.0
    flexion_angle_max_deg: float = 60.0
    min_visibility_tracking: float = 0.4
    min_visibility_calibration: float = 0.7


# ==========================================
# 2. BASE EXERCISE CLASS
# ==========================================
class BaseExercise:
    def __init__(self):
        self.counter = 0
        self.stage = "extended"
        self.smoothed_angle = None
        self.smoothing_factor = 0.2

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

    def process_frame(self, landmarks, h, w):
        raise NotImplementedError("Subclasses must implement this method")


# ==========================================
# 3. SPECIFIC EXERCISE CLASS (ELBOW FLEXION)
# ==========================================
class ElbowFlexion(BaseExercise):
    def __init__(self, side="right", config: ElbowFlexionConfig = None):
        super().__init__()
        self.stage = "calibration_start"
        self.side = side.lower()
        self.cfg = config or ElbowFlexionConfig()

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

            if self.smoothed_angle is None:
                self.smoothed_angle = raw_angle
            else:
                self.smoothed_angle = (self.smoothing_factor * raw_angle) + (
                        (1 - self.smoothing_factor) * self.smoothed_angle)

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

                if self.error_timer_start is None:
                    self.error_timer_start = time.time()
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
            elif self.stage == "error":
                # Good form has returned -- resume rep tracking from
                # wherever the user was before the error instead of
                # silently requiring a full extension first.
                self.stage = self.stage_before_error or "extended"
                self.stage_before_error = None

            if self.error_timer_start is not None:
                feedback_msg = self.active_error_msg
                if (time.time() - self.error_timer_start) > cfg.error_display_seconds:
                    self.error_timer_start = None

            # ==========================================
            # REP COUNTING LOGIC
            # ==========================================
            if good_form:
                if self.smoothed_angle > self.target_ext and is_wrist_down:
                    if self.stage == 'flexed':
                        self.counter += 1
                        print(f"Rep completed! Total: {self.counter}")
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

    def __init__(self, side, csv_path, result_queue, config: ElbowFlexionConfig = None):
        super().__init__(daemon=True)
        self.result_queue = result_queue
        self.exercise = ElbowFlexion(side=side, config=config)
        self.csv_path = csv_path
        self._stop_event = threading.Event()
        self._error = None

    def stop(self):
        self._stop_event.set()

    def run(self):
        cap = None
        pose = None
        csv_file = None
        try:
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                self._push_error("Could not open camera.")
                return

            pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

            os.makedirs(os.path.dirname(self.csv_path), exist_ok=True)
            csv_file = open(self.csv_path, mode='w', newline='')
            csv_writer = csv.writer(csv_file)
            csv_writer.writerow(["Timestamp", "Smoothed_Angle", "Stage", "Good_Form", "Feedback"])
            frames_since_flush = 0

            while not self._stop_event.is_set():
                loop_start = time.time()

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
                        csv_writer.writerow([time.time(), round(angle, 2), stage, is_good, msg])
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

                # Pace the loop to roughly 30fps without busy-waiting.
                elapsed = time.time() - loop_start
                remaining = (1.0 / 30.0) - elapsed
                if remaining > 0:
                    time.sleep(remaining)

        except Exception as exc:  # noqa: BLE001 - surface any failure to the UI instead of dying silently
            self._push_error(f"Camera/pose error: {exc}")
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

        if nose.visibility > 0.5:
            nx, ny = int(nose.x * w), int(nose.y * h)
            ear_dist = abs(l_ear.x - r_ear.x) * w
            dynamic_radius = int(ear_dist * 1.5) if ear_dist > 0 else int(h * 0.08)
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

    def __init__(self, root):
        self.root = root
        self.root.title("Intelligent Physiotherapy Assistant")
        self.root.geometry("1000x650")
        self.root.configure(bg="#2C3E50")

        self.worker = None
        self.result_queue = queue.Queue(maxsize=2)
        self._poll_job = None

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

        tk.Button(self.main_menu_frame, text="Elbow Flexion (Right Arm)", font=btn_font, bg="#2980B9", fg="white",
                  width=30, height=2,
                  command=lambda: self.start_tracking("right")).pack(pady=10)

        tk.Button(self.main_menu_frame, text="Elbow Flexion (Left Arm)", font=btn_font, bg="#2980B9", fg="white",
                  width=30, height=2,
                  command=lambda: self.start_tracking("left")).pack(pady=10)

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

    def start_tracking(self, side):
        self.main_menu_frame.pack_forget()
        self.tracking_frame.pack(fill="both", expand=True)

        session_id = int(time.time())
        csv_path = os.path.join(SESSIONS_DIR, f"session_data_{session_id}.csv")

        # Drain any stale results from a previous session.
        while not self.result_queue.empty():
            try:
                self.result_queue.get_nowait()
            except queue.Empty:
                break

        self.worker = VideoWorker(side=side, csv_path=csv_path, result_queue=self.result_queue)
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
            self.worker.stop()
            self.worker.join(timeout=2.0)
            self.worker = None

        self.tracking_frame.pack_forget()
        self.main_menu_frame.pack(fill="both", expand=True)

    def on_close(self):
        self.stop_tracking()
        self.root.destroy()
        sys.exit(0)


if __name__ == "__main__":
    root = tk.Tk()
    app = PhysioApp(root)
    root.mainloop()