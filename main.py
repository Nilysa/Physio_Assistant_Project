import cv2
import csv
import mediapipe as mp
import numpy as np
import time
import sys
import tkinter as tk
from tkinter import font
from PIL import Image, ImageTk

# ==========================================
# GLOBALS & DEPENDENCIES
# ==========================================
mp_pose = mp.solutions.pose
mp_draw = mp.solutions.drawing_utils


# ==========================================
# 1. BASE EXERCISE CLASS
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
# 2. SPECIFIC EXERCISE CLASS (ELBOW FLEXION)
# ==========================================
class ElbowFlexion(BaseExercise):
    def __init__(self, side="right"):
        super().__init__()
        self.stage = "calibration_start"
        self.side = side.lower()

        self.baseline_upper_ratio = 0.0
        self.baseline_forearm_ratio = 0.0
        self.target_ext = 0.0
        self.target_flex = 180.0

        self.calib_frames = 0
        self.flex_calib_frames = 0

        self.active_error_msg = ""
        self.error_timer_start = None

    def process_frame(self, landmarks, h, w):
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
        req_vis = 0.7 if self.stage.startswith("calib") else 0.4

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
            # We calculate shoulder width early so calibration can mandate a sagittal stance
            shoulder_width = abs(t_shoulder.x - opp_shoulder.x) * w
            is_side_profile = shoulder_width < (0.35 * torso_length)

            # ==========================================
            # ROBUST CALIBRATION (Frame-Based Data Storage)
            # ==========================================
            if self.stage.startswith("calibration"):

                # Halt calibration immediately if the user turns toward the camera
                if not is_side_profile:
                    self.calib_frames = max(0, self.calib_frames - 2)
                    self.flex_calib_frames = max(0, self.flex_calib_frames - 2)
                    return self.smoothed_angle, elbow, False, "TURN TO SIDE PROFILE!"

                if self.stage == "calibration_start":
                    if self.smoothed_angle > 140 and wrist[1] > elbow[1]:
                        self.calib_frames += 1
                        seconds_left = max(0, 3 - (self.calib_frames // 30))
                        feedback_msg = f"STAND STILL: {seconds_left}s"

                        if self.calib_frames >= 90:
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
                    if self.calib_frames > 45:
                        self.stage = "calibration_flex"
                        self.calib_frames = 0

                elif self.stage == "calibration_flex":
                    if self.smoothed_angle < self.target_flex:
                        self.target_flex = self.smoothed_angle

                    if self.smoothed_angle < 60:
                        self.flex_calib_frames += 1
                        seconds_left = max(0, 2 - (self.flex_calib_frames // 30))
                        feedback_msg = f"HOLD THIS FLEX: {seconds_left}s"

                        if self.flex_calib_frames >= 60:
                            self.target_flex += 15
                            self.stage = "extended"
                            self.flex_calib_frames = 0
                    else:
                        self.flex_calib_frames = max(0, self.flex_calib_frames - 2)
                        feedback_msg = "BEND ARM FULLY"

                return self.smoothed_angle, elbow, True, feedback_msg

            # ==========================================
            # FORM ENFORCEMENT & RESTORED Z-AXIS
            # ==========================================
            is_upper_arm_2d_stable = upper_ratio > (self.baseline_upper_ratio * 0.75)
            is_forearm_2d_stable = forearm_ratio > (self.baseline_forearm_ratio * 0.75)

            wrist_depth_diff = abs(t_wrist.z - t_shoulder.z)
            elbow_depth_diff = abs(t_elbow.z - t_shoulder.z)
            is_arm_in_z_plane = (wrist_depth_diff < 0.40) and (elbow_depth_diff < 0.30)

            is_in_plane = is_upper_arm_2d_stable and is_forearm_2d_stable and is_arm_in_z_plane

            dx = elbow[0] - shoulder[0]
            dy = elbow[1] - shoulder[1]
            vertical_angle = np.degrees(np.arctan2(abs(dx), max(dy, 1)))
            is_elbow_pinned = vertical_angle < 20

            is_elbow_down = elbow[1] > shoulder[1]
            is_wrist_down = wrist[1] > elbow[1]

            is_trunk_stable = abs(shoulder[0] - hip[0]) < (0.20 * torso_length)
            # shoulder_width and is_side_profile are already calculated above

            good_form = is_elbow_down and is_in_plane and is_trunk_stable and is_side_profile and is_elbow_pinned

            if not good_form:
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

            if self.error_timer_start is not None:
                feedback_msg = self.active_error_msg
                if (time.time() - self.error_timer_start) > 1.5:
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
                self.stage = "error"
            return None, None, False, "Stand fully in frame!"


# ==========================================
# 3. TKINTER APPLICATION CLASS
# ==========================================
class PhysioApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Intelligent Physiotherapy Assistant")
        self.root.geometry("1000x650")
        self.root.configure(bg="#2C3E50")

        self.cap = None
        self.pose = None
        self.current_exercise = None
        self.csv_file = None
        self.csv_writer = None

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

        # Dashboard Buttons
        tk.Button(self.main_menu_frame, text="Elbow Flexion (Right Arm)", font=btn_font, bg="#2980B9", fg="white",
                  width=30, height=2,
                  command=lambda: self.start_tracking("right")).pack(pady=10)

        tk.Button(self.main_menu_frame, text="Elbow Flexion (Left Arm)", font=btn_font, bg="#2980B9", fg="white",
                  width=30, height=2,
                  command=lambda: self.start_tracking("left")).pack(pady=10)

    def build_tracking_dashboard(self):
        # Video Feed Label
        self.video_label = tk.Label(self.tracking_frame, bg="black")
        self.video_label.pack(side="left", padx=20, pady=20)

        # Right Side Dashboard Info
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

        # Initialize dependencies manually without 'with' statement so they persist
        self.pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
        self.cap = cv2.VideoCapture(0)
        self.current_exercise = ElbowFlexion(side=side)

        # Setup Logging
        session_id = int(time.time())
        self.csv_file = open(f"session_data_{session_id}.csv", mode='w', newline='')
        self.csv_writer = csv.writer(self.csv_file)
        self.csv_writer.writerow(["Timestamp", "Smoothed_Angle", "Stage", "Good_Form", "Feedback"])

        self.update_frame()

    def update_frame(self):
        start_time = time.time()

        if self.cap is None or not self.cap.isOpened():
            return

        success, img = self.cap.read()
        if success:
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = self.pose.process(img_rgb)
            h, w, _ = img.shape

            if results.pose_landmarks:
                # Privacy Masking
                nose = results.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE.value]
                l_ear = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_EAR.value]
                r_ear = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_EAR.value]

                if nose.visibility > 0.5:
                    nx, ny = int(nose.x * w), int(nose.y * h)
                    ear_dist = abs(l_ear.x - r_ear.x) * w
                    dynamic_radius = int(ear_dist * 1.5) if ear_dist > 0 else int(h * 0.08)
                    cv2.circle(img_rgb, (nx, ny), dynamic_radius, (25, 25, 25), -1)

                mp_draw.draw_landmarks(img_rgb, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

                angle, joint_pos, is_good, msg = self.current_exercise.process_frame(results.pose_landmarks.landmark, h, w)

                # Head-Boundary Guard (VISUAL WARNING ONLY - No longer breaks the state machine)
                if nose.visibility > 0.5 and nose.y < 0.10:
                    is_good = False
                    # Appends the step back warning while keeping calibration text visible
                    msg = "STEP BACK! " + (msg if msg else "")

                if angle is not None:
                    self.csv_writer.writerow([time.time(), round(angle, 2), self.current_exercise.stage, is_good, msg])
                    color = (0, 255, 0) if is_good else (255, 0, 0)
                    cv2.putText(img_rgb, f"{int(angle)} deg", tuple(np.multiply(joint_pos, [1, 1]).astype(int)),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)

                # --- ON-SCREEN TEXT OVERLAY ---
                if msg:
                    cv2.putText(img_rgb, msg, (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3, cv2.LINE_AA)

                # Update UI Dashboard Labels
                self.lbl_reps.config(text=f"Reps: {self.current_exercise.counter}")
                self.lbl_stage.config(text=f"Stage: {self.current_exercise.stage.replace('_', ' ').title()}")
                self.lbl_feedback.config(text=msg if msg else "Form: Optimal", fg="#E74C3C" if msg else "#2ECC71")

            # Convert OpenCV frame to Tkinter Image
            img_pil = Image.fromarray(img_rgb)
            img_tk = ImageTk.PhotoImage(image=img_pil)
            self.video_label.imgtk = img_tk
            self.video_label.configure(image=img_tk)

        # Dynamic Frame Scheduling
        elapsed_ms = int((time.time() - start_time) * 1000)
        delay = max(1, 33 - elapsed_ms)
        self.root.after(delay, self.update_frame)

    def stop_tracking(self):
        if self.cap:
            self.cap.release()
            self.cap = None
        if self.pose:
            self.pose.close()
            self.pose = None
        if self.csv_file:
            self.csv_file.close()
            self.csv_file = None

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