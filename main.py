import cv2
import csv
import mediapipe as mp
import numpy as np
import time
import sys

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
        """Universal math function for all exercises."""
        a = np.array(a)
        b = np.array(b)
        c = np.array(c)

        ba = a - b
        bc = c - b

        # Prevent zero-division during math
        norm_ba = max(np.linalg.norm(ba), 1e-6)
        norm_bc = max(np.linalg.norm(bc), 1e-6)

        cosine_angle = np.dot(ba, bc) / (norm_ba * norm_bc)
        cosine_angle = np.clip(cosine_angle, -1.0, 1.0)

        angle = np.arccos(cosine_angle)
        return np.degrees(angle)

    def process_frame(self, landmarks, h, w):
        """To be overridden by specific exercise classes."""
        raise NotImplementedError("Subclasses must implement this method")


# ==========================================
# 2. SPECIFIC EXERCISE CLASS (ELBOW FLEXION)
# ==========================================
class ElbowFlexion(BaseExercise):
    def __init__(self, side="right"):
        super().__init__()
        self.stage = "calibration_start"
        self.side = side.lower()  # Accepts 'right' or 'left'

        self.baseline_upper_ratio = 0.0
        self.baseline_forearm_ratio = 0.0
        self.target_ext = 0.0
        self.target_flex = 180.0

        # Replacing frame counters with time-based tracking
        self.timer_start = None

        self.active_error_msg = ""
        self.error_timer_start = None

    def process_frame(self, landmarks, h, w):
        # 1. DYNAMIC LATERALITY (Left vs Right Arm Selection)
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

            # 2. TIME-BASED CALIBRATION LOGIC
            if self.stage.startswith("calibration"):
                if self.stage == "calibration_start":
                    if self.smoothed_angle > 140 and wrist[1] > elbow[1]:
                        if self.timer_start is None:
                            self.timer_start = time.time()

                        elapsed = time.time() - self.timer_start
                        seconds_left = max(0, 3 - int(elapsed))
                        feedback_msg = f"STAND STILL: {seconds_left}s"

                        if elapsed >= 3.0:
                            self.baseline_upper_ratio = upper_ratio
                            self.baseline_forearm_ratio = forearm_ratio
                            self.target_ext = self.smoothed_angle - 15
                            self.stage = "calibration_transition"
                            self.timer_start = None
                    else:
                        self.timer_start = None
                        feedback_msg = "DROP ARM STRAIGHT TO START"

                elif self.stage == "calibration_transition":
                    feedback_msg = "SUCCESS! NOW BEND ARM..."
                    if self.timer_start is None:
                        self.timer_start = time.time()
                    if (time.time() - self.timer_start) > 1.5:
                        self.stage = "calibration_flex"
                        self.timer_start = None

                elif self.stage == "calibration_flex":
                    if self.smoothed_angle < self.target_flex:
                        self.target_flex = self.smoothed_angle

                    if self.smoothed_angle < 60:
                        if self.timer_start is None:
                            self.timer_start = time.time()

                        elapsed = time.time() - self.timer_start
                        seconds_left = max(0, 2 - int(elapsed))
                        feedback_msg = f"HOLD THIS FLEX: {seconds_left}s"

                        if elapsed >= 2.0:
                            self.target_flex += 15  # Give the user a 15-degree buffer
                            self.stage = "extended"
                            self.timer_start = None
                    else:
                        self.timer_start = None
                        feedback_msg = "BEND ARM FULLY"

                return self.smoothed_angle, elbow, True, feedback_msg

            # --- ADJUSTED TOLERANCES (Fixed for Human Anatomy) ---
            # Relaxed back to 0.75 to account for the natural carrying angle of the elbow during flexion
            is_upper_arm_2d_stable = upper_ratio > (self.baseline_upper_ratio * 0.75)
            is_forearm_2d_stable = forearm_ratio > (self.baseline_forearm_ratio * 0.75)

            # NOTE: Z-axis math is inherently noisy in MediaPipe (Arrowsmith et al., 2023).
            # We are removing the Z-axis check completely and relying purely on 2D apparent length.
            is_in_plane = is_upper_arm_2d_stable and is_forearm_2d_stable

            dx = elbow[0] - shoulder[0]
            dy = elbow[1] - shoulder[1]
            vertical_angle = np.degrees(np.arctan2(abs(dx), max(dy, 1)))
            is_elbow_pinned = vertical_angle < 20

            is_elbow_down = elbow[1] > shoulder[1]
            is_wrist_down = wrist[1] > elbow[1]

            is_trunk_stable = abs(shoulder[0] - hip[0]) < (0.20 * torso_length)
            shoulder_width = abs(t_shoulder.x - opp_shoulder.x) * w
            is_side_profile = shoulder_width < (0.35 * torso_length)

            good_form = is_elbow_down and is_in_plane and is_trunk_stable and is_side_profile and is_elbow_pinned

            # 3. TIME-BASED ERROR LOGIC
            if not good_form:
                self.stage = "error"
                if self.error_timer_start is None:
                    self.error_timer_start = time.time()
                    if not is_elbow_pinned:
                        self.active_error_msg = "KEEP ELBOW PINNED!"
                    elif not is_in_plane:
                        self.active_error_msg = "ARM SWINGING OUT!"
                    elif not is_side_profile:
                        self.active_error_msg = "TURN TO SIDE!"
                    elif not is_trunk_stable:
                        self.active_error_msg = "TRUNK SWAY!"
                    else:
                        self.active_error_msg = "INCORRECT FORM!"

            if self.error_timer_start is not None:
                feedback_msg = self.active_error_msg
                if (time.time() - self.error_timer_start) > 1.5:
                    self.error_timer_start = None  # Clear error after 1.5 seconds

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
# 3. MAIN APPLICATION LOOP
# ==========================================
def main():
    # Setup CSV Data Logging (Essential for thesis graphs!)
    session_id = int(time.time())
    csv_filename = f"session_data_{session_id}.csv"

    with open(csv_filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Timestamp", "Smoothed_Angle", "Stage", "Good_Form", "Feedback"])

        with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
            cap = cv2.VideoCapture(0)

            if not cap.isOpened():
                print("Error: Could not access the webcam. Check permissions.")
                sys.exit(1)

            # Pass 'left' or 'right' depending on which arm the patient is rehabbing
            current_exercise = ElbowFlexion(side="right")

            try:
                while True:
                    success, img = cap.read()
                    if not success:
                        continue

                    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    results = pose.process(img_rgb)
                    h, w, _ = img.shape

                    if results.pose_landmarks:

                        # --- PRIVACY CONSTRAINT: FACE ANONYMIZATION ---
                        nose = results.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE.value]
                        l_ear = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_EAR.value]
                        r_ear = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_EAR.value]

                        if nose.visibility > 0.5:
                            nx, ny = int(nose.x * w), int(nose.y * h)

                            # Calculate distance between ears in pixels to dynamically scale the mask
                            ear_dist = abs(l_ear.x - r_ear.x) * w

                            # Multiply by 1.5 to ensure it covers the whole head, with a fallback just in case
                            dynamic_radius = int(ear_dist * 1.5) if ear_dist > 0 else int(h * 0.08)

                            cv2.circle(img, (nx, ny), dynamic_radius, (25, 25, 25), -1)

                        # --- MISSING LINES RESTORED HERE ---
                        mp_draw.draw_landmarks(img, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
                        angle, joint_pos, is_good, msg = current_exercise.process_frame(results.pose_landmarks.landmark,
                                                                                        h, w)
                        # -----------------------------------

                        # --- DATA LOGGING ---
                        if angle is not None:
                            # Write real-time data to CSV for thesis analysis
                            writer.writerow([time.time(), round(angle, 2), current_exercise.stage, is_good, msg])

                            color = (0, 255, 0) if is_good else (0, 0, 255)
                            cv2.putText(img, f"{int(angle)} deg",
                                        tuple(np.multiply(joint_pos, [1, 1]).astype(int)),
                                        cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)

                        if msg:
                            cv2.putText(img, msg, (50, 150),
                                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3, cv2.LINE_AA)
                    # --- GUI OVERLAYS ---
                    cv2.rectangle(img, (0, 0), (250, 73), (245, 117, 16), -1)
                    cv2.putText(img, 'REPS', (15, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
                    cv2.putText(img, str(current_exercise.counter), (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 2,
                                (255, 255, 255), 2, cv2.LINE_AA)
                    cv2.putText(img, 'STAGE', (100, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
                    cv2.putText(img, current_exercise.stage, (100, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2,
                                cv2.LINE_AA)

                    cv2.imshow("Physio Assistant Tracker", img)

                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break

            except Exception as e:
                print(f"A critical error occurred: {e}")

            finally:
                cap.release()
                cv2.destroyAllWindows()
                print(f"Resources safely released. Data saved to {csv_filename}")

if __name__ == "__main__":
    main()