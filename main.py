import cv2
import mediapipe as mp
import numpy as np


# ==========================================
# 1. BASE EXERCISE CLASS
# ==========================================
class BaseExercise:
    def __init__(self):
        self.counter = 0
        self.stage = "extended"  # Initialize safely
        self.smoothed_angle = None
        self.smoothing_factor = 0.2

    def calculate_angle(self, a, b, c):
        """Universal math function for all exercises."""
        a = np.array(a)
        b = np.array(b)
        c = np.array(c)

        ba = a - b
        bc = c - b

        cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
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
    def __init__(self):
        super().__init__()
        self.stage = "calibration_start"

        self.baseline_upper_ratio = 0.0
        self.baseline_forearm_ratio = 0.0
        self.target_ext = 0.0
        self.target_flex = 180.0

        # NEW: Timers for the UX Countdown
        self.calib_frames = 0
        self.flex_calib_frames = 0

    def process_frame(self, landmarks, h, w):
        r_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
        l_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
        r_elbow = landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value]
        r_wrist = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value]
        r_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value]

        feedback_msg = ""
        good_form = True

        # Require the full upper body to be in the frame before doing anything
        if r_shoulder.visibility > 0.7 and r_elbow.visibility > 0.7 and r_wrist.visibility > 0.7 and r_hip.visibility > 0.7:

            shoulder = [r_shoulder.x * w, r_shoulder.y * h]
            elbow = [r_elbow.x * w, r_elbow.y * h]
            wrist = [r_wrist.x * w, r_wrist.y * h]
            hip = [r_hip.x * w, r_hip.y * h]

            raw_angle = self.calculate_angle(shoulder, elbow, wrist)

            if self.smoothed_angle is None:
                self.smoothed_angle = raw_angle
            else:
                self.smoothed_angle = (self.smoothing_factor * raw_angle) + (
                            (1 - self.smoothing_factor) * self.smoothed_angle)

            # --- 1. NEW CALIBRATION UX WITH TIMERS ---
            if self.stage.startswith("calibration"):
                if self.stage == "calibration_start":
                    # Require arm to be roughly straight to start the timer
                    if self.smoothed_angle > 140 and wrist[1] > elbow[1]:
                        self.calib_frames += 1
                        seconds_left = 3 - (self.calib_frames // 30)
                        feedback_msg = f"STAND STILL TO CALIBRATE: {seconds_left}s"

                        if self.calib_frames >= 90:  # 3 seconds at 30fps
                            self.target_ext = self.smoothed_angle - 15
                            self.stage = "calibration_flex"
                            self.calib_frames = 0
                    else:
                        self.calib_frames = 0
                        feedback_msg = "DROP ARM STRAIGHT TO START"

                elif self.stage == "calibration_flex":
                    if self.smoothed_angle < self.target_flex:
                        self.target_flex = self.smoothed_angle

                    # If they bend past 60 degrees, start the hold timer
                    if self.smoothed_angle < 60:
                        self.flex_calib_frames += 1
                        seconds_left = 2 - (self.flex_calib_frames // 30)
                        feedback_msg = f"HOLD THIS FLEX: {max(0, seconds_left)}s"

                        if self.flex_calib_frames >= 60:  # 2 seconds
                            self.target_flex += 15
                            self.stage = "extended"
                            print(
                                f"Calibration Complete! Targets -> Ext: {int(self.target_ext)}, Flex: {int(self.target_flex)}")
                    else:
                        self.flex_calib_frames = 0
                        feedback_msg = "BEND ARM FULLY"

                return self.smoothed_angle, elbow, True, feedback_msg

            # --- 2. THE Z-AXIS DEPTH ALARM (Fixes the 45-degree cheat) ---
            # Checks if the wrist's depth leaps wildly out of alignment with the shoulder
            wrist_depth_diff = abs(r_wrist.z - r_shoulder.z)
            elbow_depth_diff = abs(r_elbow.z - r_shoulder.z)
            is_arm_in_z_plane = wrist_depth_diff < 0.25 and elbow_depth_diff < 0.20

            # --- 3. THE PLUMB LINE ALARM (Fixes the forward elbow drift) ---
            dx = elbow[0] - shoulder[0]
            dy = elbow[1] - shoulder[1]
            # Calculate the angle of the upper arm relative to perfect vertical
            vertical_angle = np.degrees(np.arctan2(abs(dx), max(dy, 1)))
            is_elbow_pinned = vertical_angle < 20

            # --- 4. PREVIOUS FORM CHECKS ---
            is_elbow_down = elbow[1] > shoulder[1]
            is_wrist_down = wrist[1] > elbow[1]

            torso_length = np.linalg.norm(np.array(shoulder) - np.array(hip))
            is_trunk_stable = abs(shoulder[0] - hip[0]) < (0.20 * torso_length)

            shoulder_width = abs(r_shoulder.x - l_shoulder.x) * w
            is_side_profile = shoulder_width < (0.35 * torso_length)

            # Evaluate overall form
            good_form = is_elbow_down and is_arm_in_z_plane and is_trunk_stable and is_side_profile and is_elbow_pinned

            if not good_form:
                if not is_elbow_pinned:
                    feedback_msg = "KEEP ELBOW PINNED TO SIDE!"
                elif not is_arm_in_z_plane:
                    feedback_msg = "ARM SWINGING OUT OF PLANE!"
                elif not is_side_profile:
                    feedback_msg = "TURN TO SIDE PROFILE!"
                elif not is_trunk_stable:
                    feedback_msg = "TRUNK SWAY DETECTED!"
                else:
                    feedback_msg = "INCORRECT FORM!"
                self.stage = "error"

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
            self.stage = "error"
            return None, None, False, "Stand fully in frame!"


# ==========================================
# 3. MAIN APPLICATION LOOP
# ==========================================
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

# Instantiate our active exercise
current_exercise = ElbowFlexion()

while True:
    success, img = cap.read()
    if not success:
        break

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = pose.process(img_rgb)

    if results.pose_landmarks:
        mp_draw.draw_landmarks(img, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        h, w, _ = img.shape

        # Pass the landmarks to our active exercise object
        angle, joint_pos, is_good, msg = current_exercise.process_frame(results.pose_landmarks.landmark, h, w)

        # --- DYNAMIC DRAWING ---
        if angle is not None and joint_pos is not None:
            color = (0, 255, 0) if is_good else (0, 0, 255)
            cv2.putText(img, f"{int(angle)} deg",
                        tuple(np.multiply(joint_pos, [1, 1]).astype(int)),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)

        if msg:
            cv2.putText(img, msg, (50, 150),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3, cv2.LINE_AA)

    # --- GUI DISPLAY ---
    cv2.rectangle(img, (0, 0), (250, 73), (245, 117, 16), -1)

    cv2.putText(img, 'REPS', (15, 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.putText(img, str(current_exercise.counter),
                (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, cv2.LINE_AA)

    cv2.putText(img, 'STAGE', (100, 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.putText(img, current_exercise.stage,
                (100, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

    cv2.imshow("Physio Assistant Tracker", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()