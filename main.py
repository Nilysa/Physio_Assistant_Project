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

        self.calib_frames = 0
        self.flex_calib_frames = 0

        self.active_error_msg = ""
        self.error_timer = 0

    def process_frame(self, landmarks, h, w):
        r_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
        l_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
        r_elbow = landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value]
        r_wrist = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value]
        r_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value]

        feedback_msg = ""
        good_form = True

        req_vis = 0.7 if self.stage.startswith("calib") else 0.4

        if r_shoulder.visibility > req_vis and r_elbow.visibility > req_vis and r_wrist.visibility > req_vis and r_hip.visibility > req_vis:

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

            upper_arm_length = np.linalg.norm(np.array(shoulder) - np.array(elbow))
            forearm_length = np.linalg.norm(np.array(elbow) - np.array(wrist))
            torso_length = np.linalg.norm(np.array(shoulder) - np.array(hip))

            upper_ratio = upper_arm_length / torso_length
            forearm_ratio = forearm_length / torso_length

            if self.stage.startswith("calibration"):
                if self.stage == "calibration_start":
                    if self.smoothed_angle > 140 and wrist[1] > elbow[1]:
                        self.calib_frames += 1
                        seconds_left = 3 - (self.calib_frames // 30)
                        feedback_msg = f"STAND STILL: {seconds_left}s"

                        if self.calib_frames >= 90:
                            self.baseline_upper_ratio = upper_ratio
                            self.baseline_forearm_ratio = forearm_ratio

                            self.target_ext = self.smoothed_angle - 15
                            self.stage = "calibration_transition"
                            self.calib_frames = 0
                    else:
                        self.calib_frames = 0
                        feedback_msg = "DROP ARM STRAIGHT TO START"

                elif self.stage == "calibration_transition":
                    feedback_msg = "SUCCESS! NOW BEND ARM..."
                    self.calib_frames += 1
                    if self.calib_frames > 30:
                        self.stage = "calibration_flex"
                        self.calib_frames = 0

                elif self.stage == "calibration_flex":
                    if self.smoothed_angle < self.target_flex:
                        self.target_flex = self.smoothed_angle

                    if self.smoothed_angle < 60:
                        self.flex_calib_frames += 1
                        seconds_left = 2 - (self.flex_calib_frames // 30)
                        feedback_msg = f"HOLD THIS FLEX: {max(0, seconds_left)}s"

                        if self.flex_calib_frames >= 60:
                            self.target_flex += 15
                            self.stage = "extended"
                    else:
                        self.flex_calib_frames = 0
                        feedback_msg = "BEND ARM FULLY"

                return self.smoothed_angle, elbow, True, feedback_msg

            # --- ADJUSTED TOLERANCES ---
            is_upper_arm_2d_stable = upper_ratio > (self.baseline_upper_ratio * 0.75)
            is_forearm_2d_stable = forearm_ratio > (self.baseline_forearm_ratio * 0.75)

            wrist_depth_diff = abs(r_wrist.z - r_shoulder.z)
            elbow_depth_diff = abs(r_elbow.z - r_shoulder.z)
            is_arm_in_z_plane = wrist_depth_diff < 0.40 and elbow_depth_diff < 0.30

            is_in_plane = is_upper_arm_2d_stable and is_forearm_2d_stable and is_arm_in_z_plane

            dx = elbow[0] - shoulder[0]
            dy = elbow[1] - shoulder[1]
            vertical_angle = np.degrees(np.arctan2(abs(dx), max(dy, 1)))
            is_elbow_pinned = vertical_angle < 20

            is_elbow_down = elbow[1] > shoulder[1]
            is_wrist_down = wrist[1] > elbow[1]

            is_trunk_stable = abs(shoulder[0] - hip[0]) < (0.20 * torso_length)
            shoulder_width = abs(r_shoulder.x - l_shoulder.x) * w
            is_side_profile = shoulder_width < (0.35 * torso_length)

            good_form = is_elbow_down and is_in_plane and is_trunk_stable and is_side_profile and is_elbow_pinned

            if not good_form:
                self.stage = "error"
                if self.error_timer == 0:
                    if not is_elbow_pinned:
                        self.active_error_msg = "KEEP ELBOW PINNED TO SIDE!"
                    elif not is_in_plane:
                        self.active_error_msg = "ARM SWINGING OUT OF PLANE!"
                    elif not is_side_profile:
                        self.active_error_msg = "TURN TO SIDE PROFILE!"
                    elif not is_trunk_stable:
                        self.active_error_msg = "TRUNK SWAY DETECTED!"
                    else:
                        self.active_error_msg = "INCORRECT FORM!"

                    self.error_timer = 45

            if self.error_timer > 0:
                feedback_msg = self.active_error_msg
                self.error_timer -= 1

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

            lost_msg = "Stand fully in frame to calibrate!" if self.stage.startswith(
                "calib") else "Stand fully in frame!"
            return None, None, False, lost_msg


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