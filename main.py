import cv2
import mediapipe as mp
import numpy as np

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_draw = mp.solutions.drawing_utils


def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    ba = a - b
    bc = c - b

    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    cosine_angle = np.clip(cosine_angle, -1.0, 1.0)

    angle = np.arccos(cosine_angle)
    return np.degrees(angle)


cap = cv2.VideoCapture(0)

smoothed_angle = None
smoothing_factor = 0.2

counter = 0
stage = None

while True:
    success, img = cap.read()
    if not success:
        break

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = pose.process(img_rgb)

    if results.pose_landmarks:
        mp_draw.draw_landmarks(img, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        landmarks = results.pose_landmarks.landmark

        r_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
        r_elbow = landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value]
        r_wrist = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value]

        h, w, c = img.shape

        if r_shoulder.visibility > 0.5 and r_elbow.visibility > 0.5 and r_wrist.visibility > 0.5:

            shoulder = [r_shoulder.x * w, r_shoulder.y * h]
            elbow = [r_elbow.x * w, r_elbow.y * h]
            wrist = [r_wrist.x * w, r_wrist.y * h]

            raw_angle = calculate_angle(shoulder, elbow, wrist)

            if smoothed_angle is None:
                smoothed_angle = raw_angle
            else:
                smoothed_angle = (smoothing_factor * raw_angle) + ((1 - smoothing_factor) * smoothed_angle)

            # --- SPATIAL AWARENESS CHECKS ---
            upper_arm_length = np.linalg.norm(np.array(shoulder) - np.array(elbow))
            forearm_length = np.linalg.norm(np.array(elbow) - np.array(wrist))

            is_elbow_down = elbow[1] > shoulder[1]
            is_wrist_down = wrist[1] > elbow[1]

            # Increased strictness to 0.65 to catch the "flexing toward camera" cheat faster
            is_in_plane = (forearm_length / upper_arm_length) > 0.65

            good_form = is_elbow_down and is_in_plane

            if not good_form:
                cv2.putText(img, "INCORRECT FORM!", (50, 150),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3, cv2.LINE_AA)
                # THE PENALTY BOX: Instantly invalidate the current rep
                stage = "error"

            cv2.putText(img, f"{int(smoothed_angle)} deg",
                        tuple(np.multiply(elbow, [1, 1]).astype(int)),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0) if good_form else (0, 0, 255), 2, cv2.LINE_AA)

            # --- RUTHLESS STATE MACHINE LOGIC ---
            if good_form:
                # The reset line: You must return here to clear an error or finish a rep
                if smoothed_angle > 150 and is_wrist_down:
                    if stage == 'flexed':
                        counter += 1
                        print(f"Rep completed! Total: {counter}")
                    # Resets the machine to 'extended' whether you came from a good rep or an error
                    stage = "extended"

                    # You can only enter the 'flexed' state if your previous state was perfectly 'extended'
                elif smoothed_angle < 40 and stage == 'extended':
                    stage = "flexed"

        else:
            cv2.putText(img, "Tracking Lost!", (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
            # If the AI loses sight of your joints, invalidate the rep just to be safe
            stage = "error"

    # --- GUI DISPLAY ---
    cv2.rectangle(img, (0, 0), (250, 73), (245, 117, 16), -1)

    cv2.putText(img, 'REPS', (15, 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.putText(img, str(counter),
                (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, cv2.LINE_AA)

    cv2.putText(img, 'STAGE', (100, 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.putText(img, stage if stage else "waiting",
                (100, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

    cv2.imshow("Physio Assistant Tracker", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()