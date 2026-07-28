import cv2
import mediapipe as mp
import numpy as np

# Initialize MediaPipe Pose tracking
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_draw = mp.solutions.drawing_utils


def calculate_angle(a, b, c):
    """
    Calculates the angle between three points using vectors.
    a, b, c are (x, y) coordinates. b is the vertex (e.g., elbow).
    """
    a = np.array(a)  # First point (e.g., shoulder)
    b = np.array(b)  # Mid point (e.g., elbow)
    c = np.array(c)  # End point (e.g., wrist)

    # Create vectors BA and BC
    ba = a - b
    bc = c - b

    # Calculate cosine of the angle using the dot product formula
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))

    # Keep the cosine value within the valid range [-1, 1] to avoid math errors
    cosine_angle = np.clip(cosine_angle, -1.0, 1.0)

    # Calculate the angle in radians, then convert to degrees
    angle = np.arccos(cosine_angle)
    return np.degrees(angle)


cap = cv2.VideoCapture(0)

# Initialize a variable to hold the smoothed angle over time
smoothed_angle = None
# Set the smoothing factor (0.0 to 1.0).
# Lower = smoother but slightly delayed. Higher = faster but more jitter.
smoothing_factor = 0.2

while True:
    success, img = cap.read()
    if not success:
        break

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = pose.process(img_rgb)

    if results.pose_landmarks:
        mp_draw.draw_landmarks(img, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        landmarks = results.pose_landmarks.landmark

        # Extract landmarks and their visibility scores
        r_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
        r_elbow = landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value]
        r_wrist = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value]

        h, w, c = img.shape

        # 1. Visibility Filter (Das et al., 2025)
        # Only calculate if the AI is at least 70% confident it sees all three joints
        if r_shoulder.visibility > 0.7 and r_elbow.visibility > 0.7 and r_wrist.visibility > 0.7:

            shoulder = [r_shoulder.x * w, r_shoulder.y * h]
            elbow = [r_elbow.x * w, r_elbow.y * h]
            wrist = [r_wrist.x * w, r_wrist.y * h]

            # Calculate the raw, jittery angle
            raw_angle = calculate_angle(shoulder, elbow, wrist)

            # 2. Signal Smoothing Filter (Song et al., 2023)
            if smoothed_angle is None:
                smoothed_angle = raw_angle  # First frame
            else:
                # Apply Exponential Moving Average
                smoothed_angle = (smoothing_factor * raw_angle) + ((1 - smoothing_factor) * smoothed_angle)

            # Display the SMOOTHED angle at the elbow's position
            cv2.putText(img, f"{int(smoothed_angle)} deg",
                        tuple(np.multiply(elbow, [1, 1]).astype(int)),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        else:
            # If visibility is too low (e.g. extreme self-occlusion), warn the user
            cv2.putText(img, "Tracking Lost!", (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

    cv2.imshow("Physio Assistant Tracker", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()