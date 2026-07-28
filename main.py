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

while True:
    success, img = cap.read()
    if not success:
        break

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = pose.process(img_rgb)

    if results.pose_landmarks:
        # Draw the skeleton
        mp_draw.draw_landmarks(img, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        # Extract all landmarks
        landmarks = results.pose_landmarks.landmark

        # Get coordinates for Right Shoulder (12), Right Elbow (14), and Right Wrist (16)
        # We multiply by image width/height to get actual pixel coordinates
        h, w, c = img.shape
        shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x * w,
                    landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y * h]
        elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x * w,
                 landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y * h]
        wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x * w,
                 landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y * h]

        # Calculate the angle
        angle = calculate_angle(shoulder, elbow, wrist)

        # Display the angle at the elbow's position
        cv2.putText(img, str(int(angle)),
                    tuple(np.multiply(elbow, [1, 1]).astype(int)),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

    cv2.imshow("Physio Assistant Tracker", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()