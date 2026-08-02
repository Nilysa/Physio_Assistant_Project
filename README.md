# Physio_Assistant_Project

## Overview
The Physio Assistant is an intelligent, edge-computing tele-rehabilitation application designed for real-time motion monitoring[cite: 3]. Built on machine vision, it tracks 33 structural body keypoints using a single consumer webcam, eliminating the need for physical wearable sensors[cite: 2, 3]. To ensure strict patient privacy and high-performance execution without cloud latency, all kinematic calculations and video processing pipelines are executed entirely locally[cite: 1, 3].

### Tech Stack
*   **Language:** Python 3.12
*   **Computer Vision Engine:** OpenCV (4.11.0)
*   **Pose Estimation AI:** MediaPipe (0.10.21)
*   **Mathematics & Matrices:** NumPy (1.26.4)[cite: 1]

---

## System Architecture

The application utilizes a decoupled, Object-Oriented software architecture to isolate the core computer vision loop from the exercise-specific mathematical constraints[cite: 1]. 

1.  **Frame Acquisition:** Video feed is captured frame-by-frame via `cv2.VideoCapture(0)` and undergoes color-space conversion from BGR to RGB[cite: 1].
2.  **Inference:** Frames are passed to the MediaPipe Pose legacy API (`mp.solutions`) for structural landmark detection, chosen to maintain architectural simplicity and support clean vector mathematics[cite: 1].
3.  **Exercise Engine:** The extracted data flows into a modular class hierarchy, utilizing a `BaseExercise` parent class and specialized subclasses (e.g., `ElbowFlexion`)[cite: 1].

### Landmark Topology Reference
The system utilizes MediaPipe's 33-point topological skeletal map to construct the necessary kinematic vectors.

![MediaPipe Pose Landmark Topology](images/mediapipe_topology.png)

---

## Technical Decisions & Biomechanical Logic

To transition raw AI inference into a reliable, clinical-grade tool, several mathematical and filtering constraints were implemented:

*   **Kinematic Angle Engine:** Joint angles are calculated in real-time utilizing 2D vector dot product geometry[cite: 1]. The mathematical backbone of the system extracts the angle $\theta$ between three coordinates (e.g., shoulder, elbow, wrist) using the formula $\theta = \arccos(\frac{\vec{BA} \cdot \vec{BC}}{|\vec{BA}||\vec{BC}|})$[cite: 1].
*   **Signal Processing:** To mitigate the inherent pixel jitter and high-frequency noise of standard webcams, an Exponential Moving Average (EMA) digital filter is applied to stabilize all angle outputs[cite: 1].
*   **Dynamic Visibility Scaling:** The system dynamically adjusts its confidence requirements, enforcing a strict 0.7 visibility threshold during calibration to secure baselines, while relaxing the threshold to 0.4 during active workouts to prevent erroneous penalties caused by natural motion blur[cite: 1].
*   **State Machine & Spatial Constraints:** A strict state machine tracks repetition validity[cite: 1]. Form breakdown is mathematically identified via multiple constraints:
    *   **Trunk Sway Lock:** Tracks horizontal drift between the shoulder and hip to penalize backward momentum[cite: 1].
    *   **Z-Axis Depth Lock:** Identifies out-of-plane deviations (foreshortening) by combining a 2D arm-to-torso ratio lock with a gross Z-axis depth threshold, locked at a generalized 25% tolerance[cite: 1].
*   **Dynamic Auto-Calibration:** Hardcoded clinical angles were abandoned in favor of an active calibration phase[cite: 1]. The software reads a user's initial extended and flexed positions to adapt universally to individual limb proportions and personal Range of Motion (ROM) limits[cite: 1].

---

## Quickstart Guide

### 1. Environment Setup
It is highly recommended to use a virtual environment to manage dependencies. Ensure you are using **Python 3.12** for wheel compatibility[cite: 1].

```bash
# Create and activate virtual environment
python3.12 -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
```

### 2. Installation

Install the core requirements via pip[cite: 1].

```Bash
pip install mediapipe==0.10.21 opencv-python==4.11.0 numpy==1.26.4
```

(Note: If you encounter network proxy issues during installation in PowerShell, clear active proxy variables using $env:HTTP_PROXY = $null[cite: 1].)
### 3. Execution

Run the main script to initialize the webcam and begin tracking. Please ensure your camera is positioned at a 90-degree sagittal (side-profile) angle and you are standing far back enough to capture your full torso and limbs[cite: 1].

```Bash
python main.py
```