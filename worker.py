"""
Background video/pose worker thread.

Runs the camera capture + MediaPipe inference loop off the Tkinter main
thread and hands frames/results to the GUI through a bounded
queue.Queue (see _push_result / _push_error). Frame pacing uses
time.monotonic() and graceful shutdown is driven by threading.Event,
both unchanged from the original single-file version.

Depends on constants.py (logging + shared constants) and engine.py
(mp_pose/mp_draw + the default ElbowFlexion exercise class) -- it does
NOT import app.py, which is what keeps worker.py safely reusable
(and testable) independent of the GUI.
"""
import csv
import datetime
import os
import queue
import sys
import threading
import time

import cv2
import numpy as np

from config import AppConfig
from constants import (
    COLOR_BAD_RGB,
    COLOR_GOOD_RGB,
    CSV_FLUSH_EVERY_N_FRAMES,
    logger,
)
from engine import ElbowFlexion, mp_draw, mp_pose
import mediapipe as mp

mp_face = mp.solutions.face_detection
class VideoWorker(threading.Thread):
    FACE_MASK_HOLD_SECONDS = 0.6
    FACE_MASK_SMOOTHING_TIME_CONSTANT_S = 0.12
    FACE_MASK_RADIUS_SMOOTHING_TIME_CONSTANT_S = 0.6   # much slower than position

    def __init__(self, side, csv_path, result_queue, app_cfg: AppConfig, config=None, exercise_cls=ElbowFlexion,
                 model_complexity=1):
        super().__init__(daemon=True)
        self.app_cfg = app_cfg
        self.result_queue = result_queue
        self.exercise = exercise_cls(side=side, config=config)
        self.csv_path = csv_path
        self.model_complexity = model_complexity
        self._stop_event = threading.Event()
        self._mask_center = None
        self._mask_radius = None
        self._mask_last_seen = None
        self._mask_last_smooth_time = None

    def stop(self):
        self._stop_event.set()

    def wait_until_stopped(self, timeout=None):
        self.join(timeout=timeout)
        return not self.is_alive()

    def run(self):
        cap = None
        pose = None
        face_detector = None
        csv_file = None
        csv_writer = None
        try:
            if sys.platform.startswith("win"):
                cap = cv2.VideoCapture(self.app_cfg.camera_index, cv2.CAP_DSHOW)
            elif sys.platform.startswith("linux"):
                cap = cv2.VideoCapture(self.app_cfg.camera_index, cv2.CAP_V4L2)
            else:
                cap = cv2.VideoCapture(self.app_cfg.camera_index)

            if not cap.isOpened():
                self._push_error("Could not open camera.")
                return

            pose = mp_pose.Pose(
                model_complexity=self.model_complexity,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
            )

            face_detector = mp_face.FaceDetection(model_selection=1, min_detection_confidence=0.5)

            os.makedirs(os.path.dirname(self.csv_path), exist_ok=True)
            csv_file = open(self.csv_path, mode='w', newline='')
            csv_writer = csv.writer(csv_file)
            csv_writer.writerow(["Timestamp_ISO", "Smoothed_Angle", "Stage", "Good_Form", "Feedback"])

            frames_since_flush = 0
            failed_reads = 0

            while not self._stop_event.is_set():
                loop_start = time.monotonic()
                success, img = cap.read()
                if not success:
                    failed_reads += 1
                    if failed_reads > 50:
                        self._push_error("Camera disconnected or feed lost.")
                        break
                    time.sleep(0.05)
                    continue

                failed_reads = 0

                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                h, w, _ = img.shape
                scale = self.app_cfg.inference_max_dim / float(max(h, w))
                if scale < 1.0:
                    small_w, small_h = int(w * scale), int(h * scale)
                    img_for_inference = cv2.resize(img_rgb, (small_w, small_h), interpolation=cv2.INTER_AREA)
                else:
                    img_for_inference = img_rgb

                results = pose.process(img_for_inference)
                self._mask_any_face(img_rgb, face_detector, h, w,
                                    results.pose_landmarks.landmark if results.pose_landmarks else None)
                angle, is_good, msg = None, False, ""
                stage = self.exercise.stage
                counter = self.exercise.counter

                if results.pose_landmarks:
                    mp_draw.draw_landmarks(img_rgb, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

                    nose = results.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE.value]

                    # --- NEW: GUARD EXECUTED BEFORE THE PHYSICS ENGINE ---
                    if nose.visibility > 0.5 and nose.y < 0.10:
                        is_good = False
                        msg = "STEP BACK!"
                        angle, joint_pos = None, (0, 0)
                    else:
                        angle, joint_pos, is_good, msg = self.exercise.process_frame(results.pose_landmarks.landmark, h, w)

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

                else:
                    is_good = False
                    msg = "NO PERSON DETECTED"

                self._push_result(img_rgb, counter, stage, msg, is_good)

                elapsed = time.monotonic() - loop_start
                remaining = (1.0 / self.app_cfg.target_fps) - elapsed
                if remaining > 0:
                    time.sleep(remaining)

        except Exception:
            logger.exception("Camera/pose worker crashed")
            self._push_error("Camera/pose error - check logs for details.")
        finally:
            if csv_writer is not None:
                csv_writer.writerow([])
                csv_writer.writerow(["--- SESSION SUMMARY ---"])
                csv_writer.writerow(["Total Reps", self.exercise.counter])
                for err_type, err_count in self.exercise.error_counts.items():
                    csv_writer.writerow([f"Error: {err_type}", err_count])
            if csv_file is not None:
                csv_file.close()
            if cap is not None:
                cap.release()
            if pose is not None:
                pose.close()
            if face_detector is not None:
                face_detector.close()

    def _get_face_candidate(self, img_rgb, face_detector, h, w, pose_landmarks):
        if pose_landmarks is not None:
            idx = mp_pose.PoseLandmark
            face_x, face_y = [], []
            for i in range(11):
                lm = pose_landmarks[i]
                if lm.visibility > 0.1:
                    face_x.append(lm.x * w)
                    face_y.append(lm.y * h)

            # Torso length (shoulder-to-hip) stays valid at any body rotation,
            # unlike shoulder width, which is *designed* to collapse toward
            # zero in side profile (see side_profile_max_shoulder_ratio).
            candidates = [
                (idx.RIGHT_SHOULDER, idx.RIGHT_HIP),
                (idx.LEFT_SHOULDER, idx.LEFT_HIP),
            ]
            best = None
            for sh_lm, hip_lm in candidates:
                sh, hip = pose_landmarks[sh_lm.value], pose_landmarks[hip_lm.value]
                vis = min(sh.visibility, hip.visibility)
                if best is None or vis > best[0]:
                    best = (vis, sh, hip)
            vis, sh, hip = best

            if face_x and vis > 0.3:
                torso_len = np.hypot((sh.x - hip.x) * w, (sh.y - hip.y) * h)
                if torso_len > 1e-3:
                    cx = sum(face_x) / len(face_x)
                    cy = sum(face_y) / len(face_y)
                    return cx, cy, torso_len * 0.28  # tune this if too big/small

        result = face_detector.process(img_rgb)
        if result.detections:
            box = result.detections[0].location_data.relative_bounding_box
            cx = (box.xmin + box.width / 2) * w
            cy = (box.ymin + box.height / 2) * h
            r = max(box.width * w, box.height * h) * 0.75
            return cx, cy, r

        return None

    def _mask_any_face(self, img_rgb, face_detector, h, w, pose_landmarks=None):
        now = time.monotonic()
        candidate = self._get_face_candidate(img_rgb, face_detector, h, w, pose_landmarks)

        if candidate is not None:
            cx, cy, r = candidate
            if self._mask_center is None or self._mask_last_smooth_time is None:
                self._mask_center, self._mask_radius = (cx, cy), r
            else:
                dt = max(now - self._mask_last_smooth_time, 0.0)
                alpha_pos = 1.0 - np.exp(-dt / max(self.FACE_MASK_SMOOTHING_TIME_CONSTANT_S, 1e-6))
                alpha_rad = 1.0 - np.exp(-dt / max(self.FACE_MASK_RADIUS_SMOOTHING_TIME_CONSTANT_S, 1e-6))
                px, py = self._mask_center
                self._mask_center = (px + alpha_pos * (cx - px), py + alpha_pos * (cy - py))
                self._mask_radius += alpha_rad * (r - self._mask_radius)
            self._mask_last_smooth_time = now
            self._mask_last_seen = now
        elif self._mask_last_seen is not None and (now - self._mask_last_seen) > self.FACE_MASK_HOLD_SECONDS:
            # No face signal at all for a while -- they've likely stepped fully
            # out of frame, so stop drawing a stale mask.
            self._mask_center, self._mask_radius, self._mask_last_smooth_time = None, None, None

        if self._mask_center is None:
            return

        cx, cy, r = int(self._mask_center[0]), int(self._mask_center[1]), int(self._mask_radius)
        x1, y1 = max(cx - r, 0), max(cy - r, 0)
        x2, y2 = min(cx + r, w), min(cy + r, h)
        if x2 <= x1 or y2 <= y1:
            return

        roi = img_rgb[y1:y2, x1:x2]
        k = max(15, (min(roi.shape[0], roi.shape[1]) // 2) | 1)
        img_rgb[y1:y2, x1:x2] = cv2.GaussianBlur(roi, (k, k), 0)
        cv2.circle(img_rgb, (cx, cy), r, (25, 25, 25), -1)

    def _push_result(self, img_rgb, counter, stage, msg, is_good):
        try:
            self.result_queue.get_nowait()
        except queue.Empty:
            pass
        try:
            self.result_queue.put_nowait({
                "type": "frame",
                "image": img_rgb,
                "counter": counter,
                "stage": stage,
                "msg": msg,
                "is_good": is_good,
            })
        except queue.Full:
            pass

    def _push_error(self, message):
        try:
            self.result_queue.put_nowait({"type": "error", "message": message})
        except queue.Full:
            pass
