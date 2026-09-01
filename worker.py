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


class VideoWorker(threading.Thread):
    def __init__(self, side, csv_path, result_queue, app_cfg: AppConfig, config=None, exercise_cls=ElbowFlexion,
                 model_complexity=1):
        super().__init__(daemon=True)
        self.app_cfg = app_cfg
        self.result_queue = result_queue
        self.exercise = exercise_cls(side=side, config=config)
        self.csv_path = csv_path
        self.model_complexity = model_complexity
        self._stop_event = threading.Event()

    def stop(self):
        self._stop_event.set()

    def wait_until_stopped(self, timeout=None):
        self.join(timeout=timeout)
        return not self.is_alive()

    def run(self):
        cap = None
        pose = None
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
                angle, is_good, msg = None, False, ""
                stage = self.exercise.stage
                counter = self.exercise.counter

                if results.pose_landmarks:
                    self._mask_face(results.pose_landmarks.landmark, img_rgb, h, w)
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

    def _mask_face(self, landmarks, img_rgb, h, w):
        face_x = []
        face_y = []

        for i in range(11):
            lm = landmarks[i]
            if lm.visibility > 0.1:
                face_x.append(lm.x * w)
                face_y.append(lm.y * h)

        if not face_x:
            return

        cx = int(sum(face_x) / len(face_x))
        cy = int(sum(face_y) / len(face_y))

        dynamic_radius = int(h * 0.12)

        cv2.circle(img_rgb, (cx, cy), dynamic_radius, (25, 25, 25), -1)

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
