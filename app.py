"""
Tkinter application shell.

PhysioApp owns the main menu, the tracking dashboard, and the
VideoWorker lifecycle (start/poll/graceful-stop). The polling loop
(root.after) and the _await_worker_shutdown teardown sequence are
unchanged from the original -- this is what prevents the GUI from
freezing while the camera thread releases its resources.

Depends on constants.py, config.py, engine.py, and worker.py. Nothing
imports app.py back, so this sits at the top of the dependency graph
alongside main.py.
"""
import os
import queue
import sys
import time
import tkinter as tk
from tkinter import font

from PIL import Image, ImageTk

from config import AppConfig
from constants import SESSIONS_DIR, logger
from engine import EXERCISE_REGISTRY
from worker import VideoWorker


class PhysioApp:
    POLL_INTERVAL_MS = 15

    def __init__(self, root):
        self.root = root
        self.app_cfg = AppConfig()
        self.root.title("Intelligent Physiotherapy Assistant")

        # INCREASED WIDTH: Give the text panel more room to breathe
        self.root.geometry("1150x650")
        self.root.configure(bg="#2C3E50")

        self.worker = None
        self.result_queue = queue.Queue(maxsize=self.app_cfg.queue_max_size)
        self._poll_job = None
        self._stopping = False

        self.main_menu_frame = tk.Frame(self.root, bg="#2C3E50")
        self.tracking_frame = tk.Frame(self.root, bg="#2C3E50")

        self.build_main_menu()
        self.build_tracking_dashboard()

        self.main_menu_frame.pack(fill="both", expand=True)
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

    def build_main_menu(self):
        title_font = font.Font(family="Helvetica", size=24, weight="bold")
        btn_font = font.Font(family="Helvetica", size=14)

        tk.Label(self.main_menu_frame, text="Intelligent Physiotherapy Assistant", font=title_font, fg="white",
                 bg="#2C3E50").pack(pady=40)
        tk.Label(self.main_menu_frame, text="Select Exercise Protocol", font=btn_font, fg="#BDC3C7", bg="#2C3E50").pack(
            pady=10)

        self.status_label = tk.Label(self.main_menu_frame, text="", font=font.Font(family="Helvetica", size=11),
                                     fg="#E74C3C", bg="#2C3E50")
        self.status_label.pack(pady=(0, 10))

        # --- UPDATED LOOP START ---
        for slug, exercise_class in EXERCISE_REGISTRY.items():
            ex_name = exercise_class.display_name

            # Fetch the custom labels, falling back to Arm if missing
            lbl_right, lbl_left = getattr(exercise_class, 'button_labels', ("Right Arm", "Left Arm"))

            tk.Button(self.main_menu_frame, text=f"{ex_name} ({lbl_right})", font=btn_font, bg="#2980B9", fg="white",
                      width=30, height=2, command=lambda s=slug: self.start_tracking(s, "right")).pack(pady=5)
            tk.Button(self.main_menu_frame, text=f"{ex_name} ({lbl_left})", font=btn_font, bg="#2980B9", fg="white",
                      width=30, height=2, command=lambda s=slug: self.start_tracking(s, "left")).pack(pady=(5, 15))
        # --- UPDATED LOOP END ---

    def build_tracking_dashboard(self):
        self.video_label = tk.Label(self.tracking_frame, bg="black")
        self.video_label.pack(side="left", padx=20, pady=20)

        info_frame = tk.Frame(self.tracking_frame, bg="#2C3E50")
        info_frame.pack(side="right", fill="both", expand=True, padx=20, pady=20)
        info_font = font.Font(family="Helvetica", size=18, weight="bold")

        self.lbl_reps = tk.Label(info_frame, text="Reps: 0", font=info_font, fg="#2ECC71", bg="#2C3E50")
        self.lbl_reps.pack(pady=20)

        # ADDED: wraplength and justify to force long stage names to two centered lines
        self.lbl_stage = tk.Label(info_frame, text="Stage: N/A", font=info_font, fg="#F1C40F", bg="#2C3E50",
                                  wraplength=300, justify="center")
        self.lbl_stage.pack(pady=20)

        # ADDED: increased wraplength, center justification, and a fixed height of 3 text lines
        self.lbl_feedback = tk.Label(info_frame, text="", font=font.Font(family="Helvetica", size=14, weight="bold"),
                                     fg="#E74C3C", bg="#2C3E50", wraplength=340, height=3, justify="center")
        self.lbl_feedback.pack(pady=40)

        btn_stop = tk.Button(info_frame, text="End Session", font=font.Font(family="Helvetica", size=14), bg="#C0392B",
                             fg="white", width=15, command=self.stop_tracking)
        btn_stop.pack(side="bottom", pady=40)

    def start_tracking(self, exercise_slug, side):
        if self.worker is not None or self._stopping:
            self.status_label.config(text="Previous session is still shutting down, please wait...")
            return

        exercise_cls = EXERCISE_REGISTRY[exercise_slug]
        self.status_label.config(text="")

        self.main_menu_frame.pack_forget()
        self.tracking_frame.pack(fill="both", expand=True)

        session_id = int(time.time())
        csv_name = f"session_{exercise_slug}_{side}_{session_id}.csv"
        csv_path = os.path.join(SESSIONS_DIR, csv_name)

        while not self.result_queue.empty():
            try:
                self.result_queue.get_nowait()
            except queue.Empty:
                break

        self.worker = VideoWorker(side=side, csv_path=csv_path, result_queue=self.result_queue,
                                  app_cfg=self.app_cfg, exercise_cls=exercise_cls)
        self.worker.start()
        self._poll_job = self.root.after(self.POLL_INTERVAL_MS, self._poll_results)

    def _poll_results(self):
        try:
            result = self.result_queue.get_nowait()
        except queue.Empty:
            result = None

        if result is not None:
            if result["type"] == "error":
                self.lbl_feedback.config(text=result["message"], fg="#E74C3C")
            else:
                img_pil = Image.fromarray(result["image"])
                img_tk = ImageTk.PhotoImage(image=img_pil)
                self.video_label.imgtk = img_tk
                self.video_label.configure(image=img_tk)

                self.lbl_reps.config(text=f"Reps: {result['counter']}")
                self.lbl_stage.config(text=f"Stage: {result['stage'].replace('_', ' ').title()}")
                msg = result["msg"]
                self.lbl_feedback.config(text=msg if msg else "Form: Optimal", fg="#E74C3C" if msg else "#2ECC71")

        if self.worker is not None:
            self._poll_job = self.root.after(self.POLL_INTERVAL_MS, self._poll_results)

    def stop_tracking(self, on_close_callback=None):
        if self._poll_job is not None:
            self.root.after_cancel(self._poll_job)
            self._poll_job = None

        if self.worker is not None:
            worker = self.worker
            self._stopping = True
            worker.stop()
            self._await_worker_shutdown(worker, callback=on_close_callback)
        else:
            if on_close_callback:
                on_close_callback()
            else:
                self.tracking_frame.pack_forget()
                self.main_menu_frame.pack(fill="both", expand=True)

    def _await_worker_shutdown(self, worker, attempt=0, callback=None):
        if not worker.is_alive():
            if self.worker is worker:
                self.worker = None
            self._stopping = False
            self.status_label.config(text="")
            logger.info("Previous worker thread confirmed stopped.")
            if callback:
                callback()
            else:
                self.tracking_frame.pack_forget()
                self.main_menu_frame.pack(fill="both", expand=True)
            return

        if attempt == 0:
            self.status_label.config(text="Releasing camera...")

        self.root.after(200, lambda: self._await_worker_shutdown(worker, attempt + 1, callback))

    def on_close(self):
        self.root.withdraw()
        self.stop_tracking(on_close_callback=lambda: sys.exit(0))
