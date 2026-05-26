import os
import time
import queue
import threading

import cv2
import numpy as np
from PIL import Image
import customtkinter as ctk
from tkinter import filedialog, messagebox

from .state import AppState
from .detector import ObjectDetector
from .homography import HomographyTransformer
from .tracker import SpeedTracker
from .logger import ViolationManager


CANVAS_W = 854
CANVAS_H = 480
UI_REFRESH_MS = 33


class SpeedRadarGUI(ctk.CTk):

    # -----------------------------------------------------------------
    #  Initialisation
    # -----------------------------------------------------------------
    def __init__(self):
        super().__init__()

        self.title("CV Speed Radar & Traffic Violation System")
        self.geometry("1320x820")
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")

        # -- application state ----------------------------------------
        self.app_state = AppState.IDLE
        self.running = False
        self.proc_thread = None
        self.cap = None

        # -- thread-safe communication --------------------------------
        self.frame_queue = queue.Queue(maxsize=2)
        self.violation_queue = queue.Queue()
        self.stop_event = threading.Event()

        # -- shared config (protected by lock) ------------------------
        self._cfg = {
            'speed_limit': 60,
            'confidence': 0.50,
            'pixel_to_meter': 0.05,
            'source_type': 'webcam',
            'source_path': None,
        }
        self._cfg_lock = threading.Lock()

        # -- CV modules -----------------------------------------------
        self.detector = ObjectDetector(confidence=self._cfg['confidence'])
        self.homography = HomographyTransformer()
        self.tracker = SpeedTracker(self.homography)
        self.logger = ViolationManager()

        # -- fps book-keeping -----------------------------------------
        self._fps_samples = []
        self._last_fps_time = time.perf_counter()

        # -- ui -------------------------------------------------------
        self._build_ui()
        self.protocol("WM_DELETE_WINDOW", self._on_closing)
        self.after(UI_REFRESH_MS, self._update_display)

    # -----------------------------------------------------------------
    #  UI construction
    # -----------------------------------------------------------------
    def _build_ui(self):
        self.grid_columnconfigure(0, weight=3)
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=4)
        self.grid_rowconfigure(1, weight=1)

        # -- left: video canvas ---------------------------------------
        self.video_frame = ctk.CTkFrame(self, corner_radius=10)
        self.video_frame.grid(row=0, column=0, sticky="nsew",
                              padx=(8, 4), pady=(8, 4))
        self.video_frame.grid_rowconfigure(0, weight=1)
        self.video_frame.grid_columnconfigure(0, weight=1)

        self.video_label = ctk.CTkLabel(
            self.video_frame,
            text="No input source selected.\n"
                 "Select a source below and click ▶ Start.",
            font=ctk.CTkFont(size=15),
            justify="center",
        )
        self.video_label.grid(row=0, column=0, sticky="nsew",
                              padx=10, pady=10)

        # -- right: control panel -------------------------------------
        ctrl = ctk.CTkScrollableFrame(self, corner_radius=10,
                                       width=320)
        ctrl.grid(row=0, column=1, sticky="nsew",
                  padx=(4, 8), pady=(8, 4))
        ctrl.grid_columnconfigure(0, weight=1)
        self._build_controls(ctrl)

        # -- bottom: violation gallery --------------------------------
        self._build_gallery()

    # -- controls -----------------------------------------------------
    def _build_controls(self, parent):
        # ----- input source ------------------------------------------
        box = ctk.CTkFrame(parent, corner_radius=8)
        box.pack(fill="x", padx=6, pady=(6, 3))
        ctk.CTkLabel(box, text="Input Source",
                     font=ctk.CTkFont(size=13, weight="bold")
                     ).pack(anchor="w", padx=10, pady=(8, 4))

        self._src_var = ctk.StringVar(value="webcam")

        for val, lbl in [("webcam", "Live Webcam"),
                          ("image",  "Image Snapshot"),
                          ("video",  "MP4 Video File")]:
            rb = ctk.CTkRadioButton(box, text=lbl, variable=self._src_var,
                                     value=val,
                                     command=self._on_source_change)
            rb.pack(anchor="w", padx=16, pady=1)

        self._browse_btn = ctk.CTkButton(box, text="Browse ...",
                                          command=self._browse_file,
                                          state="disabled")
        self._browse_btn.pack(anchor="w", padx=16, pady=(5, 2))
        self._src_label = ctk.CTkLabel(box, text="",
                                        font=ctk.CTkFont(size=11))
        self._src_label.pack(anchor="w", padx=16, pady=(0, 8))

        # ----- detection settings ------------------------------------
        box = ctk.CTkFrame(parent, corner_radius=8)
        box.pack(fill="x", padx=6, pady=3)
        ctk.CTkLabel(box, text="Detection Settings",
                     font=ctk.CTkFont(size=13, weight="bold")
                     ).pack(anchor="w", padx=10, pady=(8, 4))

        self._speed_label = ctk.CTkLabel(
            box, text=f"Speed Limit: {self._cfg['speed_limit']} km/h",
            font=ctk.CTkFont(size=11))
        self._speed_label.pack(anchor="w", padx=16, pady=(2, 0))
        self._speed_slider = ctk.CTkSlider(
            box, from_=10, to=200, number_of_steps=190,
            command=lambda v: self._on_slider_change("speed_limit", v,
                                                      self._speed_label,
                                                      "Speed Limit: {} km/h",
                                                      int))
        self._speed_slider.set(self._cfg['speed_limit'])
        self._speed_slider.pack(fill="x", padx=16, pady=2)

        self._conf_label = ctk.CTkLabel(
            box, text=f"Confidence: {self._cfg['confidence']:.2f}",
            font=ctk.CTkFont(size=11))
        self._conf_label.pack(anchor="w", padx=16, pady=(6, 0))
        self._conf_slider = ctk.CTkSlider(
            box, from_=0.10, to=0.95, number_of_steps=85,
            command=lambda v: self._on_slider_change("confidence", v,
                                                      self._conf_label,
                                                      "Confidence: {:.2f}",
                                                      float))
        self._conf_slider.set(self._cfg['confidence'])
        self._conf_slider.pack(fill="x", padx=16, pady=2)

        # ----- homography -------------------------------------------
        box = ctk.CTkFrame(parent, corner_radius=8)
        box.pack(fill="x", padx=6, pady=3)
        ctk.CTkLabel(box, text="Homography Calibration",
                     font=ctk.CTkFont(size=13, weight="bold")
                     ).pack(anchor="w", padx=10, pady=(8, 4))

        self._scale_label = ctk.CTkLabel(
            box,
            text=f"Px-to-Meter: {self._cfg['pixel_to_meter']:.4f}",
            font=ctk.CTkFont(size=11))
        self._scale_label.pack(anchor="w", padx=16, pady=(2, 0))
        self._scale_slider = ctk.CTkSlider(
            box, from_=0.001, to=0.500, number_of_steps=499,
            command=lambda v: self._on_slider_change("pixel_to_meter", v,
                                                      self._scale_label,
                                                      "Px-to-Meter: {:.4f}",
                                                      float))
        self._scale_slider.set(self._cfg['pixel_to_meter'])
        self._scale_slider.pack(fill="x", padx=16, pady=2)

        # ----- status -----------------------------------------------
        box = ctk.CTkFrame(parent, corner_radius=8)
        box.pack(fill="x", padx=6, pady=(3, 6))
        ctk.CTkLabel(box, text="System Status",
                     font=ctk.CTkFont(size=13, weight="bold")
                     ).pack(anchor="w", padx=10, pady=(8, 4))

        self._state_lbl = ctk.CTkLabel(
            box, text=f"State: {self.app_state.value}",
            font=ctk.CTkFont(size=12))
        self._state_lbl.pack(anchor="w", padx=16, pady=1)

        self._fps_lbl = ctk.CTkLabel(box, text="FPS: 0",
                                      font=ctk.CTkFont(size=12))
        self._fps_lbl.pack(anchor="w", padx=16, pady=1)

        self._violcnt_lbl = ctk.CTkLabel(box, text="Violations: 0",
                                          font=ctk.CTkFont(size=12))
        self._violcnt_lbl.pack(anchor="w", padx=16, pady=1)

        # -- start / stop buttons
        btn_row = ctk.CTkFrame(box, fg_color="transparent")
        btn_row.pack(fill="x", padx=10, pady=(8, 10))

        self._start_btn = ctk.CTkButton(
            btn_row, text="\u25b6  Start",
            command=self._start_processing,
            fg_color="#2d7a3a", hover_color="#236b2e")
        self._start_btn.pack(side="left", fill="x", expand=True,
                              padx=(0, 4))

        self._stop_btn = ctk.CTkButton(
            btn_row, text="\u25a0  Stop",
            command=self._stop_processing,
            fg_color="#7a2d2d", hover_color="#6b2323",
            state="disabled")
        self._stop_btn.pack(side="right", fill="x", expand=True,
                             padx=(4, 0))

    # -- gallery ------------------------------------------------------
    def _build_gallery(self):
        gframe = ctk.CTkFrame(self, corner_radius=10)
        gframe.grid(row=1, column=0, columnspan=2, sticky="nsew",
                    padx=8, pady=(4, 8))
        gframe.grid_columnconfigure(0, weight=1)
        gframe.grid_rowconfigure(1, weight=1)

        ctk.CTkLabel(gframe, text="Violation History Gallery",
                     font=ctk.CTkFont(size=14, weight="bold")
                     ).grid(row=0, column=0, sticky="w",
                            padx=12, pady=(8, 2))

        self._gallery = ctk.CTkScrollableFrame(
            gframe, orientation="horizontal", height=145,
            corner_radius=8)
        self._gallery.grid(row=1, column=0, sticky="ew",
                           padx=10, pady=(2, 8))
        self._gallery_imgs = []

    # -----------------------------------------------------------------
    #  Slider callbacks
    # -----------------------------------------------------------------
    def _on_slider_change(self, key, value, label, fmt, cast):
        v = cast(value)
        with self._cfg_lock:
            self._cfg[key] = v
        label.configure(text=fmt.format(v))

    def _on_source_change(self):
        st = self._src_var.get()
        with self._cfg_lock:
            self._cfg['source_type'] = st
            if st == 'webcam':
                self._cfg['source_path'] = None
        self._src_label.configure(text="")
        self._browse_btn.configure(state="normal" if st != "webcam"
                                   else "disabled")

    # -----------------------------------------------------------------
    #  File browser
    # -----------------------------------------------------------------
    def _browse_file(self):
        st = self._src_var.get()
        if st == "image":
            ft = [("Images", "*.jpg *.jpeg *.png *.bmp")]
        elif st == "video":
            ft = [("Videos", "*.mp4 *.avi *.mov *.mkv")]
        else:
            return

        path = filedialog.askopenfilename(filetypes=ft)
        if not path:
            return
        with self._cfg_lock:
            self._cfg['source_path'] = path
        self._src_label.configure(text=os.path.basename(path))

    # -----------------------------------------------------------------
    #  Start / Stop processing
    # -----------------------------------------------------------------
    def _start_processing(self):
        if self.running:
            return

        with self._cfg_lock:
            st = self._cfg['source_type']
            sp = self._cfg['source_path']

        if st == "webcam":
            self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        elif st == "video":
            if not sp or not os.path.isfile(sp):
                messagebox.showerror("Error", "Please select a video file.")
                return
            self.cap = cv2.VideoCapture(sp)
        elif st == "image":
            if not sp or not os.path.isfile(sp):
                messagebox.showerror("Error", "Please select an image file.")
                return
            self.cap = None  # handled separately

        if st != "image":
            if self.cap is None or not self.cap.isOpened():
                messagebox.showerror("Error",
                                     "Could not open video source.")
                return

        self.app_state = AppState.TRACKING
        self.running = True
        self.stop_event.clear()
        self.tracker.reset()
        self._update_state_display()

        self._start_btn.configure(state="disabled")
        self._stop_btn.configure(state="normal")

        if st == "image":
            self.proc_thread = threading.Thread(
                target=self._process_image, args=(sp,),
                daemon=True)
        else:
            self.proc_thread = threading.Thread(
                target=self._process_loop, daemon=True)

        self.proc_thread.start()

    def _stop_processing(self):
        self.stop_event.set()
        self.running = False

    # -----------------------------------------------------------------
    #  Background: video processing loop
    # -----------------------------------------------------------------
    def _process_loop(self):
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            fps = 30.0
        frame_interval = 1.0 / fps

        self.homography.set_frame_size(
            int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))

        while not self.stop_event.is_set():
            ret, frame = self.cap.read()
            if not ret:
                break

            t0 = time.perf_counter()

            # -- read current config ----------------------------------
            with self._cfg_lock:
                self.detector.confidence = self._cfg['confidence']
                scale = self._cfg['pixel_to_meter']
                speed_limit = self._cfg['speed_limit']

            # -- object detection -------------------------------------
            detections = self.detector.detect(frame)

            # -- tracking & speed calculation -------------------------
            tracks = self.tracker.update(detections, scale)

            # -- annotate frame ---------------------------------------
            annotated = self._draw_overlays(frame, tracks,
                                             speed_limit, scale)

            # -- check violations -------------------------------------
            violation_this_frame = False
            for tid, track in tracks.items():
                if track.smoothed_speed_kmh > speed_limit \
                        and track.smoothed_speed_kmh > 1.0 \
                        and not track.violation_logged:
                    self.app_state = AppState.VIOLATION_TRIGGERED
                    violation_this_frame = True
                    path = self.logger.log(
                        frame, track.last_bbox, track.class_name,
                        track.smoothed_speed_kmh, speed_limit)
                    if path:
                        try:
                            self.violation_queue.put_nowait(path)
                        except queue.Full:
                            pass
                        track.violation_logged = True
                elif track.smoothed_speed_kmh <= speed_limit * 0.8:
                    track.violation_logged = False

            if not violation_this_frame and self.app_state == AppState.VIOLATION_TRIGGERED:
                self.app_state = AppState.TRACKING

            # -- push to display queue --------------------------------
            try:
                self.frame_queue.put_nowait(annotated)
            except queue.Full:
                pass

            self._update_state_display()

            # -- maintain approximate real-time playback --------------
            elapsed = time.perf_counter() - t0
            sleep = max(0.0, frame_interval - elapsed)
            if sleep > 0.001:
                time.sleep(sleep)

        self.cap.release()
        self.cap = None
        self.running = False
        self.app_state = AppState.IDLE
        self._update_state_display()

        self.after(0, self._enable_start_btn)

    # -----------------------------------------------------------------
    #  Background: image processing (single frame)
    # -----------------------------------------------------------------
    def _process_image(self, path):
        frame = cv2.imread(path)
        if frame is None:
            self.app_state = AppState.ERROR
            self._update_state_display()
            self.after(0, lambda: messagebox.showerror(
                "Error", "Could not read image file."))
            self.running = False
            self.app_state = AppState.IDLE
            self.after(0, self._enable_start_btn)
            return

        self.homography.set_frame_size(frame.shape[1], frame.shape[0])
        detections = self.detector.detect(frame)
        annotated = frame.copy()

        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            cv2.rectangle(annotated, (x1, y1), (x2, y2),
                          (0, 255, 0), 2)
            cx, cy = det['centroid']
            cv2.circle(annotated, (cx, cy), 5, (0, 255, 255), -1)
            cv2.putText(annotated,
                        f"{det['class_name']} {det['confidence']:.2f}",
                        (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.55, (0, 255, 0), 2)

        cv2.putText(annotated, "SNAPSHOT MODE", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

        try:
            self.frame_queue.put_nowait(annotated)
        except queue.Full:
            pass

        self.running = False
        self.app_state = AppState.IDLE
        self._update_state_display()
        self.after(0, self._enable_start_btn)

    # -----------------------------------------------------------------
    #  Frame annotation helper
    # -----------------------------------------------------------------
    def _draw_overlays(self, frame, tracks, speed_limit, scale):
        out = frame.copy()
        h, w = frame.shape[:2]

        # homography source region
        src_abs = self.homography.get_absolute_src()
        if src_abs is not None:
            pts = src_abs.reshape((-1, 1, 2)).astype(np.int32)
            cv2.polylines(out, [pts], True, (255, 0, 255), 2)
            for i, (px, py) in enumerate(src_abs):
                cv2.circle(out, (int(px), int(py)), 6,
                           (255, 0, 255), -1)
                cv2.putText(out, str(i + 1),
                            (int(px) + 8, int(py) + 4),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6, (255, 0, 255), 2)

        # tracked objects
        for tid, track in tracks.items():
            if track.disappeared > 5:
                continue

            speed = track.smoothed_speed_kmh
            x1, y1, x2, y2 = track.last_bbox

            is_over = speed > speed_limit and speed > 1.0
            color = (0, 0, 255) if is_over else (0, 255, 0)

            cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
            cx, cy = track.centroid
            cv2.circle(out, (cx, cy), 5, (255, 255, 0), -1)

            # speed vector (trail)
            if len(track.positions) >= 3:
                p_prev = track.positions[-3][0]
                cv2.arrowedLine(out, (cx, cy),
                                (int(cx + (cx - p_prev[0]) * 2),
                                 int(cy + (cy - p_prev[1]) * 2)),
                                (255, 255, 0), 2, tipLength=0.3)

            lbl = f"{track.class_name} {speed:.1f} km/h"
            (tw, th), _ = cv2.getTextSize(lbl, cv2.FONT_HERSHEY_SIMPLEX,
                                          0.5, 2)
            cv2.rectangle(out, (x1, y1 - th - 6),
                          (x1 + tw + 4, y1), color, -1)
            cv2.putText(out, lbl, (x1 + 2, y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (255, 255, 255), 2)

        # corner info
        cv2.putText(out, f"Limit: {speed_limit} km/h  |  "
                    f"Scale: {scale:.4f}",
                    (10, h - 12), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (200, 200, 200), 1)

        return out

    # -----------------------------------------------------------------
    #  GUI periodic update (runs on main thread)
    # -----------------------------------------------------------------
    def _update_display(self):
        # -- new frame from processor ---------------------------------
        try:
            frame = self.frame_queue.get_nowait()

            cw = max(self.video_frame.winfo_width() - 20, 320)
            ch = max(self.video_frame.winfo_height() - 20, 240)
            fh, fw = frame.shape[:2]
            scale = min(cw / fw, ch / fh)
            dw, dh = int(fw * scale), int(fh * scale)

            disp = cv2.resize(frame, (dw, dh))
            rgb = cv2.cvtColor(disp, cv2.COLOR_BGR2RGB)
            pil = Image.fromarray(rgb)
            ctk_img = ctk.CTkImage(pil, size=(dw, dh))
            self.video_label.configure(image=ctk_img, text="")
            self.video_label.image = ctk_img

            # fps
            now = time.perf_counter()
            self._fps_samples.append(1.0 / (now - self._last_fps_time))
            self._last_fps_time = now
            if len(self._fps_samples) > 30:
                self._fps_samples.pop(0)
            fps = np.mean(self._fps_samples) if self._fps_samples else 0
            self._fps_lbl.configure(text=f"FPS: {fps:.1f}")

        except queue.Empty:
            pass

        # -- new violation thumbnail ----------------------------------
        try:
            path = self.violation_queue.get_nowait()
            self._add_gallery_thumbnail(path)
            self._violcnt_lbl.configure(
                text=f"Violations: {self.logger.count}")
        except queue.Empty:
            pass

        self.after(UI_REFRESH_MS, self._update_display)

    # -----------------------------------------------------------------
    #  Gallery helper
    # -----------------------------------------------------------------
    def _add_gallery_thumbnail(self, path):
        try:
            pil_img = Image.open(path)
            pil_img.thumbnail((160, 120), Image.LANCZOS)
            ctk_img = ctk.CTkImage(pil_img, size=(160, 120))
            lbl = ctk.CTkLabel(self._gallery, image=ctk_img,
                                text="", width=160, height=120)
            lbl.pack(side="left", padx=4, pady=4)
            self._gallery_imgs.append(ctk_img)
        except Exception:
            pass

    # -----------------------------------------------------------------
    #  State label update
    # -----------------------------------------------------------------
    def _update_state_display(self):
        state_text = f"State: {self.app_state.value}"
        if self.app_state == AppState.TRACKING:
            state_text += "  |  Objects: "
            state_text += str(len(self.tracker.tracks))
        try:
            self._state_lbl.configure(text=state_text)
        except Exception:
            pass

    def _enable_start_btn(self):
        self._start_btn.configure(state="normal")
        self._stop_btn.configure(state="disabled")
        self._update_state_display()

    # -----------------------------------------------------------------
    #  Cleanup
    # -----------------------------------------------------------------
    def _on_closing(self):
        self.stop_event.set()
        self.running = False
        if self.proc_thread and self.proc_thread.is_alive():
            self.proc_thread.join(timeout=1.0)
        if self.cap:
            self.cap.release()
        self.destroy()
