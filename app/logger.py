import os
import csv
import cv2
import numpy as np
import threading
from datetime import datetime


class ViolationManager:
    def __init__(self, output_dir='speed_violations'):
        self.output_dir = os.path.abspath(output_dir)
        self.csv_path = os.path.join(self.output_dir, 'violations.csv')
        self.lock = threading.Lock()
        self.count = 0
        self._init_storage()

    def _init_storage(self):
        os.makedirs(self.output_dir, exist_ok=True)
        if not os.path.exists(self.csv_path):
            with open(self.csv_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'Timestamp', 'Object_Type', 'Detected_Speed_kmh',
                    'Speed_Limit_kmh', 'Screenshot_Path'
                ])

    def log(self, frame, bbox, object_type, speed, speed_limit):
        timestamp = datetime.now()
        ts_str = timestamp.strftime('%Y%m%d_%H%M%S_%f')

        x1 = max(0, bbox[0])
        y1 = max(0, bbox[1])
        x2 = min(frame.shape[1], bbox[2])
        y2 = min(frame.shape[0], bbox[3])

        if x2 - x1 < 10 or y2 - y1 < 10:
            x1 = max(0, x1 - 20)
            y1 = max(0, y1 - 20)
            x2 = min(frame.shape[1], x2 + 20)
            y2 = min(frame.shape[0], y2 + 20)

        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return None

        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        edge_bgr = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        overlay = cv2.addWeighted(crop, 0.55, edge_bgr, 0.45, 0)

        h, w = overlay.shape[:2]
        if h > 0 and w > 0:
            cv2.putText(overlay, f"SPEED: {speed:.1f} km/h", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)
            cv2.putText(overlay, f"LIMIT: {speed_limit} km/h", (10, h - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

        filename = f"violation_{ts_str}.png"
        filepath = os.path.join(self.output_dir, filename)
        cv2.imwrite(filepath, overlay)

        with self.lock:
            with open(self.csv_path, 'a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow([
                    timestamp.isoformat(),
                    object_type,
                    f"{speed:.2f}",
                    speed_limit,
                    filepath,
                ])
            self.count += 1

        return filepath
