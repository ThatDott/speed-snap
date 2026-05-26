import cv2
import numpy as np
from ultralytics import YOLO


TARGET_CLASSES = [0, 2, 3, 5]
TARGET_NAMES = {0: 'person', 2: 'car', 3: 'motorcycle', 5: 'bus'}


class ObjectDetector:
    def __init__(self, model_name='yolov8n.pt', confidence=0.5):
        self.model = YOLO(model_name)
        self.confidence = confidence
        self.target_classes = TARGET_CLASSES

    def detect(self, frame):
        results = self.model(frame, verbose=False, device='cpu')[0]
        detections = []

        for box in results.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])

            if cls_id not in self.target_classes or conf < self.confidence:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            centroid = ((x1 + x2) // 2, (y1 + y2) // 2)

            detections.append({
                'bbox': (x1, y1, x2, y2),
                'centroid': centroid,
                'class_id': cls_id,
                'class_name': results.names[cls_id],
                'confidence': round(conf, 3),
            })

        return detections
