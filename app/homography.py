import cv2
import numpy as np


class HomographyTransformer:
    def __init__(self):
        self.src_rel = np.array([
            [0.20, 0.50],
            [0.80, 0.50],
            [0.95, 0.95],
            [0.05, 0.95],
        ], dtype=np.float32)

        self.dst = np.array([
            [0,   0],
            [500, 0],
            [500, 500],
            [0,   500],
        ], dtype=np.float32)

        self.M = None
        self.M_inv = None
        self.pixel_to_meter = 0.05
        self.initialized = False
        self.frame_w = None
        self.frame_h = None

    def set_frame_size(self, w, h):
        if w == self.frame_w and h == self.frame_h and self.initialized:
            return
        self.frame_w = w
        self.frame_h = h
        src_abs = self.src_rel * np.array([[w, h]], dtype=np.float32)
        self.M = cv2.getPerspectiveTransform(src_abs, self.dst)
        self.M_inv = cv2.getPerspectiveTransform(self.dst, src_abs)
        self.initialized = True

    def set_pixel_to_meter(self, scale):
        self.pixel_to_meter = scale

    def transform_point(self, pt):
        if not self.initialized:
            return pt
        inp = np.array([[list(pt)]], dtype=np.float32)
        out = cv2.perspectiveTransform(inp, self.M)
        return (float(out[0][0][0]), float(out[0][0][1]))

    def distance(self, p1, p2):
        if not self.initialized:
            dx = p2[0] - p1[0]
            dy = p2[1] - p1[1]
            return float(np.sqrt(dx**2 + dy**2)) * self.pixel_to_meter
        p1b = self.transform_point(p1)
        p2b = self.transform_point(p2)
        dx = p2b[0] - p1b[0]
        dy = p2b[1] - p1b[1]
        return float(np.sqrt(dx**2 + dy**2)) * self.pixel_to_meter

    def get_absolute_src(self):
        if self.frame_w is None or self.frame_h is None:
            return None
        return (self.src_rel * np.array([[self.frame_w, self.frame_h]])).astype(int)
