import time
import numpy as np
from collections import OrderedDict


class TrackedObject:
    def __init__(self, obj_id, class_name, centroid, bbox):
        self.id = obj_id
        self.class_name = class_name
        self.positions = [(centroid, time.time())]
        self.speed_history = []
        self.current_speed_kmh = 0.0
        self.smoothed_speed_kmh = 0.0
        self.last_bbox = bbox
        self.disappeared = 0
        self.violation_logged = False

    @property
    def centroid(self):
        return self.positions[-1][0]


class SpeedTracker:
    MAX_DISAPPEARED = 30
    MATCH_DIST_THRESHOLD = 120

    def __init__(self, homography):
        self.homography = homography
        self.tracks = OrderedDict()
        self.next_id = 0

    def reset(self):
        self.tracks.clear()
        self.next_id = 0

    def update(self, detections, pixel_to_meter):
        self.homography.set_pixel_to_meter(pixel_to_meter)

        for track in self.tracks.values():
            track.disappeared += 1

        if not detections:
            self._cleanup()
            return self.tracks

        used_det = set()

        for tid in list(self.tracks.keys()):
            track = self.tracks[tid]
            if track.disappeared == 0:
                continue

            best_idx = -1
            best_dist = float('inf')
            for i, det in enumerate(detections):
                if i in used_det:
                    continue
                d = np.linalg.norm(np.subtract(det['centroid'],
                                               track.centroid))
                if d < best_dist and d < self.MATCH_DIST_THRESHOLD:
                    best_dist = d
                    best_idx = i

            if best_idx != -1:
                det = detections[best_idx]
                used_det.add(best_idx)
                track.disappeared = 0
                track.last_bbox = det['bbox']
                track.positions.append((det['centroid'], time.time()))
                self._update_speed(track)

        for i, det in enumerate(detections):
            if i not in used_det:
                t = TrackedObject(self.next_id, det['class_name'],
                                  det['centroid'], det['bbox'])
                self.tracks[self.next_id] = t
                self.next_id += 1

        self._cleanup()
        return self.tracks

    def _update_speed(self, track):
        if len(track.positions) < 3:
            return

        now = time.time()
        recent = [(p, t) for p, t in track.positions
                  if t >= now - 0.4]
        if len(recent) < 2:
            return

        p1, t1 = recent[0]
        p2, t2 = recent[-1]
        dt = t2 - t1
        if dt <= 0:
            return

        dist_m = self.homography.distance(p1, p2)
        speed_ms = dist_m / dt
        speed_kmh = speed_ms * 3.6

        if speed_kmh < 0.5:
            speed_kmh = 0.0

        track.speed_history.append(speed_kmh)
        if len(track.speed_history) > 8:
            track.speed_history.pop(0)

        track.current_speed_kmh = speed_kmh
        track.smoothed_speed_kmh = np.mean(track.speed_history)

    def _cleanup(self):
        stale = [tid for tid, t in self.tracks.items()
                 if t.disappeared > self.MAX_DISAPPEARED]
        for tid in stale:
            del self.tracks[tid]
