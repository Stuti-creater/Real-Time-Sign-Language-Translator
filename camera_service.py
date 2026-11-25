"""
camera_service.py
=================
Opens the webcam with OpenCV and runs a background loop that, for every frame:

    1. mirrors the frame (so it behaves like a mirror),
    2. finds the hand with MediaPipe,
    3. classifies the gesture with the rule-based classifier,
    4. waits for the gesture to be held steady,
    5. draws the hand skeleton + label on the frame,
    6. stores the frame as JPEG so Flask can stream it to the website.
"""

import platform
import threading
import time

import cv2

from gesture_recognition import (
    DetectorError,
    GestureStabilizer,
    HandDetector,
    classify_landmarks,
    draw_hand,
)


class CameraError(RuntimeError):
    """Raised when the webcam cannot be opened."""


def _open_camera(preferred_index=0):
    """Try the preferred camera first, then the next few indexes."""
    indexes = [preferred_index] + [i for i in range(3) if i != preferred_index]
    for index in indexes:
        if platform.system() == "Windows":
            cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
        else:
            cap = cv2.VideoCapture(index)
        if cap.isOpened():
            ok, _ = cap.read()
            if ok:
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 960)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 540)
                return cap
        cap.release()
    return None


class CameraService:
    def __init__(self, on_commit=None):
        """on_commit(label) is called when a gesture has been held long enough."""
        self.on_commit = on_commit
        self._cap = None
        self._thread = None
        self._detector = None
        self._running = False
        self._lock = threading.Lock()
        self._jpeg = None
        self.error = None
        self._result = self._empty_result()

    @staticmethod
    def _empty_result():
        return {"hand_detected": False, "label": None, "confidence": 0.0, "progress": 0.0}

    # --- public API -------------------------------------------------------------
    @property
    def running(self):
        return self._running

    def start(self, camera_index=0):
        if self._running:
            return
        self.error = None
        try:
            if self._detector is None:
                self._detector = HandDetector()
        except DetectorError as exc:
            self.error = str(exc)
            raise

        cap = _open_camera(camera_index)
        if cap is None:
            self.error = (
                "No webcam could be opened. Check that a camera is connected, "
                "that it is not being used by another app (Zoom, Teams, ...), "
                "and that camera access is allowed in your system privacy settings."
            )
            raise CameraError(self.error)

        self._cap = cap
        self._running = True
        self._result = self._empty_result()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False
        if self._thread and self._thread is not threading.current_thread():
            self._thread.join(timeout=3)
        self._thread = None
        with self._lock:
            self._jpeg = None
            self._result = self._empty_result()

    def get_jpeg(self):
        with self._lock:
            return self._jpeg

    def get_result(self):
        with self._lock:
            return dict(self._result)

    # --- background loop --------------------------------------------------------
    def _loop(self):
        stabilizer = GestureStabilizer(hold_seconds=1.0)
        failures = 0
        try:
            while self._running:
                ok, frame = self._cap.read()
                if not ok:
                    failures += 1
                    if failures > 30:
                        self.error = "The camera stopped sending images (was it unplugged?)."
                        break
                    time.sleep(0.03)
                    continue
                failures = 0

                frame = cv2.flip(frame, 1)
                try:
                    hand = self._detector.process(frame)
                except Exception as exc:
                    self.error = f"Hand detection failed: {exc}"
                    break

                result = self._empty_result()
                now = time.time()
                if hand is None:
                    stabilizer.update(None, now)
                else:
                    points3d, points_px = hand
                    label, confidence, _ = classify_landmarks(points3d)
                    progress, committed = stabilizer.update(label, now)
                    draw_hand(frame, points_px)
                    result.update(
                        hand_detected=True, label=label,
                        confidence=confidence, progress=progress,
                    )
                    if committed and self.on_commit:
                        self.on_commit(committed)
                    self._annotate(frame, label, confidence, progress)

                ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                if ok:
                    with self._lock:
                        self._jpeg = buf.tobytes()
                        self._result = result
        finally:
            self._running = False
            if self._cap is not None:
                self._cap.release()
                self._cap = None

    @staticmethod
    def _annotate(frame, label, confidence, progress):
        """Write the gesture name on the video and a progress bar along the bottom."""
        text = f"{label}  {confidence * 100:.0f}%" if label else "Gesture not recognized"
        colour = (120, 230, 120) if label else (90, 90, 255)
        cv2.rectangle(frame, (0, 0), (frame.shape[1], 44), (40, 20, 30), -1)
        cv2.putText(frame, text, (14, 31), cv2.FONT_HERSHEY_SIMPLEX, 0.9, colour, 2, cv2.LINE_AA)
        if progress > 0:
            h, w = frame.shape[:2]
            cv2.rectangle(frame, (0, h - 10), (int(w * progress), h), (255, 120, 80), -1)
