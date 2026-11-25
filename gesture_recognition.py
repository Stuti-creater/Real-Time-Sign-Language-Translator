"""
gesture_recognition.py
======================
The "brain" of the project. It has three parts:

1. HandDetector      - uses MediaPipe to find 21 hand landmarks in a camera frame.
2. classify_landmarks - a RULE-BASED classifier: it looks at the 21 landmarks,
                        works out which fingers are straight / curled, and
                        matches that pattern to a letter or word.
3. GestureStabilizer - makes sure a gesture is held steady for about a second
                        before it is added to the text (prevents accidental typing).

No random numbers and no fake predictions are used anywhere. If the hand does not
match any rule, the result is None and the website shows "Gesture not recognized".

MediaPipe hand landmark numbering (for reference):
    0 = wrist
    1-4   = thumb   (4 is the thumb tip)
    5-8   = index   (5 = knuckle/MCP, 6 = middle joint/PIP, 8 = tip)
    9-12  = middle
    13-16 = ring
    17-20 = pinky
"""

import math
import os
import urllib.request

import cv2
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "hand_landmarker.task")
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/hand_landmarker.task"
)

# Lines drawn between landmarks to show the "skeleton" of the hand.
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (0, 17), (17, 18), (18, 19), (19, 20),
]

# --------------------------------------------------------------------------
# Supported gestures (shown on the website AND used by the classifier)
# --------------------------------------------------------------------------
SUPPORTED_GESTURES = [
    {"label": "A", "type": "letter",
     "description": "Closed fist, thumb resting against the side and pointing up."},
    {"label": "B", "type": "letter",
     "description": "Flat hand, four fingers straight up and together, thumb folded across the palm."},
    {"label": "C", "type": "letter",
     "description": "Fingers and thumb curved to make a 'C' shape."},
    {"label": "D", "type": "letter",
     "description": "Index finger straight up; thumb touches the tip of the curled middle finger."},
    {"label": "E", "type": "letter",
     "description": "All fingers curled down, thumb tucked in front touching them."},
    {"label": "F", "type": "letter",
     "description": "Thumb and index finger touch to make a circle; other three fingers up."},
    {"label": "I", "type": "letter",
     "description": "Fist with only the little finger pointing up."},
    {"label": "L", "type": "letter",
     "description": "Index finger up and thumb out to the side, forming an 'L'."},
    {"label": "V", "type": "letter",
     "description": "Index and middle fingers up and spread apart (peace sign)."},
    {"label": "W", "type": "letter",
     "description": "Index, middle and ring fingers up and spread; little finger folded."},
    {"label": "Y", "type": "letter",
     "description": "Thumb and little finger out; other fingers curled ('call me' sign)."},
    {"label": "HELLO", "type": "word",
     "description": "Open hand with all five fingers spread wide apart."},
    {"label": "I LOVE YOU", "type": "word",
     "description": "Thumb, index finger and little finger out; middle and ring fingers folded."},
]

# --------------------------------------------------------------------------
# Thresholds. If recognition feels too strict / too loose for your camera or
# hand, these numbers are the ones to tune. Distances are measured in "palm
# lengths" (wrist to base of the middle finger), so they do not depend on how
# far the hand is from the camera.
# --------------------------------------------------------------------------
EXTENDED_ANGLE = 150   # finger joint angle (degrees) at or above this = straight
CURLED_ANGLE = 100     # at or below this = fully curled; in between = "bent"
THUMB_OUT = 0.75       # thumb tip this far from index knuckle = thumb sticks out
PINCH_TOUCH = 0.35     # thumb tip and index tip closer than this = touching (F)
THUMB_TOUCH = 0.60     # thumb tip near middle fingertip (D)
THUMB_E_TOUCH = 0.50   # thumb tip near curled fingertips (E)
SPREAD_HELLO = 1.05    # index-tip to pinky-tip distance for a spread open hand
V_GAP = 0.50           # index-tip to middle-tip distance for V
W_GAP = 0.80           # index-tip to ring-tip distance for W
C_GAP_MIN, C_GAP_MAX = 0.50, 1.50   # thumb-to-index gap for a "C" shape
A_THUMB_UP = 0.10      # thumb tip must stick up past the index knuckle (A)


# --------------------------------------------------------------------------
# Small maths helpers
# --------------------------------------------------------------------------
def _dist(p, i, j):
    return float(np.linalg.norm(p[i] - p[j]))


def _angle(p, a, b, c):
    """Angle in degrees at point b, formed by points a-b-c."""
    v1, v2 = p[a] - p[b], p[c] - p[b]
    cos = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-9)
    return math.degrees(math.acos(float(np.clip(cos, -1.0, 1.0))))


def _finger_state(angle):
    """Convert a joint angle into 'ext' (straight), 'bent' or 'curl'."""
    if angle >= EXTENDED_ANGLE:
        return "ext"
    if angle <= CURLED_ANGLE:
        return "curl"
    return "bent"


def _clarity(angle, state):
    """0.6 - 1.0: how clearly a finger sits inside its state (far from the thresholds)."""
    if state == "ext":
        margin = (angle - EXTENDED_ANGLE) / 25.0
    elif state == "curl":
        margin = (CURLED_ANGLE - angle) / 25.0
    else:
        margin = min(angle - CURLED_ANGLE, EXTENDED_ANGLE - angle) / 20.0
    return 0.6 + 0.4 * min(1.0, max(0.0, margin))


# --------------------------------------------------------------------------
# Feature extraction + rule-based classification
# --------------------------------------------------------------------------
FINGER_JOINTS = {            # (knuckle, middle joint, tip)
    "index": (5, 6, 8),
    "middle": (9, 10, 12),
    "ring": (13, 14, 16),
    "pinky": (17, 18, 20),
}


def extract_features(p):
    """Turn 21 landmark points into a small set of easy-to-read measurements."""
    palm = _dist(p, 0, 9) + 1e-9                      # wrist -> middle knuckle
    angles = [_angle(p, a, b, c) for a, b, c in FINGER_JOINTS.values()]
    states = tuple(_finger_state(a) for a in angles)

    axis = (p[9] - p[0]) / palm                       # direction the hand points
    thumb_vec = p[4] - p[2]
    index_vec = p[8] - p[5]
    cos_thumb_index = float(
        np.dot(thumb_vec, index_vec)
        / (np.linalg.norm(thumb_vec) * np.linalg.norm(index_vec) + 1e-9)
    )
    return {
        "palm": palm,
        "angles": angles,
        "states": states,                              # index, middle, ring, pinky
        "thumb_out": _dist(p, 4, 5) / palm,            # thumb tip <-> index knuckle
        "pinch": _dist(p, 4, 8) / palm,                # thumb tip <-> index tip
        "thumb_mid": _dist(p, 4, 12) / palm,           # thumb tip <-> middle tip
        "thumb_up": float(np.dot(p[4] - p[5], axis)) / palm,
        "cos_thumb_index": cos_thumb_index,
        "spread": _dist(p, 8, 20) / palm,              # index tip <-> pinky tip
        "v_gap": _dist(p, 8, 12) / palm,               # index tip <-> middle tip
        "w_gap": _dist(p, 8, 16) / palm,               # index tip <-> ring tip
    }


def classify_landmarks(points):
    """
    Classify one hand.

    points: 21 landmarks, each (x, y, z) in the same units (pixels).
    Returns (label, confidence, features):
        label       - e.g. "A", "HELLO", or None if no rule matches
        confidence  - 0..1 "rule confidence": how clearly the fingers sit inside
                      the rule's expected states. It is NOT a trained probability.
    """
    p = np.asarray(points, dtype=float)
    f = extract_features(p)
    idx, mid, ring, pinky = f["states"]
    up = lambda s: s == "ext"
    down = lambda s: s != "ext"
    thumb_out = f["thumb_out"] >= THUMB_OUT
    label = None

    if up(mid) and up(ring) and up(pinky) and down(idx) and f["pinch"] < PINCH_TOUCH:
        label = "F"
    elif up(idx) and up(mid) and up(ring) and up(pinky):
        if f["spread"] >= SPREAD_HELLO:
            label = "HELLO"
        elif not thumb_out:
            label = "B"
    elif up(idx) and down(mid) and down(ring) and down(pinky):
        if thumb_out:
            if -0.3 < f["cos_thumb_index"] < 0.6:
                label = "L"
        elif f["thumb_mid"] < THUMB_TOUCH:
            label = "D"
    elif up(idx) and up(mid) and down(ring) and down(pinky):
        if f["v_gap"] >= V_GAP:
            label = "V"
    elif up(idx) and up(mid) and up(ring) and down(pinky):
        if f["w_gap"] >= W_GAP:
            label = "W"
    elif up(pinky) and down(idx) and down(mid) and down(ring):
        label = "Y" if thumb_out else "I"
    elif up(idx) and up(pinky) and down(mid) and down(ring) and thumb_out:
        label = "I LOVE YOU"
    elif down(idx) and down(mid) and down(ring) and down(pinky):
        bent_count = f["states"].count("bent")
        if bent_count >= 3 and C_GAP_MIN <= f["pinch"] <= C_GAP_MAX:
            label = "C"
        elif f["thumb_mid"] < THUMB_E_TOUCH:
            label = "E"
        elif all(s == "curl" for s in f["states"]) and f["thumb_up"] >= A_THUMB_UP:
            label = "A"

    if label is None:
        return None, 0.0, f

    clarity = [_clarity(a, s) for a, s in zip(f["angles"], f["states"])]
    confidence = min(0.99, sum(clarity) / len(clarity))
    return label, confidence, f


# --------------------------------------------------------------------------
# Stabiliser: "hold a gesture for ~1 second to type it"
# --------------------------------------------------------------------------
class GestureStabilizer:
    """
    Feed it the label of every frame. It tells you
      - progress (0..1) towards confirming the current gesture, and
      - a committed label when the gesture has been held long enough.

    After a gesture is committed it will not repeat until the hand changes to
    something else (or leaves the camera) for a short moment. This is how the same
    letter can be typed twice, e.g. L-L in "HELLO": show L, change, show L again.
    """

    def __init__(self, hold_seconds=1.0, release_seconds=0.5):
        self.hold_seconds = hold_seconds
        self.release_seconds = release_seconds
        self.reset()

    def reset(self):
        self.candidate = None
        self.candidate_since = 0.0
        self.locked = None          # last committed label, waiting for release
        self.other_since = None     # when the hand started to differ from the lock

    def update(self, label, now):
        """Returns (progress, committed_label_or_None)."""
        # Release logic: unlock once something different has been shown for a moment.
        if self.locked is not None:
            if label == self.locked:
                self.other_since = None
            else:
                if self.other_since is None:
                    self.other_since = now
                if now - self.other_since >= self.release_seconds:
                    self.locked = None
                    self.other_since = None

        if label is None:
            self.candidate = None
            return 0.0, None

        if label != self.candidate:
            self.candidate = label
            self.candidate_since = now

        if label == self.locked:
            return 0.0, None

        progress = min(1.0, (now - self.candidate_since) / self.hold_seconds)
        if progress >= 1.0:
            self.locked = label
            self.other_since = None
            return 1.0, label
        return progress, None


# --------------------------------------------------------------------------
# MediaPipe hand detector (works with old AND new MediaPipe versions)
# --------------------------------------------------------------------------
class DetectorError(RuntimeError):
    """Raised when MediaPipe cannot be started (with a beginner-friendly message)."""


class HandDetector:
    """
    Finds one hand in a BGR frame and returns its landmarks.

    - MediaPipe <= 0.10.14 (and some later ones) has the classic `mp.solutions.hands`.
    - Newer MediaPipe versions removed it and use the "Tasks" API, which needs a small
      model file (models/hand_landmarker.task). It is downloaded automatically the
      first time if it is missing and you are online.
    """

    def __init__(self):
        try:
            import mediapipe as mp
        except ImportError as exc:
            raise DetectorError(
                "MediaPipe is not installed. Run: pip install -r requirements.txt"
            ) from exc

        self._mp = mp
        self._timestamp_ms = 0
        if hasattr(mp, "solutions") and hasattr(mp.solutions, "hands"):
            self.backend = "solutions"
            self._hands = mp.solutions.hands.Hands(
                static_image_mode=False,
                max_num_hands=1,
                min_detection_confidence=0.6,
                min_tracking_confidence=0.5,
            )
        else:
            self.backend = "tasks"
            self._landmarker = self._create_task_landmarker()

    def _create_task_landmarker(self):
        self._ensure_model()
        try:
            from mediapipe.tasks import python as mp_python
            from mediapipe.tasks.python import vision

            options = vision.HandLandmarkerOptions(
                base_options=mp_python.BaseOptions(model_asset_path=MODEL_PATH),
                running_mode=vision.RunningMode.VIDEO,
                num_hands=1,
                min_hand_detection_confidence=0.6,
                min_hand_presence_confidence=0.5,
                min_tracking_confidence=0.5,
            )
            return vision.HandLandmarker.create_from_options(options)
        except Exception as exc:
            raise DetectorError(f"Could not start MediaPipe HandLandmarker: {exc}") from exc

    @staticmethod
    def _ensure_model():
        if os.path.exists(MODEL_PATH) and os.path.getsize(MODEL_PATH) > 1000:
            return
        os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
        try:
            print("Downloading MediaPipe hand model (one time, ~8 MB)...")
            urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        except Exception as exc:
            if os.path.exists(MODEL_PATH):
                os.remove(MODEL_PATH)
            raise DetectorError(
                "The hand model file is missing and could not be downloaded "
                f"({exc}). Download it manually from:\n{MODEL_URL}\n"
                f"and save it as: {MODEL_PATH}\n"
                "Or install an older MediaPipe that needs no model file: "
                "pip install mediapipe==0.10.14 (Python 3.8 - 3.12)."
            ) from exc

    def process(self, frame_bgr):
        """
        Returns None if no hand is visible, otherwise a tuple
        (points_3d, points_px):
            points_3d - (21, 3) array in pixel units, used by the classifier
            points_px - (21, 2) int array, used for drawing
        """
        h, w = frame_bgr.shape[:2]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        if self.backend == "solutions":
            result = self._hands.process(rgb)
            if not result.multi_hand_landmarks:
                return None
            landmarks = result.multi_hand_landmarks[0].landmark
        else:
            self._timestamp_ms += 33          # must always increase
            mp_image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb)
            result = self._landmarker.detect_for_video(mp_image, self._timestamp_ms)
            if not result.hand_landmarks:
                return None
            landmarks = result.hand_landmarks[0]

        pts3d = np.array([[lm.x * w, lm.y * h, lm.z * w] for lm in landmarks], dtype=float)
        ptspx = pts3d[:, :2].astype(int)
        return pts3d, ptspx

    def close(self):
        try:
            if self.backend == "solutions":
                self._hands.close()
            else:
                self._landmarker.close()
        except Exception:
            pass


def draw_hand(frame, points_px):
    """Draw the hand skeleton on the frame (in place)."""
    for a, b in HAND_CONNECTIONS:
        cv2.line(frame, tuple(points_px[a]), tuple(points_px[b]), (255, 140, 90), 2, cv2.LINE_AA)
    for x, y in points_px:
        cv2.circle(frame, (int(x), int(y)), 4, (255, 255, 255), -1, cv2.LINE_AA)
        cv2.circle(frame, (int(x), int(y)), 4, (200, 60, 150), 1, cv2.LINE_AA)
