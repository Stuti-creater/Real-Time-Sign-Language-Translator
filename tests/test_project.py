"""
Automated checks. Run from the project folder with:

    python -m unittest discover -s tests -v

What is tested:
  * every supported gesture is recognised from a synthetic hand (and at any rotation)
  * unsupported hand shapes give None  ("Gesture not recognized"), never a guess
  * the hold-to-type stabiliser behaves correctly
  * the text controls (space / delete / clear / history)
  * the Flask API, including the "no webcam" error path
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gesture_recognition import SUPPORTED_GESTURES, GestureStabilizer, classify_landmarks  # noqa: E402
from synthetic_hands import CURLED, STRAIGHT, hand_for, make_hand, rotate  # noqa: E402
from translation_state import TranslationState  # noqa: E402


class ClassifierTests(unittest.TestCase):
    def test_every_supported_gesture_is_recognised(self):
        for g in SUPPORTED_GESTURES:
            label, conf, _ = classify_landmarks(hand_for(g["label"]))
            self.assertEqual(label, g["label"], f"{g['label']} was classified as {label}")
            self.assertGreater(conf, 0.6)
            self.assertLessEqual(conf, 0.99)

    def test_rotation_does_not_matter(self):
        for angle in (-60, -25, 30, 90, 180):
            for g in SUPPORTED_GESTURES:
                label, _, _ = classify_landmarks(rotate(hand_for(g["label"]), angle))
                self.assertEqual(label, g["label"], f"{g['label']} @ {angle} deg -> {label}")

    def test_scale_does_not_matter(self):
        for g in SUPPORTED_GESTURES:
            label, _, _ = classify_landmarks(hand_for(g["label"]) * 250.0)
            self.assertEqual(label, g["label"])

    def test_unsupported_shapes_return_none(self):
        s = {k: STRAIGHT for k in ("index", "middle", "ring", "pinky")}
        c = {k: CURLED for k in s}
        unsupported = [
            make_hand({**c, "index": STRAIGHT, "middle": STRAIGHT}, "folded"),   # U (fingers together)
            make_hand({**c, "index": STRAIGHT, "pinky": STRAIGHT}, "folded"),     # index + pinky, thumb in
            make_hand({**c, "middle": STRAIGHT}, "folded"),                       # middle finger only
            make_hand({**s, "pinky": CURLED}, "folded"),                          # 3 fingers together
            make_hand(s, "out"),                                                  # flat hand with thumb out
        ]
        for hand in unsupported:
            label, conf, _ = classify_landmarks(hand)
            self.assertIsNone(label)
            self.assertEqual(conf, 0.0)


class StabilizerTests(unittest.TestCase):
    def test_commits_after_hold_and_not_again_until_release(self):
        st = GestureStabilizer(hold_seconds=1.0, release_seconds=0.5)
        committed = []
        t = 0.0
        for _ in range(40):                       # 4 s of holding "A", 10 fps
            _, c = st.update("A", t)
            if c:
                committed.append((round(t, 1), c))
            t += 0.1
        self.assertEqual([c for _, c in committed], ["A"])      # typed exactly once
        self.assertTrue(1.0 <= committed[0][0] <= 1.2)          # after ~1 second

    def test_same_letter_twice_after_changing_gesture(self):
        st = GestureStabilizer(hold_seconds=1.0, release_seconds=0.5)
        committed, t = [], 0.0
        for label, seconds in (("L", 1.2), ("A", 0.8), ("L", 1.2)):
            for _ in range(int(seconds * 10)):
                _, c = st.update(label, t)
                if c:
                    committed.append(c)
                t += 0.1
        self.assertEqual(committed, ["L", "L"])

    def test_brief_flicker_does_not_repeat(self):
        st = GestureStabilizer(hold_seconds=1.0, release_seconds=0.5)
        committed, t = [], 0.0
        seq = ["A"] * 12 + [None] * 2 + ["A"] * 20       # 0.2 s flicker only
        for label in seq:
            _, c = st.update(label, t)
            if c:
                committed.append(c)
            t += 0.1
        self.assertEqual(committed, ["A"])

    def test_changing_gesture_resets_progress(self):
        st = GestureStabilizer(hold_seconds=1.0)
        for i in range(8):
            st.update("A", i * 0.1)
        progress, c = st.update("B", 0.9)
        self.assertEqual((progress, c), (0.0, None))


class TextStateTests(unittest.TestCase):
    def test_word_building_and_controls(self):
        ts = TranslationState()
        for ch in "HI":
            ts.append_label(ch)
        ts.space()
        ts.space()                                  # double space is ignored
        ts.append_label("HELLO")
        self.assertEqual(ts.snapshot()["text"], "HI HELLO ")
        ts.delete_last()
        self.assertEqual(ts.snapshot()["text"], "HI HELLO")
        ts.clear()
        self.assertEqual(ts.snapshot()["text"], "")
        self.assertEqual(ts.snapshot()["history"][0]["text"], "HI HELLO")

    def test_history_limits_and_clear(self):
        ts = TranslationState()
        for i in range(30):
            ts.append_label("A")
            ts.append_label(chr(66 + i % 20))
            ts.clear()
        self.assertLessEqual(len(ts.snapshot()["history"]), 20)
        ts.clear_history()
        self.assertEqual(ts.snapshot()["history"], [])

    def test_empty_clear_saves_nothing(self):
        ts = TranslationState()
        self.assertFalse(ts.clear())
        self.assertEqual(ts.snapshot()["history"], [])


class FlaskApiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import app as app_module
        cls.m = app_module
        cls.client = app_module.app.test_client()

    def setUp(self):
        self.m.state.clear()
        self.m.state.clear_history()

    def test_index_page_and_static_files_are_connected(self):
        page = self.client.get("/")
        self.assertEqual(page.status_code, 200)
        html = page.get_data(as_text=True)
        self.assertIn("Real-Time Sign Language Translator", html)
        for url in ("/static/style.css", "/static/script.js"):
            self.assertIn(url, html)
            self.assertEqual(self.client.get(url).status_code, 200, url)

    def test_status_when_camera_off(self):
        data = self.client.get("/api/status").get_json()
        self.assertFalse(data["camera_running"])
        self.assertEqual(data["gesture"], "Camera is off")

    def test_gestures_endpoint_matches_classifier(self):
        data = self.client.get("/api/gestures").get_json()
        self.assertEqual([g["label"] for g in data], [g["label"] for g in SUPPORTED_GESTURES])

    def test_start_without_camera_returns_clean_error(self):
        class NoCam:
            pass
        import camera_service
        original = camera_service._open_camera
        camera_service._open_camera = lambda index=0: None
        try:
            resp = self.client.post("/api/camera/start")
        finally:
            camera_service._open_camera = original
        # 503 = camera problem, 503 also if MediaPipe itself cannot start; both are clean JSON errors
        self.assertEqual(resp.status_code, 503)
        body = resp.get_json()
        self.assertFalse(body["ok"])
        self.assertTrue(body["error"])

    def test_video_feed_refused_when_camera_off(self):
        self.assertEqual(self.client.get("/video_feed").status_code, 409)

    def test_text_controls(self):
        self.m.state.append_label("H")
        self.m.state.append_label("I")
        r = self.client.post("/api/text/space").get_json()
        self.assertEqual(r["text"], "HI ")
        r = self.client.post("/api/text/delete").get_json()
        self.assertEqual(r["text"], "HI")
        r = self.client.post("/api/text/save").get_json()
        self.assertEqual(r["history"][0]["text"], "HI")
        self.assertEqual(self.client.post("/api/text/save").status_code, 400)   # duplicate
        r = self.client.post("/api/text/clear").get_json()
        self.assertEqual(r["text"], "")
        self.assertEqual(self.client.post("/api/text/nonsense").status_code, 404)
        self.client.post("/api/history/clear")
        self.assertEqual(self.client.get("/api/status").get_json()["history"], [])

    def test_speak_with_empty_text_reports_failure_cleanly(self):
        r = self.client.post("/api/speak").get_json()
        self.assertFalse(r["ok"])
        self.assertIn("no text", r["message"].lower())


class CameraPipelineTests(unittest.TestCase):
    """Runs the real camera loop with a fake webcam + fake detector (no hardware needed)."""

    def test_pipeline_types_letters_and_streams_frames(self):
        import time
        import numpy as np
        import camera_service
        from camera_service import CameraService

        class FakeCap:
            def isOpened(self): return True
            def read(self):
                time.sleep(0.01)
                return True, np.zeros((360, 640, 3), np.uint8)
            def set(self, *a): return True
            def release(self): pass

        class FakeDetector:
            """Shows letter A, then B, then no hand."""
            def __init__(self): self.t0 = time.time()
            def process(self, frame):
                dt = time.time() - self.t0
                if dt < 1.6:
                    label = "A"
                elif dt < 3.0:
                    label = "B"
                else:
                    return None
                pts = hand_for(label) * 150.0 + np.array([200.0, 100.0, 0.0])
                return pts, pts[:, :2].astype(int)

        typed = []
        original = camera_service._open_camera
        camera_service._open_camera = lambda index=0: FakeCap()
        cam = CameraService(on_commit=typed.append)
        cam._detector = FakeDetector()
        try:
            cam.start()
            time.sleep(0.5)
            jpeg = cam.get_jpeg()
            self.assertTrue(jpeg and jpeg[:2] == b"\xff\xd8")        # a real JPEG frame
            self.assertTrue(cam.get_result()["hand_detected"])
            time.sleep(3.2)
        finally:
            cam.stop()
            camera_service._open_camera = original
        self.assertEqual(typed, ["A", "B"])
        self.assertFalse(cam.running)
        self.assertIsNone(cam.get_jpeg())


if __name__ == "__main__":
    unittest.main()
