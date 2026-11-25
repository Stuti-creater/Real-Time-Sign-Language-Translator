"""
app.py - Flask backend for the Real-Time Sign Language Translator.

Run with:   python app.py
Then open:  http://127.0.0.1:5000
"""

import os
import threading
import time
import webbrowser

from flask import Flask, Response, jsonify, render_template, request

from camera_service import CameraError, CameraService
from gesture_recognition import SUPPORTED_GESTURES, DetectorError
from text_to_speech import speak
from translation_state import TranslationState

app = Flask(__name__)

state = TranslationState()
# Whenever a gesture is held steady long enough, add it to the translated text.
camera = CameraService(on_commit=state.append_label)


# ---------------------------------------------------------------- pages
@app.route("/")
def index():
    return render_template("index.html")


# ---------------------------------------------------------------- camera
@app.route("/api/camera/start", methods=["POST"])
def camera_start():
    try:
        camera.start(int(os.environ.get("CAMERA_INDEX", 0)))
    except (CameraError, DetectorError) as exc:
        return jsonify(ok=False, error=str(exc)), 503
    except Exception as exc:  # anything unexpected: report it instead of crashing
        return jsonify(ok=False, error=f"Unexpected error: {exc}"), 500
    return jsonify(ok=True)


@app.route("/api/camera/stop", methods=["POST"])
def camera_stop():
    camera.stop()
    return jsonify(ok=True)


def _mjpeg_frames():
    """Generator that yields the newest camera frame again and again (MJPEG stream)."""
    last = None
    while camera.running:
        jpeg = camera.get_jpeg()
        if jpeg is not None and jpeg is not last:
            last = jpeg
            yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpeg + b"\r\n"
        time.sleep(0.02)


@app.route("/video_feed")
def video_feed():
    if not camera.running:
        return jsonify(ok=False, error="Camera is not running."), 409
    return Response(_mjpeg_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")


# ---------------------------------------------------------------- live status
@app.route("/api/status")
def status():
    result = camera.get_result()
    if not camera.running:
        gesture = "Camera is off"
    elif not result["hand_detected"]:
        gesture = "No hand detected"
    elif result["label"] is None:
        gesture = "Gesture not recognized"
    else:
        gesture = result["label"]

    data = state.snapshot()
    return jsonify(
        camera_running=camera.running,
        error=camera.error,
        gesture=gesture,
        recognized=result["label"] is not None,
        confidence=round(result["confidence"], 3),
        progress=round(result["progress"], 3),
        text=data["text"],
        history=data["history"],
    )


@app.route("/api/gestures")
def gestures():
    return jsonify(SUPPORTED_GESTURES)


# ---------------------------------------------------------------- text controls
@app.route("/api/text/<action>", methods=["POST"])
def text_action(action):
    if action == "clear":
        state.clear()
    elif action == "delete":
        state.delete_last()
    elif action == "space":
        state.space()
    elif action == "save":
        if not state.save():
            return jsonify(ok=False, error="Nothing new to save."), 400
    else:
        return jsonify(ok=False, error="Unknown action."), 404
    return jsonify(ok=True, **state.snapshot())


@app.route("/api/history/clear", methods=["POST"])
def history_clear():
    state.clear_history()
    return jsonify(ok=True)


# ---------------------------------------------------------------- speech
@app.route("/api/speak", methods=["POST"])
def speak_text():
    text = state.snapshot()["text"]
    ok, message = speak(text)
    if ok:
        state.save()  # spoken sentences go to the history
    # ok=False is not a server error: the browser then uses its own voice instead.
    return jsonify(ok=ok, message=message, text=text)


# ---------------------------------------------------------------- start-up
if __name__ == "__main__":
    host, port = "127.0.0.1", int(os.environ.get("PORT", 5000))
    url = f"http://{host}:{port}"
    print(f"\n  Real-Time Sign Language Translator running at {url}\n  Press CTRL+C to stop.\n")
    if os.environ.get("NO_BROWSER") != "1":
        threading.Timer(1.2, lambda: webbrowser.open(url)).start()
    try:
        app.run(host=host, port=port, debug=False, threaded=True)
    finally:
        camera.stop()
