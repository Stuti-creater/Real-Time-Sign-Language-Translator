"""
text_to_speech.py
=================
Speaks text aloud using pyttsx3, a free and OFFLINE text-to-speech library.

Speech runs in a background thread so the website never freezes while the
computer is talking. If pyttsx3 is not working (for example on a Linux machine
without espeak), speak() returns (False, reason) and the website automatically
falls back to the browser's built-in speech voice.
"""

import threading

_speaking = threading.Lock()


def speak(text):
    """
    Start speaking `text`. Returns (ok, message).
    Returns quickly: ok=True means speech has started, not that it has finished.
    """
    text = (text or "").strip()
    if not text:
        return False, "There is no text to speak."

    try:
        import pyttsx3  # noqa: F401  (checked here so we can give a clear message)
    except ImportError:
        return False, "pyttsx3 is not installed. Run: pip install pyttsx3"

    if not _speaking.acquire(blocking=False):
        return False, "Already speaking, please wait a moment."

    started = threading.Event()
    outcome = {"error": None}

    def worker():
        try:
            import pyttsx3

            engine = pyttsx3.init()
            engine.setProperty("rate", 150)
            engine.say(text.lower())
            started.set()
            engine.runAndWait()
        except Exception as exc:  # no audio device, missing espeak, etc.
            outcome["error"] = str(exc) or exc.__class__.__name__
        finally:
            started.set()
            _speaking.release()

    threading.Thread(target=worker, daemon=True).start()
    started.wait(timeout=8)

    if outcome["error"]:
        return False, f"Text-to-speech engine error: {outcome['error']}"
    return True, "Speaking..."
