"""
translation_state.py
====================
Keeps the translated text and the history. All the text-editing buttons
(Clear, Delete, Space, Save) work on this object. A lock makes it safe to use from
both the camera thread and the Flask request threads.
"""

import threading
from datetime import datetime

MAX_HISTORY = 20


class TranslationState:
    def __init__(self):
        self._lock = threading.Lock()
        self.text = ""
        self.history = []          # newest first: {"text": str, "time": "HH:MM:SS"}

    # --- adding recognised gestures -------------------------------------------
    def append_label(self, label):
        """A single letter is added directly; a word gets spaces around it."""
        with self._lock:
            if len(label) == 1:
                self.text += label
            else:
                if self.text and not self.text.endswith(" "):
                    self.text += " "
                self.text += label + " "

    # --- text controls --------------------------------------------------------
    def space(self):
        with self._lock:
            if self.text and not self.text.endswith(" "):
                self.text += " "

    def delete_last(self):
        with self._lock:
            self.text = self.text[:-1]

    def save(self):
        """Store the current text in the history (returns True if something was saved)."""
        with self._lock:
            return self._save_locked()

    def clear(self):
        """Clear the text. Non-empty text is saved to the history first."""
        with self._lock:
            saved = self._save_locked()
            self.text = ""
            return saved

    def _save_locked(self):
        entry = self.text.strip()
        if not entry:
            return False
        if self.history and self.history[0]["text"] == entry:
            return False
        self.history.insert(0, {"text": entry, "time": datetime.now().strftime("%H:%M:%S")})
        del self.history[MAX_HISTORY:]
        return True

    def clear_history(self):
        with self._lock:
            self.history = []

    def snapshot(self):
        with self._lock:
            return {"text": self.text, "history": list(self.history)}
