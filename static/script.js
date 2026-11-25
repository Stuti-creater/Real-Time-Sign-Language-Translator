/* ===========================================================================
   Real-Time Sign Language Translator - browser code

   How it talks to the Flask backend:
     POST /api/camera/start | /api/camera/stop   -> turn the webcam on / off
     GET  /video_feed                            -> live MJPEG video (an <img>)
     GET  /api/status                            -> gesture, confidence, text, history
     POST /api/text/<clear|delete|space|save>    -> text controls
     POST /api/history/clear                     -> clear history
     POST /api/speak                             -> text-to-speech
     GET  /api/gestures                          -> list of supported gestures
   =========================================================================== */
(function () {
  "use strict";

  var $ = function (id) { return document.getElementById(id); };
  var el = {
    video: $("video"), wrap: $("videoWrap"), placeholder: $("placeholder"),
    btnStart: $("btnStart"), btnStop: $("btnStop"),
    errorBox: $("errorBox"), errorText: $("errorText"),
    pill: $("statusPill"), statusText: $("statusText"),
    gestureName: $("gestureName"), confText: $("confText"), confBar: $("confBar"),
    holdBar: $("holdBar"), holdText: $("holdText"),
    textBox: $("textBox"), historyList: $("historyList"), gestureGrid: $("gestureGrid"),
    toast: $("toast")
  };

  var lastText = null, lastHistory = null, lastGesture = null, pollTimer = null, toastTimer = null;
  var cameraOn = false;

  /* ----------------------------------------------------------- helpers */
  function api(url, method, body) {
    return fetch(url, {
      method: method || "GET",
      headers: body ? { "Content-Type": "application/json" } : undefined,
      body: body ? JSON.stringify(body) : undefined
    }).then(function (r) {
      return r.json().catch(function () { return {}; }).then(function (data) {
        data._status = r.status;
        return data;
      });
    });
  }

  function toast(msg) {
    el.toast.textContent = msg;
    el.toast.classList.add("show");
    clearTimeout(toastTimer);
    toastTimer = setTimeout(function () { el.toast.classList.remove("show"); }, 2200);
  }

  function showError(msg) {
    if (msg) { el.errorText.textContent = msg; el.errorBox.hidden = false; }
    else { el.errorBox.hidden = true; }
  }

  function escapeHtml(s) {
    return s.replace(/[&<>"']/g, function (c) {
      return { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c];
    });
  }

  /* ----------------------------------------------------------- camera */
  function setCameraUI(on) {
    cameraOn = on;
    el.btnStart.disabled = on;
    el.btnStop.disabled = !on;
    el.wrap.classList.toggle("live", on);
    el.placeholder.hidden = on;
    el.video.hidden = !on;
    el.pill.classList.toggle("live", on);
    el.statusText.textContent = on ? "Camera live" : "Camera off";
    if (on) {
      el.video.src = "/video_feed?t=" + Date.now();   // new URL = fresh stream
    } else {
      el.video.removeAttribute("src");
    }
  }

  function startCamera() {
    showError(null);
    el.btnStart.disabled = true;
    el.statusText.textContent = "Starting camera...";
    api("/api/camera/start", "POST").then(function (res) {
      if (res.ok) {
        setCameraUI(true);
      } else {
        setCameraUI(false);
        showError(res.error || "The camera could not be started.");
      }
    }).catch(function () {
      setCameraUI(false);
      showError("Cannot reach the server. Is `python app.py` still running?");
    });
  }

  function stopCamera() {
    api("/api/camera/stop", "POST").then(function () {
      setCameraUI(false);
      renderGesture("Camera is off", false, 0, 0);
    }).catch(function () { setCameraUI(false); });
  }

  el.video.addEventListener("error", function () {
    if (cameraOn) { /* stream ended: the status poll will explain why */ }
  });

  /* ----------------------------------------------------------- rendering */
  function renderGesture(name, recognized, confidence, progress) {
    var g = el.gestureName;
    if (name !== lastGesture) {
      g.textContent = name;
      g.classList.toggle("long", name.length > 5 && recognized);
      var muted = name === "Camera is off" || name === "No hand detected";
      g.classList.toggle("muted", muted);
      g.classList.toggle("bad", name === "Gesture not recognized");
      if (recognized) {
        g.classList.add("pop");
        setTimeout(function () { g.classList.remove("pop"); }, 200);
      }
      lastGesture = name;
      document.querySelectorAll(".g-item").forEach(function (item) {
        item.classList.toggle("active", recognized && item.dataset.label === name);
      });
    }
    el.confText.textContent = recognized ? Math.round(confidence * 100) + "% rule match" : "—";
    el.confBar.style.width = recognized ? Math.round(confidence * 100) + "%" : "0%";
    el.holdBar.style.width = Math.round(progress * 100) + "%";
    el.holdText.textContent = recognized
      ? (progress > 0 ? "Hold steady... " + Math.round(progress * 100) + "%" : "Change sign to type it again")
      : "Hold a sign steady to add it to the text";
  }

  function renderText(text) {
    if (text === lastText) return;
    lastText = text;
    if (!text) {
      el.textBox.innerHTML = '<span class="placeholder-text">Your translation will appear here…</span>';
    } else {
      el.textBox.innerHTML = escapeHtml(text) + '<span class="caret"></span>';
      el.textBox.scrollTop = el.textBox.scrollHeight;
    }
  }

  function renderHistory(history) {
    var key = JSON.stringify(history);
    if (key === lastHistory) return;
    lastHistory = key;
    if (!history.length) {
      el.historyList.innerHTML = '<li class="empty">Nothing yet. Press Clear, Save or Speak to store a sentence.</li>';
      return;
    }
    el.historyList.innerHTML = history.map(function (h) {
      return '<li><span class="t">' + escapeHtml(h.text) + '</span><span class="time">' + h.time + "</span></li>";
    }).join("");
  }

  /* ----------------------------------------------------------- polling */
  function poll() {
    api("/api/status").then(function (s) {
      renderGesture(s.gesture, s.recognized, s.confidence, s.progress);
      renderText(s.text);
      renderHistory(s.history);

      if (cameraOn && !s.camera_running) {      // camera stopped by itself (error / unplugged)
        setCameraUI(false);
        showError(s.error || "The camera stopped.");
      }
      if (!cameraOn && s.camera_running) {      // page was reloaded while camera was running
        setCameraUI(true);
      }
    }).catch(function () {
      el.statusText.textContent = "Server not reachable";
      el.pill.classList.remove("live");
    });
  }

  /* ----------------------------------------------------------- text buttons */
  function textAction(action, okMessage) {
    api("/api/text/" + action, "POST").then(function (res) {
      if (res.ok) {
        renderText(res.text);
        renderHistory(res.history);
        if (okMessage) toast(okMessage);
      } else if (res.error) {
        toast(res.error);
      }
    });
  }

  function currentText() { return lastText || ""; }

  function copyText() {
    var text = currentText().trim();
    if (!text) { toast("Nothing to copy yet"); return; }
    function fallback() {
      var ta = document.createElement("textarea");
      ta.value = text; ta.style.position = "fixed"; ta.style.opacity = "0";
      document.body.appendChild(ta); ta.select();
      var ok = false;
      try { ok = document.execCommand("copy"); } catch (e) { ok = false; }
      document.body.removeChild(ta);
      toast(ok ? "Copied to clipboard" : "Copy failed - select the text manually");
    }
    if (navigator.clipboard && window.isSecureContext) {
      navigator.clipboard.writeText(text).then(function () { toast("Copied to clipboard"); }, fallback);
    } else { fallback(); }
  }

  function speak() {
    if (!currentText().trim()) { toast("Nothing to speak yet"); return; }
    api("/api/speak", "POST").then(function (res) {
      if (res.ok) { toast("Speaking..."); return; }
      // Python voice not available -> use the browser's built-in voice instead
      if (window.speechSynthesis && res.text) {
        window.speechSynthesis.cancel();
        window.speechSynthesis.speak(new SpeechSynthesisUtterance(res.text.toLowerCase()));
        toast("Speaking (browser voice)");
        textAction("save");
      } else {
        toast(res.message || "Speech is not available");
      }
    }).catch(function () { toast("Could not reach the server"); });
  }

  /* ----------------------------------------------------------- supported gestures */
  function loadGestures() {
    api("/api/gestures").then(function (list) {
      if (!Array.isArray(list)) throw new Error("bad response");
      el.gestureGrid.innerHTML = list.map(function (g) {
        return '<div class="g-item ' + g.type + '" data-label="' + escapeHtml(g.label) +
          '" title="' + escapeHtml(g.description) + '"><div class="l">' + escapeHtml(g.label) +
          '</div><div class="d">' + escapeHtml(g.description) + "</div></div>";
      }).join("");
    }).catch(function () {
      el.gestureGrid.innerHTML = '<p class="empty">Could not load the gesture list.</p>';
    });
  }

  /* ----------------------------------------------------------- wire up */
  el.btnStart.addEventListener("click", startCamera);
  el.btnStop.addEventListener("click", stopCamera);
  $("btnSpace").addEventListener("click", function () { textAction("space"); });
  $("btnDelete").addEventListener("click", function () { textAction("delete"); });
  $("btnClear").addEventListener("click", function () { textAction("clear", "Text cleared (saved to history)"); });
  $("btnSave").addEventListener("click", function () { textAction("save", "Saved to history"); });
  $("btnCopy").addEventListener("click", copyText);
  $("btnSpeak").addEventListener("click", speak);
  $("btnClearHistory").addEventListener("click", function () {
    api("/api/history/clear", "POST").then(function () { renderHistory([]); toast("History cleared"); });
  });

  loadGestures();
  poll();
  pollTimer = setInterval(poll, 250);
})();
