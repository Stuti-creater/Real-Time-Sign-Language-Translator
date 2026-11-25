# REAL-TIME SIGN LANGUAGE TRANSLATOR

**A Project Report**

Submitted by: ______________________  Roll No.: ____________
Course / Department: ______________________  Institution: ______________________
Guide: ______________________  Academic Year: ____________

---

## 1. Abstract
Communication between sign-language users and people who do not know sign language is
difficult. This project presents a **Real-Time Sign Language Translator**, a web
application that uses a webcam to recognise static hand gestures and convert them into
English text and speech. The system uses **OpenCV** to capture video, **MediaPipe** to
extract 21 hand landmarks, and a **rule-based classifier** that analyses finger angles and
distances to recognise 13 gestures (letters A, B, C, D, E, F, I, L, V, W, Y and the words
HELLO and I LOVE YOU). A **Flask** backend streams the video and exposes a JSON API to an
**HTML/CSS/JavaScript** dashboard that provides live translation, word building, text
editing, text-to-speech (pyttsx3) and translation history. The system runs in real time on an
ordinary laptop without a GPU. Unsupported gestures are reported as "Gesture not
recognized" rather than guessed.

## 2. Introduction
Sign language is a visual language that uses hand shapes, movements and expressions.
Advances in computer vision, especially real-time hand-landmark detection, make it
possible to build affordable assistive tools. MediaPipe Hands can locate 21 3-D points on a
hand from a single RGB camera. Once these points are known, a hand pose can be analysed
geometrically. This project uses that idea to translate finger-spelling-style gestures
into text and speech through a simple, modern web interface.

## 3. Problem Definition
Most hearing people cannot understand sign language, and interpreters are not always
available. The aim is to design a low-cost, software-only system that:
1. captures a user's hand in real time with an ordinary webcam,
2. identifies a defined set of static gestures,
3. converts them into readable text and speech, and
4. presents everything in an easy-to-use web interface.

**Scope:** static gestures only; one hand; a documented, limited vocabulary. Full sign-language
translation (grammar, motion, facial expression) is out of scope.

## 4. Proposed System
The proposed system is a client-server application. The server (Python) reads the camera,
detects the hand, classifies the gesture and streams annotated video. The client
(browser) shows the video, the recognised gesture with confidence, the translated text,
and provides controls.

**Key design decisions**
- *Landmarks + rules instead of image classification:* works without a training dataset,
  is fast, and every decision is explainable.
- *Hold-to-type stabiliser:* a gesture must be steady for ~1 second to be typed, which
  prevents accidental letters.
- *Honest output:* when no rule matches, the UI shows "Gesture not recognized".
- *Graceful failure:* camera and model problems produce clear messages instead of crashes.

## 5. System Architecture
```
 +---------+   frames   +-----------------+  21 landmarks  +------------------+
 | Webcam  | ---------> | OpenCV capture  | -------------> | MediaPipe Hands  |
 +---------+            | (camera_service)|                +--------+---------+
                        +--------+--------+                         |
                                 | annotated JPEG                   v
                                 |                         +------------------+
                                 |                         | Rule-based       |
                                 |                         | classifier       |
                                 |                         +--------+---------+
                                 |                                  | label + confidence
                                 |                         +--------v---------+
                                 |                         | Stabiliser       |
                                 |                         | (hold ~1 second) |
                                 |                         +--------+---------+
                                 v                                  v
                        +------------------------------------------------------+
                        |                   Flask backend (app.py)             |
                        |  /video_feed   /api/status   /api/text/*   /api/speak|
                        +--------------------------+---------------------------+
                                                   | HTTP (MJPEG + JSON)
                                         +---------v----------+
                                         | Browser dashboard  |
                                         | HTML + CSS + JS    |
                                         +--------------------+
```

**Data flow:** frame -> mirror -> landmarks -> features -> rule match -> stabiliser -> text state
-> JSON status -> browser.

## 6. Modules
| Module | File | Responsibility |
|---|---|---|
| Web server | `app.py` | Routes, MJPEG stream, JSON API, error handling |
| Camera service | `camera_service.py` | Opens webcam, runs the per-frame pipeline in a thread, draws overlay |
| Hand detector | `gesture_recognition.py` (`HandDetector`) | MediaPipe landmarks (supports old `solutions` API and new Tasks API) |
| Classifier | `gesture_recognition.py` (`classify_landmarks`) | Finger states, distances, rules, confidence |
| Stabiliser | `gesture_recognition.py` (`GestureStabilizer`) | Hold-to-confirm, no repeats until the hand changes |
| Translation state | `translation_state.py` | Text, word building, delete/space/clear, history (max 20) |
| Text-to-speech | `text_to_speech.py` | Offline speech via pyttsx3 in a background thread |
| Frontend | `templates/index.html`, `static/style.css`, `static/script.js` | Dashboard, polling, buttons, clipboard, browser-voice fallback |
| Tests | `tests/` | Synthetic-hand rule tests, stabiliser, text, API and pipeline tests |

## 7. Hardware and Software Requirements
**Hardware:** laptop/PC with a webcam (720p or better recommended), 4 GB RAM (8 GB preferred),
dual-core CPU or better, speakers for text-to-speech. No GPU required.

**Software:** Windows 10/11, macOS or Linux; Python 3.9-3.12 (3.13 supported with newer
MediaPipe); Flask >= 3.0; opencv-python >= 4.8; mediapipe >= 0.10.14; NumPy >= 1.24;
pyttsx3 >= 2.90; a modern web browser. (Linux voice: espeak-ng.)

## 8. Implementation

### 8.1 Hand detection
MediaPipe returns 21 landmarks (x, y normalised; z relative depth). They are converted to
pixel units so distances are not distorted by the image aspect ratio. The frame is mirrored
so the preview behaves like a mirror. Only one hand is tracked.

### 8.2 Feature extraction
- **Finger state:** the angle at the middle joint (PIP) of each finger. >= 150 degrees = *straight*,
  <= 100 degrees = *curled*, in between = *bent*.
- **Palm length** (wrist to middle knuckle) is used as a unit, so features are independent of
  hand size and camera distance; angles and ratios are independent of hand rotation.
- **Distances:** thumb-tip to index-knuckle (thumb out), thumb-tip to index-tip (pinch),
  thumb-tip to middle-tip (touch), index-tip to pinky-tip (spread), index-tip to middle-tip
  (V gap), index-tip to ring-tip (W gap).

### 8.3 Rules
| Gesture | Rule (summary) |
|---|---|
| A | four fingers curled, thumb pointing up past the knuckle, not touching fingertips |
| B | four fingers straight, tips close together, thumb not out |
| C | at least three fingers bent, thumb-index gap 0.5-1.5 palm lengths |
| D | index straight, others down, thumb touching middle fingertip |
| E | four fingers down, thumb tip touching the curled fingertips |
| F | thumb and index touching; middle, ring, pinky straight |
| I | only pinky straight, thumb not out |
| L | index straight, others down, thumb out about perpendicular to index |
| V | index and middle straight and spread; ring and pinky down |
| W | index, middle, ring straight and spread; pinky down |
| Y | pinky straight, thumb out, others down |
| HELLO | all four fingers straight and widely spread |
| I LOVE YOU | thumb, index, pinky out; middle and ring down |

If none match, the classifier returns *no label*.

### 8.4 Confidence score
Confidence is the average "clarity" of the four finger states (0.6 near a threshold to 0.99
well inside it). It indicates how clearly the hand matches the rule; it is **not** a trained
probability, and the interface says so.

### 8.5 Stabiliser and word building
A gesture must be seen continuously for 1 second; the progress bar shows this. After typing,
the same gesture is blocked until a different gesture has been shown for 0.5 s. Letters are
appended directly, words (HELLO, I LOVE YOU) are appended with spaces.

### 8.6 Backend API
| Endpoint | Method | Purpose |
|---|---|---|
| `/` | GET | Dashboard page |
| `/api/camera/start`, `/api/camera/stop` | POST | Turn camera on/off (503 + message if unavailable) |
| `/video_feed` | GET | MJPEG stream of annotated frames |
| `/api/status` | GET | Gesture, confidence, hold progress, text, history |
| `/api/text/<clear|delete|space|save>` | POST | Text controls |
| `/api/history/clear` | POST | Clear history |
| `/api/speak` | POST | Speak current text (pyttsx3) |
| `/api/gestures` | GET | Supported-gesture list for the guide |

### 8.7 Frontend
A responsive three-column dashboard with a blue/purple gradient theme: instructions (left),
camera + gesture readout + translated text + controls (centre), history and supported
gestures (right), and an About section. JavaScript polls the status endpoint every 250 ms,
handles clipboard copy (with fallback), and falls back to the browser's speech synthesis if the
Python voice is unavailable.

### 8.8 Error handling
Missing camera, busy camera, unplugged camera, missing MediaPipe model, missing TTS engine,
server unreachable and empty-text actions are all detected and shown as readable messages.

## 9. Testing
Testing was done at three levels.

**9.1 Automated tests (19, `python -m unittest discover -s tests -v`)** - all passing at delivery:
- every supported gesture is recognised from an artificial hand model, at five different
  rotations and at a different scale;
- five unsupported shapes (e.g. U with fingers together, flat hand with thumb out, "horns")
  return no label;
- stabiliser: types once per hold, allows the same letter again after a different sign,
  ignores brief tracking flicker, resets on gesture change;
- text state: word building, delete, space, clear, history limit, history clear;
- Flask API: page and static files load, status, text controls, camera-unavailable error (503),
  video-feed refusal when off, empty speech request;
- full camera pipeline with a simulated webcam and detector: JPEG frames are produced and
  letters are typed in order.

**9.2 Static checks:** all Python files compile; the JavaScript passes a syntax check; HTML
references to `/static/style.css` and `/static/script.js` resolve; the page was rendered in a
headless browser and inspected for layout and console errors.

**9.3 Live testing procedure (to be completed on the demonstration machine)**
Test each gesture 10 times with real hands under normal indoor light and record the result.

| Gesture | Trials | Correct | Not recognized | Wrong letter | Notes |
|---|---|---|---|---|---|
| A |  |  |  |  |  |
| B |  |  |  |  |  |
| C |  |  |  |  |  |
| D |  |  |  |  |  |
| E |  |  |  |  |  |
| F |  |  |  |  |  |
| I |  |  |  |  |  |
| L |  |  |  |  |  |
| V |  |  |  |  |  |
| W |  |  |  |  |  |
| Y |  |  |  |  |  |
| HELLO |  |  |  |  |  |
| I LOVE YOU |  |  |  |  |  |

## 10. Results
- The complete pipeline (camera -> landmarks -> rules -> stabiliser -> web UI -> speech) is
  implemented and works end to end.
- The rule logic classified all 13 gestures correctly on artificial landmark data and
  rejected unsupported shapes, independent of hand rotation and size.
- The interface displays the live preview with a drawn hand skeleton, the recognised
  gesture with a rule-match score, the editable translation, and the history.
- Real-world accuracy figures depend on the user, camera and lighting and should be entered
  from the live test table in section 9.3. (No accuracy percentage is claimed here because it
  has not been measured on real hands.)

## 11. Limitations
1. Static gestures only (13 signs); no motion-based signs such as J and Z, and no grammar.
2. Rules and thresholds were verified on artificial hands; real hands may require tuning of the
   constants in `gesture_recognition.py`. Similar shapes (A/E, D/"1", U/V) may be confused or rejected.
3. One hand, fully visible, roughly palm-facing the camera.
4. Sensitive to lighting and cluttered backgrounds.
5. Single-user local server; state is held in memory and lost on restart.
6. Confidence is a rule-match score, not a statistical probability.

## 12. Future Scope
- Train a machine-learning classifier on recorded landmarks to cover the full alphabet and digits.
- Recognise dynamic signs using sequence models (LSTM / Transformer) on landmark sequences.
- Use MediaPipe Holistic for two hands, face and body pose.
- Add autocorrect and word prediction; support multiple sign languages and spoken languages.
- Mobile app / WebRTC deployment so the camera runs in the user's browser while the model runs in the cloud.

## 13. Conclusion
The project demonstrates a working real-time sign-language-to-text-and-speech system built from
free, open-source tools. By combining MediaPipe landmarks with transparent geometric rules,
it achieves responsive recognition of a defined vocabulary on ordinary hardware and presents
the result through a modern web dashboard. Its honest, explainable design and clear limitations
make it a sound foundation for future machine-learning extensions.

## 14. References
1. MediaPipe Hands / Hand Landmarker - Google, https://developers.google.com/mediapipe
2. OpenCV documentation - https://docs.opencv.org
3. Flask documentation - https://flask.palletsprojects.com
4. pyttsx3 documentation - https://pypi.org/project/pyttsx3
5. Zhang, F. et al., "MediaPipe Hands: On-device Real-time Hand Tracking", arXiv:2006.10214, 2020.
