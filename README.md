# Real-Time Sign Language Translator

A web application that watches your hand through a webcam, recognises a small set of
static sign-language gestures, and turns them into English text that can be edited,
copied, saved and spoken aloud.

> **Honest scope:** this is a basic college-level project. It recognises **13 static
> gestures** (listed below) using a **rule-based classifier on MediaPipe hand
> landmarks**. It does **not** translate full sign language, and anything it does not
> know is shown as *"Gesture not recognized"*. Nothing is faked or randomly generated.

---

## 1. Problem statement
Millions of people who are deaf or hard of hearing communicate with sign language, but
most hearing people do not know it. This creates a communication barrier in everyday
situations. A computer-vision tool that converts signs to text and speech can help
bridge that gap and also help beginners learn the basics of finger-spelling.

## 2. Objectives
- Capture live video from a webcam and display it in a web page.
- Detect the hand and its 21 landmarks with MediaPipe.
- Recognise a manageable vocabulary of static gestures and show a confidence score.
- Build words and sentences from recognised letters.
- Provide text controls (clear, delete, space, copy), text-to-speech and a history.
- Keep the code simple, readable and easy to explain.

## 3. Technologies used
| Technology | Purpose |
|---|---|
| Python 3 | Main language |
| OpenCV | Webcam capture, mirroring, drawing, JPEG encoding |
| MediaPipe | Hand detection and 21 landmark points |
| NumPy | Distance and angle calculations |
| Flask | Backend web server and JSON API |
| HTML / CSS / JavaScript | Dashboard user interface |
| pyttsx3 | Free, offline text-to-speech |

All libraries are free and open source.

## 4. System requirements
- Windows 10/11, macOS or Linux
- **Python 3.9 - 3.12 recommended** (3.13 also works with newer MediaPipe, see Troubleshooting)
- A working webcam
- 4 GB RAM or more; no GPU needed
- A modern browser (Chrome, Edge, Firefox)
- Internet **only for installation** (and once for the model file on newer MediaPipe)

## 5. Project folder structure
```
Real_Time_Sign_Language_Translator/
├── app.py                  # Flask server: pages + JSON API + video stream
├── camera_service.py       # OpenCV webcam loop: detect -> classify -> draw -> stream
├── gesture_recognition.py  # MediaPipe detector + rule-based classifier + stabiliser
├── translation_state.py    # Translated text and history (Space/Delete/Clear/Save)
├── text_to_speech.py       # pyttsx3 text-to-speech (runs in a background thread)
├── requirements.txt        # Python dependencies
├── README.md               # This file
├── PROJECT_REPORT.md       # Formal report for submission
├── templates/
│   └── index.html          # The dashboard page
├── static/
│   ├── style.css           # Blue/purple gradient theme
│   └── script.js           # Talks to the Flask API
├── models/
│   └── README.txt          # Where the optional MediaPipe model file goes
└── tests/
    ├── synthetic_hands.py  # Builds artificial hands for testing the rules
    └── test_project.py     # 19 automated tests
```

## 6. Installation
Open a terminal **inside the project folder** after unzipping.

**Windows (PowerShell / Command Prompt)**
```bash
cd Real_Time_Sign_Language_Translator
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

**macOS / Linux**
```bash
cd Real_Time_Sign_Language_Translator
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```
*Linux only, for the offline voice:* `sudo apt install espeak-ng libespeak1`

## 7. How to run
```bash
python app.py
```
Your browser opens at **http://127.0.0.1:5000**. Click **Start Camera**, allow camera
access if the browser asks, and show a gesture. Press `CTRL+C` in the terminal to quit.

Optional: `python -m unittest discover -s tests -v` runs the automated tests (no camera needed).

## 8. Supported gestures
| Sign | How to make it |
|---|---|
| **A** | Closed fist, thumb against the side pointing up |
| **B** | Flat hand, four fingers up and together, thumb across the palm |
| **C** | Fingers and thumb curved like a "C" |
| **D** | Index finger up; thumb touches the curled middle finger |
| **E** | All fingers curled down, thumb tucked in front |
| **F** | Thumb and index touch in a circle; other three fingers up |
| **I** | Fist with only the little finger up |
| **L** | Index finger up, thumb out sideways |
| **V** | Index and middle fingers up and spread (peace sign) |
| **W** | Index, middle, ring fingers up and spread; little finger folded |
| **Y** | Thumb and little finger out, others curled |
| **HELLO** | Open hand, five fingers spread wide |
| **I LOVE YOU** | Thumb, index and little finger out; middle and ring folded |

Shapes are inspired by ASL finger-spelling but simplified, so they are not always identical
to a sign language textbook. Letters that need movement (J, Z) are not included.

### Using it well
- Hand 30-60 cm from the camera, **palm facing the camera**, whole hand in frame.
- Plain background and good, even lighting. One hand only.
- **Hold a sign steady for about 1 second** - the progress bar fills and the letter is typed.
- To type the **same letter twice** (e.g. the two L's in HELLO) show a different sign in between.
- Left or right hand both work.

## 9. How the system works
```
Webcam --> OpenCV frame --> mirror --> MediaPipe (21 hand landmarks)
      --> rule-based classifier --> hold-steady check (stabiliser)
      --> Flask (/video_feed, /api/status) --> Browser dashboard
```
1. `camera_service.py` reads frames from the webcam with OpenCV in a background thread.
2. `gesture_recognition.HandDetector` asks MediaPipe for the 21 landmarks of the hand.
3. **Feature extraction:** for each finger the angle at its middle joint tells whether it is
   *straight*, *bent* or *curled*. Distances (thumb-to-index tip, finger spread, ...) are
   divided by the palm length, so the result does not depend on how close the hand is.
   Angles and ratios also do not change when the hand is rotated.
4. **Rules** match the pattern to a gesture, e.g. *Y = thumb out + pinky straight + other three curled*.
   If nothing matches the result is `None` -> **"Gesture not recognized"**.
5. **Confidence** = how clearly each finger sits inside its expected state (far from the
   decision thresholds = higher). It is a *rule-match score*, not a trained probability.
6. The **stabiliser** only types a gesture after it is held ~1 s, then waits for the hand to
   change before repeating it. This avoids accidental letters.
7. Flask streams the annotated video as MJPEG and the page polls `/api/status` four times a
   second for the gesture, text and history.

### Why rules instead of machine learning?
The brief says machine learning *only if practical*. A trained model needs hundreds of
labelled photos per gesture, plus careful testing across different hands and lighting.
Rules on MediaPipe landmarks are accurate enough for static signs, need no dataset, run
instantly on any laptop, and - importantly for a viva - **every decision can be explained**.
(See *Future improvements* for the ML upgrade path.)

## 10. Features
- Start / Stop camera with live preview and drawn hand skeleton
- Real-time recognition of 13 static gestures with confidence
- "No hand detected" and "Gesture not recognized" states
- Word building (letters) and whole words (HELLO, I LOVE YOU)
- Text controls: Space, Delete, Clear, Copy, Save
- Text-to-speech (offline pyttsx3, with the browser voice as automatic fallback)
- Translation history (last 20) with Clear history
- Instructions, tips, supported-gesture guide and About section
- Clear error messages when the camera is missing, busy or disconnected
- Responsive blue/purple glass-style dashboard, works fully offline after install

## 11. Troubleshooting
| Problem | Fix |
|---|---|
| "No webcam could be opened" | Close Zoom/Teams/other apps using the camera; allow camera access in OS privacy settings; set another camera: `CAMERA_INDEX=1 python app.py` (Windows: `set CAMERA_INDEX=1`) |
| "hand model file is missing" | Newer MediaPipe needs `models/hand_landmarker.task`. Connect to the internet once, or follow `models/README.txt`. Alternative: `pip install mediapipe==0.10.14` (Python 3.8-3.12) |
| `pip install mediapipe` fails | Use Python 3.10-3.12 and upgrade pip: `python -m pip install --upgrade pip` |
| No voice | Linux: `sudo apt install espeak-ng libespeak1`. The page falls back to the browser voice automatically |
| Gesture flickers / not recognised | Better light, plain background, hold hand flatter to the camera, match the guide. Thresholds can be tuned at the top of `gesture_recognition.py` |
| Port 5000 busy (macOS AirPlay) | `PORT=5001 python app.py` (Windows: `set PORT=5001`) |

## 12. Limitations
- Only 13 **static** gestures; no motion signs (J, Z), no grammar, no full sign language.
- Rule thresholds were verified on artificial hand data and must be tuned on real hands;
  similar shapes (A vs E, D vs "1", U vs V) can be confused or rejected.
- One hand at a time; the hand must be fully visible and roughly facing the camera.
- Works best with good lighting; very dark or cluttered scenes reduce MediaPipe accuracy.
- Single-user local app (state is kept in the server process).

## 13. Future improvements
- Collect a dataset of landmarks and train a classifier (k-NN / Random Forest / small neural net) for all 26 letters and digits.
- Dynamic gestures (J, Z, words) with an LSTM over landmark sequences.
- Two-hand signs and facial expression/body pose using MediaPipe Holistic.
- Word suggestions / autocorrect, multiple languages, mobile and cloud deployment.
- User accounts and saved conversation history.

## 14. Conclusion
The project shows a complete, working computer-vision pipeline - camera, landmark
detection, gesture logic, web backend and a polished interface - while staying honest
about what it can and cannot recognise. It is a solid foundation that can grow into a
full sign-language recognition system.
