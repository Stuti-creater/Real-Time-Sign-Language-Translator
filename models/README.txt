Hand landmark model (only needed with newer MediaPipe versions)
===============================================================

Newer MediaPipe releases (those without "mp.solutions") need this file:

    hand_landmarker.task

The app downloads it automatically the first time you click Start Camera
(internet needed once). To download it by hand, open this link in a browser
and save the file in THIS folder under the same name:

https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task

With Python 3.8-3.12 you can instead run  pip install mediapipe==0.10.14
which does not need any model file.
