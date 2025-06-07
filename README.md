# Drowsiness & Distraction Detector

`drowsiness_distraction_detection.py` implements a real-time driver monitoring system that uses facial landmarks and head‐pose estimation to detect both drowsiness and distraction from your webcam feed.

## Key Features

- **Live Video Capture**  
  Grabs frames continuously from the default webcam via OpenCV.

- **Face & Landmark Detection**  
  Uses dlib’s frontal-face detector plus a 68-point facial landmark predictor to localize eyes and key facial points.

- **Drowsiness Detection (EAR)**  
  Computes the Eye Aspect Ratio (EAR) from eye landmarks; flags “DROWSINESS ALERT!” when EAR stays below a configurable threshold (default `0.3`) for a set number of consecutive frames (default `24`).

- **Distraction Detection (Head Pose)**  
  Performs 3D head‐pose estimation via OpenCV’s `solvePnP` on six model points (nose tip, chin, eyes, mouth corners), extracts yaw and pitch angles, and flags “DISTRACTION ALERT!” if the driver looks away (configurable thresholds: yaw > `35°`, pitch outside `160°–175°`) for too many frames.

- **On-Screen Overlays**  
  Draws convex‐hull outlines around each eye, displays current EAR/yaw/pitch values, and renders alert text directly on the video stream.

- **Clean Shutdown**  
  Listens for “q” key to gracefully release the camera and close all windows.

## Dependencies

- Python 3.11  
- OpenCV (`opencv-python`)  
- dlib  
- imutils  
- NumPy  

You also need the pretrained landmark model file: `shape_predictor_68_face_landmarks.dat`

## Usage

1. Install dependencies:
   ```bash```
   ```
   pip install opencv-python dlib imutils numpy
3. Download the dlib shape predictor:
     
   Go to `http://dlib.net/files/` to download the predictor `shape_predictor_68_face_landmarks.dat`
   
4. Run the script:
   ```bash```
   ```
   python drowsiness_distraction_detection.py
6. Press `q` on your keyboard to exit
