import cv2
import dlib
import numpy as np
from imutils import face_utils

# ——— Constants ———
EAR_THRESHOLD   = 0.3       # threshold for "eye closed"
YAW_THRESHOLD = 35          # threshold for "head shaking" (degree)
PITCH_THRESHOLD_UPPER = 175 # upper threshold for "head nodding" (degree)
PITCH_THRESHOLD_LOWER = 160 # lower threshold for "head nodding" (degree)
CONSEC_FRAMES   = 24        # consecutive frames for drowsiness and distraction

margin = 10                 # margin from the edge
line_height = 25            # how far apart each line is

# ——— Load Models ———
detector  = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# facial landmark indices for the eyes
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

# 3D model points of facial landmarks
model_points = np.array([
    (0.0, 0.0, 0.0),             # Nose tip
    (0.0, -330.0, -65.0),        # Chin
    (-225.0, 170.0, -135.0),     # Left eye left corner
    (225.0, 170.0, -135.0),      # Right eye right corner
    (-150.0, -150.0, -125.0),    # Left Mouth corner
    (150.0, -150.0, -125.0)      # Right mouth corner
], dtype=np.float64)

# compute Eye Aspect Ratio
def eye_aspect_ratio(eye):
    A = np.linalg.norm(eye[1] - eye[5])
    B = np.linalg.norm(eye[2] - eye[4])
    C = np.linalg.norm(eye[0] - eye[3])
    return (A + B) / (2.0 * C)

# ——— Start Webcam ———
cap = cv2.VideoCapture(0)
frame_counter_drownsiness = 0
frame_counter_distraction = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray, 0)

    size = frame.shape
    focal_length = size[1]
    center = (size[1] / 2, size[0] / 2)
    camera_matrix = np.array([
        [focal_length, 0, center[0]],
        [0, focal_length, center[1]],
        [0, 0, 1]
    ], dtype="double")
    dist_coeffs = np.zeros((4,1))  # assuming no lens distortion

    for face in faces:
        shape = predictor(gray, face)
        landmarks = face_utils.shape_to_np(shape)
        h, w = frame.shape[:2]

        # ——— Drowsiness (EAR) ———
        # extract eyes and compute EAR
        leftEye  = landmarks[lStart:lEnd]
        rightEye = landmarks[rStart:rEnd]
        ear = (eye_aspect_ratio(leftEye) + eye_aspect_ratio(rightEye)) / 2.0

        # draw eye contours
        cv2.polylines(frame, [cv2.convexHull(leftEye)],  True, (0, 255, 0), 1)
        cv2.polylines(frame, [cv2.convexHull(rightEye)], True, (0, 255, 0), 1)

        # display ear number 
        cv2.putText(frame, f"EAR: {ear:.2f}",   (margin, h - margin - (2 * line_height)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

        # check drowsiness
        if ear < EAR_THRESHOLD:
            frame_counter_drownsiness += 1
            if frame_counter_drownsiness >= CONSEC_FRAMES:
                cv2.putText(frame, "DROWSINESS ALERT!", (margin, 2*margin),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)
        else:
            frame_counter_drownsiness = 0

        # ——— Head Pose Estimation ———
        # 2D image points
        image_points = np.array([
            landmarks[30],     # Nose tip
            landmarks[8],      # Chin
            landmarks[36],     # Left eye left corner
            landmarks[45],     # Right eye right corner
            landmarks[48],     # Left Mouth corner
            landmarks[54]      # Right mouth corner
        ], dtype="double")

        # solvePnP
        success, rotation_vec, translation_vec = cv2.solvePnP(
            model_points, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE
        )

        # project the 3D model points to 2D image points
        projected_pts, _ = cv2.projectPoints(
            model_points, rotation_vec, translation_vec, camera_matrix, dist_coeffs
        )

        # draw red circle at each projected point
        for idx, p in enumerate(projected_pts):
            x, y = int(p[0][0]), int(p[0][1])
            cv2.circle(frame, (x, y), 4, (0, 0, 255), -1)

        # convert to angles
        rmat, _ = cv2.Rodrigues(rotation_vec)
        proj_matrix = np.hstack((rmat, translation_vec))
        _, _, _, _, _, _, euler_angles = cv2.decomposeProjectionMatrix(proj_matrix)
        yaw   = euler_angles[1,0]
        pitch = euler_angles[0,0]

        # display angles number
        cv2.putText(frame, f"Yaw: {yaw:.1f}",   (margin, h - margin - (1 * line_height)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        cv2.putText(frame, f"Pitch: {pitch:.1f}", (margin, h - margin),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

        # check distraction
        if abs(yaw) > YAW_THRESHOLD or abs(pitch) > PITCH_THRESHOLD_UPPER or abs(pitch) < PITCH_THRESHOLD_LOWER:
            frame_counter_distraction += 1
            if frame_counter_distraction >= CONSEC_FRAMES:
                cv2.putText(frame, "DISTRACTION ALERT!", (margin, 2*margin + (1 * line_height)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)
        else:
            frame_counter_distraction = 0

    cv2.imshow("Drowsiness and Distraction Detector", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
