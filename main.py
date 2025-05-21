import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['MEDIAPIPE_DISABLE_GPU'] = '1'

import cv2
import mediapipe as mp
import numpy as np

# Open webcam safely
cap = cv2.VideoCapture(1)
if not cap.isOpened():
    print("Could not access webcam.")
    exit()

# Initialize FaceMesh with minimal options
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

RIGHT_EYELID = [159, 153, 155]
RIGHT_IRIS = [474, 475, 476, 477]

def get_center(landmarks, indices, w, h):
    return np.mean([[l.x * w, l.y * h] for i, l in enumerate(landmarks) if i in indices], axis=0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w, _ = frame.shape
    results = face_mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    if results.multi_face_landmarks:
        face = results.multi_face_landmarks[0]
        eyelid = get_center(face.landmark, RIGHT_EYELID, w, h)
        iris = get_center(face.landmark, RIGHT_IRIS, w, h)

        # Draw tracking
        cv2.line(frame, tuple(eyelid.astype(int)), tuple(iris.astype(int)), (0, 255, 255), 2)
        cv2.circle(frame, tuple(eyelid.astype(int)), 3, (255, 0, 255), -1)
        cv2.circle(frame, tuple(iris.astype(int)), 3, (0, 255, 0), -1)

    cv2.imshow("Eye Line", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
