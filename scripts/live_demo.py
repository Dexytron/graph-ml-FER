import cv2
import mediapipe as mp
import time
import numpy as np
import plotting_utils as utils
import image_landmarks_generation as ilg
from model_inference import infer_model

frame_counter = 0
camera = cv2.VideoCapture(0)


with mp.solutions.face_mesh.FaceMesh(min_detection_confidence=0.9, refine_landmarks=True) as face_mesh:
    start_time = time.time()
    while True:
        frame_counter += 1
        ret, frame = camera.read()
        if not ret:
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        results = face_mesh.process(rgb_frame)

        if results.multi_face_landmarks:
            landmarks = ilg.landmarks_detection(frame, results, True, marker_size=3)[:468]  # Omit the iris landmarks
            normalized_landmarks = ilg.normalize_landmarks(landmarks, ilg.reference_landmarks)
            expression = infer_model(normalized_landmarks)
        else:
            expression = 'None'

        end_time = time.time() - start_time
        fps = frame_counter / end_time

        frame = utils.textWithBackground(frame, f'FPS: {round(fps, 1)} EXPRESSION: {expression}',
                                         cv2.FONT_HERSHEY_COMPLEX,1.0, (20, 50), bgOpacity=0.9, textThickness=2)
        cv2.imshow('frame', frame)
        key = cv2.waitKey(1)
        if key == ord('q') or key == ord('Q'):
            break
    cv2.destroyAllWindows()
    camera.release()
