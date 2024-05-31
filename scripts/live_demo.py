import cv2
import mediapipe as mp
import time
import plotting_utils as utils
import image_landmarks_generation as ilg
from model_inference import infer_model

frame_counter = 0
camera = cv2.VideoCapture(0)


with mp.solutions.face_mesh.FaceMesh(refine_landmarks=True) as face_mesh:
    start_time = time.time()
    while True:
        frame_counter += 1
        ret, frame = camera.read()
        if not ret:
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        results = face_mesh.process(rgb_frame)

        if results.multi_face_landmarks:
            landmarks = ilg.landmarks_detection(frame, results, True, marker_size=3)
            landmarks = landmarks[:468]  # Omit the iris landmarks
            normalized_landmarks = ilg.normalize_landmarks(landmarks, ilg.reference_landmarks)
            expression = infer_model(normalized_landmarks)
        else:
            expression = 'None'

        end_time = time.time() - start_time
        fps = frame_counter / end_time

        cv2.flip(frame, 1, frame)
        frame = utils.textWithBackground(frame, f'FPS: {round(fps, 1)} EXPRESSION: {expression}', cv2.FONT_HERSHEY_SIMPLEX, 1.0, (40, 40), bgOpacity=0.5, textThickness=2, corner='bottom-right')
        cv2.imshow('frame', frame)

        key = cv2.waitKey(1)
        if key == ord('q') or key == ord('Q'):
            break

    cv2.destroyAllWindows()
    camera.release()
