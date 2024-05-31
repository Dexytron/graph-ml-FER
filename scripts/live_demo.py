import cv2
import mediapipe as mp
import time
import numpy as np
import plotting_utils as utils
import image_landmarks_generation as ilg
from model_inference import infer_model

# variables
frame_counter = 0

# constants
FONTS = cv2.FONT_HERSHEY_COMPLEX

# face bounder indices
FACE_OVAL = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109]

# lips indices for Landmarks
LIPS = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95, 185, 40, 39, 37, 0, 267, 269, 270, 409, 415, 310, 311, 312, 13, 82, 81, 42, 183, 78]
LOWER_LIPS = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95]
UPPER_LIPS = [185, 40, 39, 37, 0, 267, 269, 270, 409, 415, 310, 311, 312, 13, 82, 81, 42, 183, 78]
# Left eyes indices
LEFT_EYE = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
LEFT_EYEBROW = [336, 296, 334, 293, 300, 276, 283, 282, 295, 285]
LEFT_IRIS = [474, 475, 476, 477]

# right eyes indices
RIGHT_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
RIGHT_EYEBROW = [70, 63, 105, 66, 107, 55, 65, 52, 53, 46]
RIGHT_IRIS = [469, 470, 471, 472]


map_face_mesh = mp.solutions.face_mesh

# camera object
camera = cv2.VideoCapture(0)


# landmark detection function
def landmarksDetection(img, results, draw=False):
    img_height, img_width = img.shape[:2]
    mesh_coord = [(int(point.x * img_width), int(point.y * img_height)) for point in
                  results.multi_face_landmarks[0].landmark]
    if draw:
        [cv2.circle(img, p, 2, utils.GREEN, -1) for p in mesh_coord]
    return mesh_coord


with map_face_mesh.FaceMesh(min_detection_confidence=0.9, refine_landmarks=True) as face_mesh:
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

        frame = utils.textWithBackground(frame, f'FPS: {round(fps, 1)} EXPRESSION: {expression}', FONTS,
                                         1.0, (20, 50), bgOpacity=0.9, textThickness=2)
        cv2.imshow('frame', frame)
        key = cv2.waitKey(1)
        if key == ord('q') or key == ord('Q'):
            break
    cv2.destroyAllWindows()
    camera.release()
