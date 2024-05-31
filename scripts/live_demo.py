import cv2
import mediapipe as mp
import time

import numpy as np
from numpy import greater
import plotting_utils as utils
import image_landmarks_generation as ilg
from infer_model import infer_model

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
    # starting time here
    start_time = time.time()
    # starting Video loop here.
    while True:
        frame_counter += 1  # frame counter
        ret, frame = camera.read()  # getting frame from camera
        if not ret:
            break  # no more frames break
        #  resizing frame
        # frame = cv2.resize(frame, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
        # writing orginal image image thumbnail
        # cv2.imwrite(f'img/img_{frame_counter}.png', frame)
        # print(frame_counter)

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        results = face_mesh.process(rgb_frame)
        if results.multi_face_landmarks:
            mesh_coords = landmarksDetection(frame, results, False)
            frame = utils.fillPolyTrans(frame, [mesh_coords[p] for p in FACE_OVAL], utils.WHITE, opacity=0.4)
            frame = utils.fillPolyTrans(frame, [mesh_coords[p] for p in LEFT_EYE], utils.GREEN, opacity=0.4)
            frame = utils.fillPolyTrans(frame, [mesh_coords[p] for p in RIGHT_EYE], utils.GREEN, opacity=0.4)
            frame = utils.fillPolyTrans(frame, [mesh_coords[p] for p in LEFT_EYEBROW], utils.ORANGE, opacity=0.4)
            frame = utils.fillPolyTrans(frame, [mesh_coords[p] for p in RIGHT_EYEBROW], utils.ORANGE, opacity=0.4)
            frame = utils.fillPolyTrans(frame, [mesh_coords[p] for p in LIPS], utils.BLACK, opacity=0.3)
            # Changes points
            [cv2.circle(frame, mesh_coords[p], 1, utils.GREEN, -1, cv2.LINE_AA) for p in LIPS]
            [cv2.circle(frame, mesh_coords[p], 1, utils.BLACK, - 1, cv2.LINE_AA) for p in RIGHT_EYE]
            [cv2.circle(frame, mesh_coords[p], 1, utils.BLACK, -1, cv2.LINE_AA) for p in LEFT_EYE]

            [cv2.circle(frame, mesh_coords[p], 1, utils.BLACK, -1, cv2.LINE_AA) for p in RIGHT_EYEBROW]
            [cv2.circle(frame, mesh_coords[p], 1, utils.BLACK, -1, cv2.LINE_AA) for p in LEFT_EYEBROW]
            [cv2.circle(frame, mesh_coords[p], 1, utils.RED, -1, cv2.LINE_AA) for p in FACE_OVAL]

            img_h, img_w = frame.shape[:2]
            mesh_points = np.array(
                [np.multiply([p.x, p.y], [img_w, img_h]).astype(int) for p in results.multi_face_landmarks[0].landmark])
            # print(mesh_points.shape)
            cv2.polylines(frame, [mesh_points[LEFT_IRIS]], True, (0,255,0), 1, cv2.LINE_AA)
            cv2.polylines(frame, [mesh_points[RIGHT_IRIS]], True, (0,255,0), 1, cv2.LINE_AA)
            (l_cx, l_cy), l_radius = cv2.minEnclosingCircle(mesh_points[LEFT_IRIS])
            (r_cx, r_cy), r_radius = cv2.minEnclosingCircle(mesh_points[RIGHT_IRIS])
            center_left = np.array([l_cx, l_cy], dtype=np.int32)
            center_right = np.array([r_cx, r_cy], dtype=np.int32)
            cv2.circle(frame, center_left, int(l_radius), (255, 0, 255), 1, cv2.LINE_AA)
            cv2.circle(frame, center_right, int(r_radius), (255, 0, 255), 1, cv2.LINE_AA)

            # determine expression
            landmarks = ilg.landmarks_detection(frame, results, False)
            landmarks = landmarks[:468]  # Omit the iris landmarks
            bbox = ilg.get_bounding_box(landmarks)

            normalized_landmarks = ilg.normalize_landmarks(landmarks, ilg.reference_landmarks)
            expression = infer_model(normalized_landmarks)
        else:
            expression = 'None'
        # calculating  frame per seconds FPS
        end_time = time.time() - start_time
        fps = frame_counter / end_time

        frame = utils.textWithBackground(frame, f'FPS: {round(fps, 1)} EXPRESSION: {expression}', FONTS, 1.0, (20, 50), bgOpacity=0.9,
                                         textThickness=2)
        # writing image for thumbnail drawing shape
        # cv2.imwrite(f'img/frame_{frame_counter}.png', frame)
        cv2.imshow('frame', frame)
        key = cv2.waitKey(1)
        if key == ord('q') or key == ord('Q'):
            break
    cv2.destroyAllWindows()
    camera.release()
