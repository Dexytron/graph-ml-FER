import cv2
import mediapipe as mp
import numpy as np
import plotly.graph_objects as go
from scipy.spatial import procrustes
import scripts

mp_face_mesh = mp.solutions.face_mesh


def landmarks_detection(img, results, draw=False, marker_color=(0, 255, 0), marker_size=1):
    """
    Function to detect landmarks on the face
    :param img: cv2 image object
    :param results: face_mesh results object
    :param draw: boolean to draw landmarks on the image
    :return: list of face mesh coordinates
    """
    img_height, img_width = img.shape[:2]
    # List of (x, y, z) tuples
    mesh_coords = [(point.x * img_width, point.y * img_height, point.z * img_width) for point in
                   results.multi_face_landmarks[0].landmark]
    if draw:
        [cv2.drawMarker(img, (int(p[0]), int(p[1])), marker_color, markerType=cv2.MARKER_STAR, markerSize=marker_size) for p in mesh_coords]
    return mesh_coords


def get_bounding_box(mesh_coords):
    """
    Function to get the bounding box of the face
    :param mesh_coords: list of face mesh coordinates
    :return: x_min, y_min, z_min, x_max, y_max, z_max
    """
    x_coords = [p[0] for p in mesh_coords]
    y_coords = [p[1] for p in mesh_coords]
    z_coords = [p[2] for p in mesh_coords]
    x_min, x_max = min(x_coords), max(x_coords)
    y_min, y_max = min(y_coords), max(y_coords)
    z_min, z_max = min(z_coords), max(z_coords)
    return x_min, y_min, z_min, x_max, y_max, z_max


def center_landmarks(mesh_coords):
    """
    Center the landmarks to the origin
    :param mesh_coords: list of face mesh coordinates
    :return: list of centered landmarks
    """
    centroid = np.mean(mesh_coords, axis=0)
    centered_coords = mesh_coords - centroid
    return centered_coords


def scale_landmarks_0_1(centered_coords):
    """
    Scale the landmarks to fit within a range of 0 to 1
    :param centered_coords: list of centered landmarks
    :return: list of scaled landmarks
    """
    min_coords = np.min(centered_coords, axis=0)
    max_coords = np.max(centered_coords, axis=0)
    scaled_coords = (centered_coords - min_coords) / (max_coords - min_coords)
    return scaled_coords


def align_landmarks_procrustes(mesh_coords, reference_landmarks):
    """
    Align the landmarks using Procrustes analysis
    :param mesh_coords: list of face mesh coordinates
    :param reference_landmarks: list of reference face mesh coordinates
    :return: list of aligned landmarks
    """
    _, aligned_coords, _ = procrustes(reference_landmarks, mesh_coords)
    return aligned_coords


def normalize_landmarks(mesh_coords, reference_landmarks):
    """
    Function to normalize landmarks
    :param mesh_coords: list of face mesh coordinates
    :param reference_landmarks: list of reference face mesh coordinates
    :return: list of normalized landmarks
    """
    if reference_landmarks is None:
        return mesh_coords
    centered_coords = center_landmarks(mesh_coords)
    aligned_coords = align_landmarks_procrustes(centered_coords, reference_landmarks)
    normalized_coords = scale_landmarks_0_1(aligned_coords)
    normalized_coords = normalized_coords.astype(np.float16)
    return normalized_coords


def plot_landmarks(normalized_coords):
    """
    Function create interactive 3D plot for normalized landmarks
    :param normalized_coords: list of normalized landmarks
    :return: None
    """
    x_vals = normalized_coords[:, 0]
    y_vals = normalized_coords[:, 1]
    z_vals = normalized_coords[:, 2]

    # Create interactive 3D plot with Plotly
    fig = go.Figure(data=[go.Scatter3d(
        x=x_vals, y=y_vals, z=z_vals,
        mode='markers',
        marker=dict(size=2, color='red')
    )])
    fig.update_layout(scene=dict(xaxis_title='X Axis', yaxis_title='Y Axis', zaxis_title='Z Axis'))
    fig.show()


def get_landmarks(image, reference_landmarks, review=False):
    """
    Function to get the landmarks of the face in the image
    :param image: cv2 image object
    :param reference_landmarks: list of reference face mesh coordinates
    :param review: whether to display the image with the detected face
    :return: bbox, normalized landmark coordinates
    """
    assert image is not None, "Image not found"
    with mp_face_mesh.FaceMesh(static_image_mode=True, refine_landmarks=True, min_detection_confidence=0.75) as face_mesh:
        if len(image.shape) == 2:  # Interpret bw images as rgb
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

        results = face_mesh.process(image)

        if results.multi_face_landmarks:
            landmarks = landmarks_detection(image, results, True)
            landmarks = landmarks[:468]  # Omit the iris landmarks
            bbox = get_bounding_box(landmarks)

            normalized_landmarks = normalize_landmarks(landmarks, reference_landmarks)
            if review:
                x_min, y_min, z_min, x_max, y_max, z_max = bbox
                cropped_face = image[int(y_min):int(y_max), int(x_min):int(x_max)]
                scale_factor = 5  # Change this factor to make the window bigger or smaller
                big_cropped_face = cv2.resize(cropped_face, (0, 0), fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_NEAREST)
                cv2.imshow('FACE DETECTED', big_cropped_face)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
            return bbox, normalized_landmarks
        else:
            if review:
                cv2.imshow('NO FACE DETECTED', image)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
            return None, None

reference_image = cv2.imread(scripts.__path__[0] + "/reference_image.jpeg")
_, reference_landmarks = get_landmarks(reference_image, reference_landmarks=None)

if __name__ == "__main__":
    IMAGE_PATHS = ["reference_image.jpeg"]
    for image_path in IMAGE_PATHS:
        image = cv2.imread(image_path)
        bbox, landmarks = get_landmarks(image, reference_landmarks, review=True)
        if bbox is None:
            print("NO FACE DETECTED: " + image_path)
        else:
            # plot_landmarks(landmarks)
            pass
