import numpy as np

# Indices des landmarks MediaPipe Face Mesh pour les yeux
# (6 points par oeil, format standard EAR)
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]


def euclidean_dist(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))


def calculate_ear(landmarks, eye_indices, image_w, image_h):
    """
    Calcule l'Eye Aspect Ratio pour un oeil donne.
    landmarks: liste des landmarks MediaPipe (normalises 0-1)
    eye_indices: indices des 6 points de l'oeil
    """
    points = []
    for idx in eye_indices:
        lm = landmarks[idx]
        points.append((lm.x * image_w, lm.y * image_h))

    # EAR = (|p2-p6| + |p3-p5|) / (2 * |p1-p4|)
    vertical_1 = euclidean_dist(points[1], points[5])
    vertical_2 = euclidean_dist(points[2], points[4])
    horizontal = euclidean_dist(points[0], points[3])

    ear = (vertical_1 + vertical_2) / (2.0 * horizontal)
    return ear


def get_average_ear(landmarks, image_w, image_h):
    left_ear = calculate_ear(landmarks, LEFT_EYE, image_w, image_h)
    right_ear = calculate_ear(landmarks, RIGHT_EYE, image_w, image_h)
    return (left_ear + right_ear) / 2.0