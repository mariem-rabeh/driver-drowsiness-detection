import os
import cv2
import mediapipe as mp
from ear_utils import get_average_ear

mp_face_mesh = mp.solutions.face_mesh

def get_label_for_frame(annotation_path):
    """Lit le fichier YOLO et retourne 'open', 'closed', ou None si vide/mixte."""
    if not os.path.exists(annotation_path) or os.path.getsize(annotation_path) == 0:
        return None

    labels_found = set()
    with open(annotation_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if parts:
                class_id = int(parts[0])
                labels_found.add(class_id)  # 0 = eye_open, 1 = eye_closed (verifie l'ordre reel)

    if len(labels_found) == 1:
        return "open" if 0 in labels_found else "closed"
    return "mixed"  # les deux yeux ont des labels differents, cas ambigu a ignorer


def main():
    frames_dir = "data/frames"
    annotations_dir = r"data\annotations\nthu_batch_1\task_2614183_annotations_2026_09_22_23_16_50_yolo 1.1\obj_train_data"

    open_ears = []
    closed_ears = []

    with mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, 
                                 refine_landmarks=True, min_detection_confidence=0.5) as face_mesh:
        
        for frame_file in os.listdir(frames_dir):
            if not frame_file.lower().endswith((".jpg", ".png", ".jpeg")):
                continue

            frame_name = os.path.splitext(frame_file)[0]
            annotation_path = os.path.join(annotations_dir, frame_name + ".txt")

            label = get_label_for_frame(annotation_path)
            if label not in ("open", "closed"):
                continue  # skip frames vides ou mixtes

            image = cv2.imread(os.path.join(frames_dir, frame_file))
            if image is None:
                continue

            h, w = image.shape[:2]
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb_image)

            if not results.multi_face_landmarks:
                continue

            landmarks = results.multi_face_landmarks[0].landmark
            ear = get_average_ear(landmarks, w, h)

            if label == "open":
                open_ears.append(ear)
            else:
                closed_ears.append(ear)

    print(f"Echantillons 'open' avec visage detecte : {len(open_ears)}")
    print(f"Echantillons 'closed' avec visage detecte : {len(closed_ears)}")

    if open_ears and closed_ears:
        avg_open = sum(open_ears) / len(open_ears)
        avg_closed = sum(closed_ears) / len(closed_ears)
        suggested_threshold = (avg_open + avg_closed) / 2

        print(f"\nEAR moyen (open)   : {avg_open:.4f}")
        print(f"EAR moyen (closed) : {avg_closed:.4f}")
        print(f"Seuil suggere      : {suggested_threshold:.4f}")


if __name__ == "__main__":
    main()