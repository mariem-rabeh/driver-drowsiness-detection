import os
import cv2
import numpy as np
import mediapipe as mp
import pickle
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from ear_utils import get_average_ear

mp_face_mesh = mp.solutions.face_mesh


def extract_ear_features(base_dir, split, label):
    """Extrait l'EAR pour chaque image d'un dossier drowsy/notdrowsy."""
    folder = os.path.join(base_dir, split, label)
    features = []

    with mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1,
                                 refine_landmarks=True, min_detection_confidence=0.5) as face_mesh:
        for filename in os.listdir(folder):
            if not filename.lower().endswith((".jpg", ".png", ".jpeg")):
                continue

            image = cv2.imread(os.path.join(folder, filename))
            if image is None:
                continue

            h, w = image.shape[:2]
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb_image)

            if not results.multi_face_landmarks:
                continue

            landmarks = results.multi_face_landmarks[0].landmark
            ear = get_average_ear(landmarks, w, h)
            features.append(ear)

    return features


def build_dataset(base_dir, split):
    drowsy_ears = extract_ear_features(base_dir, split, "drowsy")
    notdrowsy_ears = extract_ear_features(base_dir, split, "notdrowsy")

    X = np.array(drowsy_ears + notdrowsy_ears).reshape(-1, 1)
    y = np.array([1] * len(drowsy_ears) + [0] * len(notdrowsy_ears))  # 1=drowsy, 0=notdrowsy

    print(f"{split}: {len(drowsy_ears)} drowsy, {len(notdrowsy_ears)} notdrowsy")
    return X, y


def main():
    BASE_DIR = "data/mrl_sample"

    print("Extraction des features (train)...")
    X_train, y_train = build_dataset(BASE_DIR, "train_data")

    print("Extraction des features (test)...")
    X_test, y_test = build_dataset(BASE_DIR, "test_data")

    print("\nEntrainement du classifieur SVM...")
    clf = SVC(kernel="linear", probability=True)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)

    print("\n=== Resultats SVM (feature EAR) ===")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(classification_report(y_test, y_pred, target_names=["notdrowsy", "drowsy"]))

    # Comparaison avec le seuil fixe calibre precedemment
    FIXED_THRESHOLD = 0.2231
    y_pred_threshold = (X_test.flatten() < FIXED_THRESHOLD).astype(int)

    print("\n=== Resultats seuil fixe (0.2231) ===")
    print(f"Accuracy: {accuracy_score(y_test, y_pred_threshold):.4f}")
    print(classification_report(y_test, y_pred_threshold, target_names=["notdrowsy", "drowsy"]))

    # Sauvegarde du modele
    os.makedirs("models", exist_ok=True)
    with open("models/svm_ear_classifier.pkl", "wb") as f:
        pickle.dump(clf, f)
    print("\nModele sauvegarde: models/svm_ear_classifier.pkl")


if __name__ == "__main__":
    main()