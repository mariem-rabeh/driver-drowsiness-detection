import os
from pathlib import Path
from collections import Counter

def parse_yolo_annotations(annotations_dir, obj_names_path):
    # Charger les noms de classes dans l'ordre
    with open(obj_names_path, "r") as f:
        class_names = [line.strip() for line in f if line.strip()]

    counts = Counter()
    empty_files = 0
    total_files = 0

    for filename in os.listdir(annotations_dir):
        if not filename.endswith(".txt"):
            continue
        total_files += 1
        filepath = os.path.join(annotations_dir, filename)

        if os.path.getsize(filepath) == 0:
            empty_files += 1
            continue

        with open(filepath, "r") as f:
            for line in f:
                parts = line.strip().split()
                if not parts:
                    continue
                class_id = int(parts[0])
                class_name = class_names[class_id]
                counts[class_name] += 1

    print(f"Total fichiers : {total_files}")
    print(f"Fichiers vides (aucun oeil annote) : {empty_files}")
    print(f"Fichiers annotes : {total_files - empty_files}")
    print("\nRepartition des labels (bounding boxes) :")
    for name, count in counts.items():
        print(f"  {name} : {count}")

    return counts


if __name__ == "__main__":
    ANNOTATIONS_DIR = r"data\annotations\nthu_batch_1\task_2614183_annotations_2026_09_22_23_16_50_yolo 1.1\obj_train_data"
    OBJ_NAMES = r"data\annotations\nthu_batch_1\task_2614183_annotations_2026_09_22_23_16_50_yolo 1.1\obj.names"

    parse_yolo_annotations(ANNOTATIONS_DIR, OBJ_NAMES)