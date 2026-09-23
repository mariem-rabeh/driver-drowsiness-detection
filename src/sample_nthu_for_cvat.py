import os
import random
import shutil

def sample_for_cvat(source_dir, dest_dir, n_samples=250, prefix="", seed=42):
    random.seed(seed)
    os.makedirs(dest_dir, exist_ok=True)

    all_files = [f for f in os.listdir(source_dir)
                 if f.lower().endswith(('.jpg', '.png', '.jpeg'))]

    sampled = random.sample(all_files, min(n_samples, len(all_files)))

    for f in sampled:
        new_name = f"{prefix}_{f}" if prefix else f
        shutil.copy(os.path.join(source_dir, f), os.path.join(dest_dir, new_name))

    print(f"{len(sampled)} images copiees : {source_dir} -> {dest_dir}")


if __name__ == "__main__":
    BASE_SRC = r"C:\Users\maria\Desktop\archive\train_data"
    DEST = "data/frames"

    # On melange les deux classes car dans CVAT on annote l'etat des yeux,
    # pas la classe drowsy/notdrowsy globale
    sample_for_cvat(os.path.join(BASE_SRC, "drowsy"), DEST, n_samples=250, prefix="drowsy")
    sample_for_cvat(os.path.join(BASE_SRC, "notdrowsy"), DEST, n_samples=250, prefix="notdrowsy")