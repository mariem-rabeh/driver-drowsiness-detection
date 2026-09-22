import os
import random
import shutil

def sample_images(source_dir, dest_dir, n_samples=1000, seed=42):
    random.seed(seed)
    os.makedirs(dest_dir, exist_ok=True)
    
    all_files = [f for f in os.listdir(source_dir) 
                 if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
    
    sampled = random.sample(all_files, min(n_samples, len(all_files)))
    
    for f in sampled:
        shutil.copy(os.path.join(source_dir, f), os.path.join(dest_dir, f))
    
    print(f"{len(sampled)} images copiees : {source_dir} -> {dest_dir}")


if __name__ == "__main__":
    BASE_SRC = r"C:\Users\maria\Desktop\archive"
    BASE_DEST = "data/mrl_sample"

    splits = ["train_data", "validation_data", "test_data"]
    classes = ["drowsy", "notdrowsy"]

    n_per_split = {"train_data": 1500, "validation_data": 400, "test_data": 400}

    for split in splits:
        for cls in classes:
            src = os.path.join(BASE_SRC, split, cls)
            dest = os.path.join(BASE_DEST, split, cls)
            if os.path.exists(src):
                sample_images(src, dest, n_samples=n_per_split[split])
            else:
                print(f"ATTENTION: dossier introuvable -> {src}")