import os
import cv2
from sklearn.model_selection import train_test_split

aug_dir = "Dataset_Container_augmentation"

dataset_dir = "Divisao"

IMG_HEIGHT = 224
IMG_WIDTH = 224

for split in ["train", "val", "test"]:
    os.makedirs(os.path.join(dataset_dir, split), exist_ok=True)

classes = [d for d in os.listdir(aug_dir) if os.path.isdir(os.path.join(aug_dir, d))]

for classe in classes:
    class_path = os.path.join(aug_dir, classe)

    files = os.listdir(class_path)

    real_imgs = [f for f in files if not f.lower().startswith("aug")]
    gen_imgs  = [f for f in files if f.lower().startswith("aug")]

    print(f"Classe: {classe} | Reais: {len(real_imgs)} | Geradas: {len(gen_imgs)}")

    train_real, temp = train_test_split(real_imgs, test_size=0.60, random_state=42, shuffle=True)
    val_real, test_real = train_test_split(temp, test_size=0.5, random_state=42)

    train_final = train_real + gen_imgs
    val_final   = val_real
    test_final  = test_real

    splits = {
        "train": train_final,
        "val": val_final,
        "test": test_final
    }

    for split_name, split_files in splits.items():
        out_dir = os.path.join(dataset_dir, split_name, classe)
        os.makedirs(out_dir, exist_ok=True)

        for img_name in split_files:
            src = os.path.join(class_path, img_name)
            dst = os.path.join(out_dir, img_name)

            img = cv2.imread(src)

            img_resized = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
            cv2.imwrite(dst, img_resized)

print("Divisão concluída.")
