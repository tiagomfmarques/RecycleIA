import os
import math
import cv2
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import shutil

dataset_dir = "Dataset_Container"
aug_dir = "Dataset_Container_augmentation"

if not os.path.exists(aug_dir):
    shutil.copytree(dataset_dir, aug_dir, dirs_exist_ok=True)

classes = [d for d in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, d))]
max_images = max([len(os.listdir(os.path.join(dataset_dir, c))) for c in classes])


datagen_geom = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    fill_mode='nearest'
)

def pixel_augment(img):
    # Brilho e contraste
    alpha = np.random.uniform(0.9, 1.1)  # contraste suave
    beta = np.random.randint(-10, 10)    # brilho suave
    img = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)

    # Blur
    if np.random.rand() < 0.3:
        ksize = np.random.choice([3])
        img = cv2.GaussianBlur(img, (ksize, ksize), 0)

    # Grayscale
    if np.random.rand() < 0.2:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    return img

for classe in classes:
    src_path = os.path.join(dataset_dir, classe)
    dst_path = os.path.join(aug_dir, classe)
    images = [f for f in os.listdir(src_path) if f.lower().endswith(('.png','.jpg','.jpeg','.bmp'))]
    n_imgs = len(images)

    fator = math.ceil(math.sqrt(max_images / n_imgs))
    if fator <= 1:
        print(f"Classe: {classe}, Nº original: {n_imgs}, Fator: {fator} → sem aumento")
        continue
    print(f"Classe: {classe}, Nº original: {n_imgs}, Fator: {fator} → aumentando ~{fator}x")

    for img_name in images:
        img_path = os.path.join(src_path, img_name)
        img = cv2.imread(img_path)

        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        elif img.shape[2] != 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        num_to_generate = fator - 1
        n_geom = int(num_to_generate * 0.5)
        n_pixel = num_to_generate - n_geom

        x = np.expand_dims(img, axis=0)
        for i, batch in enumerate(datagen_geom.flow(x, batch_size=1, save_to_dir=dst_path, save_prefix='aug', save_format='jpg')):
            if i >= n_geom:
                break

        for i in range(n_pixel):
            img_pix = pixel_augment(img.copy())
            save_name = f"aug_{i}_{img_name}"
            cv2.imwrite(os.path.join(dst_path, save_name), img_pix)
