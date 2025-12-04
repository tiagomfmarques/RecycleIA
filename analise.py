import os

dataset_dir = "Dataset_Container_augmentation" # "Dataset_Container"

classes = [d for d in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, d))]

num_imagens = {}

for classe in classes:
    class_path = os.path.join(dataset_dir, classe)
    imagens = [f for f in os.listdir(class_path) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
    num_imagens[classe] = len(imagens)

# Número de Imagens por classe

for classe, count in num_imagens.items():
    print(f"Classe: {classe} | Nº de imagens: {count}")
