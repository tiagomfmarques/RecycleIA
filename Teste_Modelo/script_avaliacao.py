import numpy as np
import pandas as pd
import os
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.preprocessing import image

BASE_DIR = "Dataset_Container"
OUTPUT_CSV = "scores_simulados.csv"
NUM_CLASSES = 7
MODEL_WEIGHTS_PATH = "modelo_contentor.weights.h5"
IMG_HEIGHT, IMG_WIDTH = 224, 224

CLASS_NAMES = [
    "container_battery",
    "container_biodegradable",
    "container_blue",
    "container_default",
    "container_green",
    "container_oil",
    "container_yellow",
]

global_model = None


def build_model(input_shape=(IMG_HEIGHT, IMG_WIDTH, 3), num_classes=NUM_CLASSES):
    base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=input_shape)
    for layer in base_model.layers:
        layer.trainable = False
    for layer in base_model.layers[-50:]:
        layer.trainable = True

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.3)(x)
    preds = Dense(num_classes, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=preds)

    return model


def load_and_preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(IMG_HEIGHT, IMG_WIDTH))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0
    return img_array


def classifier_score(img_path):
    global global_model

    if global_model is None:
        try:
            global_model = build_model()

            focal_loss_fn = tf.keras.losses.CategoricalFocalCrossentropy(
                gamma=2.0,
                alpha=0.25
            )
            global_model.compile(optimizer='adam', loss=focal_loss_fn)

            global_model.load_weights(MODEL_WEIGHTS_PATH)
            print("Modelo Carregado")
        except Exception as e:
            print(f"ERRO ao carregar o modelo ou pesos: {e}")
            raise

    processed_image = load_and_preprocess_image(img_path)
    scores = global_model.predict(processed_image, verbose=0)[0]

    return scores


def run_test():
    all_image_data = []

    for root, dirs, files in os.walk(BASE_DIR):
        if root == BASE_DIR:
            continue

        true_class_name = os.path.basename(root)

        for filename in files:
            if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                img_path = os.path.join(root, filename)
                all_image_data.append((img_path, true_class_name))

    if not all_image_data:
        print(f"Nenhuma imagem encontrada em '{BASE_DIR}'. Verifique o path e a estrutura de pastas.")
        return

    print(f"Imagens: {len(all_image_data)}")

    results_list = []

    for path, true_class in all_image_data:
        scores = classifier_score(path)

        predicted_index = np.argmax(scores)
        confidence = scores[predicted_index]
        predicted_class_name = CLASS_NAMES[predicted_index]

        confidence_percent = f"{confidence * 100:.2f}%"

        nome_imagem = os.path.basename(path)

        row = {
            "Imagem": nome_imagem,
            "Classe Verdadeira": true_class,
            "Classe Prevista": predicted_class_name,
            "Confiança": confidence_percent
        }

        results_list.append(row)

    df_results = pd.DataFrame(results_list)
    df_results.to_csv(OUTPUT_CSV, index=False)

    print("\nFinalizado")


if __name__ == '__main__':

    if not os.path.isdir(BASE_DIR):
        print(f"A pasta de teste '{BASE_DIR}' não foi encontrada.")
    elif not os.path.exists(MODEL_WEIGHTS_PATH):
        print(f"O ficheiro de pesos do modelo '{MODEL_WEIGHTS_PATH}' não foi encontrado.")
    else:
        run_test()