import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.applications.densenet import preprocess_input as densenet_preprocess
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet_preprocess
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as mobilenet_preprocess
import cv2
import matplotlib.pyplot as plt

MODEL_CHOICE = "densenet"  # 'densenet', 'resnet', 'mobilenet'
IMG_PATH = "Imagens_Teste/papel.png"
IMG_HEIGHT, IMG_WIDTH = 224, 224
NUM_CLASSES = 7
CLASS_LABELS_PATH = "classes.txt"

MODEL_WEIGHTS = {
    "densenet": "../Modelo/Resultados/DenseNet121/densenet121.weights.h5",
    "resnet": "../Modelo/Resultados/Resnet50_2/resnet50.weights.h5",
    "mobilenet": "../Modelo/Resultados/MobileNetV2/mobilenetv2.weights.h5"
}

if MODEL_CHOICE == "densenet":
    from tensorflow.keras.applications import DenseNet121
    base_model = DenseNet121(weights=None, include_top=False, input_shape=(IMG_HEIGHT, IMG_WIDTH, 3))
    last_conv_layer_name = "conv5_block16_2_conv"
elif MODEL_CHOICE == "resnet":
    from tensorflow.keras.applications import ResNet50
    base_model = ResNet50(weights=None, include_top=False, input_shape=(IMG_HEIGHT, IMG_WIDTH, 3))
    last_conv_layer_name = "conv5_block3_out"
elif MODEL_CHOICE == "mobilenet":
    from tensorflow.keras.applications import MobileNetV2
    base_model = MobileNetV2(weights=None, include_top=False, input_shape=(IMG_HEIGHT, IMG_WIDTH, 3))
    last_conv_layer_name = "Conv_1"

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.3)(x)
preds = Dense(NUM_CLASSES, activation='softmax')(x)
model = Model(inputs=base_model.input, outputs=preds)

model.load_weights(MODEL_WEIGHTS[MODEL_CHOICE])

with open(CLASS_LABELS_PATH, "r", encoding="utf-8") as f:
    classes = [line.strip() for line in f.readlines() if line.strip()]
print("Classes carregadas:", classes)

def preprocess_image(img_path, model_choice):
    img = image.load_img(img_path, target_size=(IMG_HEIGHT, IMG_WIDTH))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)

    if model_choice == 'densenet':
        img_array = densenet_preprocess(img_array)
    elif model_choice == 'resnet':
        img_array = resnet_preprocess(img_array)
    elif model_choice == 'mobilenet':
        img_array = mobilenet_preprocess(img_array)

    return img_array

img_array = preprocess_image(IMG_PATH, MODEL_CHOICE)

predictions = model.predict(img_array)
pred_idx = np.argmax(predictions, axis=1)[0]
pred_class = classes[pred_idx]
confidence = predictions[0][pred_idx]

print(f"Imagem: {IMG_PATH}")
print(f"Classe prevista: {pred_class}")
print(f"Confiança: {confidence*100:.2f}%")

def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    grad_model = Model([model.inputs],
                       [model.get_layer(last_conv_layer_name).output, model.output])
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]

    grads = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

heatmap = make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=pred_idx)

img = cv2.imread(IMG_PATH)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
heatmap = np.clip(heatmap, 0, 1)
heatmap_uint8 = np.uint8(255 * heatmap)
heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)
alpha = 0.4
superimposed_img = cv2.addWeighted(img, 1 - alpha, heatmap_color, alpha, 0)

plt.figure(figsize=(10, 5))
plt.suptitle(f"Classe prevista: {pred_class} ({confidence*100:.2f}%)", fontsize=14)

plt.subplot(1, 2, 1)
plt.title("Imagem Original")
plt.imshow(img)
plt.axis("off")

plt.subplot(1, 2, 2)
plt.title("Grad-CAM")
plt.imshow(superimposed_img)
plt.axis("off")

plt.tight_layout()
plt.show()
