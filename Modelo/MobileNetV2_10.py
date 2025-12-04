import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix

# Configurações

DATASET_DIR = "../Divisao"
RESULTS_DIR = "Resultados/MobileNetV2_10runs"
IMG_HEIGHT, IMG_WIDTH = 224, 224
BATCH_SIZE = 16
EPOCHS = 30
NUM_CLASSES = 7
RUNS = 10
os.makedirs(RESULTS_DIR, exist_ok=True)

# Função de Custo: Focal Loss

def focal_loss(gamma=2.0, alpha=0.25):
    def focal_loss_fixed(y_true, y_pred):
        epsilon = K.epsilon()
        y_pred = K.clip(y_pred, epsilon, 1.0 - epsilon)

        cross_entropy = -y_true * K.log(y_pred)
        p_t = y_true * y_pred + (1 - y_true) * (1 - y_pred)
        modulating_factor = K.pow(1.0 - p_t, gamma)
        alpha_factor = y_true * alpha + (1 - y_true) * (1 - alpha)
        loss = alpha_factor * modulating_factor * cross_entropy

        return K.sum(loss, axis=-1)

    return focal_loss_fixed

# Modelo MobileNetV2

def build_model(num_classes):
    base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(IMG_HEIGHT, IMG_WIDTH, 3))

    for layer in base_model.layers[:-20]:
        layer.trainable = False
    for layer in base_model.layers[-20:]:
        layer.trainable = True

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.3)(x)
    preds = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=preds)
    model.compile(
        optimizer=Adam(learning_rate=1e-4),
        loss=focal_loss(gamma=2.0, alpha=0.25),
        metrics=['accuracy']
    )
    return model

# Função de Pré-processamento

def process_dataset(image, label):
    image = tf.cast(image, tf.float32) / 255.0
    label = tf.one_hot(label, NUM_CLASSES)
    return image, label

# Carregamento de Dados

train_ds = tf.keras.utils.image_dataset_from_directory(
    os.path.join(DATASET_DIR, "train"),
    labels='inferred',
    label_mode='int',
    image_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    shuffle=True
)
val_ds = tf.keras.utils.image_dataset_from_directory(
    os.path.join(DATASET_DIR, "val"),
    labels='inferred',
    label_mode='int',
    image_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    shuffle=False
)
test_ds = tf.keras.utils.image_dataset_from_directory(
    os.path.join(DATASET_DIR, "test"),
    labels='inferred',
    label_mode='int',
    image_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=1,
    shuffle=False
)

train_ds_processed = train_ds.map(process_dataset).prefetch(tf.data.AUTOTUNE)
val_ds_processed = val_ds.map(process_dataset).prefetch(tf.data.AUTOTUNE)
test_ds_processed = test_ds.map(process_dataset)

class_names = train_ds.class_names
print(f"Classes: {class_names}")


train_labels_raw = np.concatenate([y.numpy() for x, y in train_ds.unbatch().as_numpy_iterator()], axis=0)

class_weights_values = compute_class_weight(
    class_weight='balanced',
    classes=np.arange(NUM_CLASSES),
    y=train_labels_raw
)
class_weights = dict(enumerate(class_weights_values))
print("Pesos de classe:", class_weights)

# Armazena os Resultados

all_acc = []
all_loss = []
all_f1 = []
all_precision = []
all_recall = []

classes_list = class_names

best_val_acc = 0
best_model_path = None

for run in range(1, RUNS + 1):
    print(f"\nRUN: {run}/{RUNS}")

    run_dir = os.path.join(RESULTS_DIR, f"run_{run}")
    os.makedirs(run_dir, exist_ok=True)

    model = build_model(NUM_CLASSES)

    checkpoint_path = os.path.join(run_dir, "mobilenetv2_best.weights.h5")
    checkpoint = ModelCheckpoint(checkpoint_path, monitor="val_accuracy", save_best_only=True, save_weights_only=True)
    reduce_lr = ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6)
    early_stop = EarlyStopping(monitor='val_loss', patience=7, restore_best_weights=True)

    history = model.fit(
        train_ds_processed,
        validation_data=val_ds_processed,
        epochs=EPOCHS,
        class_weight=class_weights,
        callbacks=[checkpoint, reduce_lr, early_stop],
        verbose=1
    )

    # Gráficos
    plt.plot(history.history['accuracy'], label='Train Acc')
    plt.plot(history.history['val_accuracy'], label='Val Acc')
    plt.legend();
    plt.grid()
    plt.savefig(os.path.join(run_dir, "accuracy_curve.png"))
    plt.close()

    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.legend();
    plt.grid()
    plt.savefig(os.path.join(run_dir, "loss_curve.png"))
    plt.close()

    # Avaliação
    model.load_weights(checkpoint_path)

    y_true_int = np.concatenate([y.numpy() for x, y in test_ds], axis=0)
    y_pred_prob = model.predict(test_ds_processed)
    y_pred = np.argmax(y_pred_prob, axis=1)

    report = classification_report(y_true_int, y_pred, target_names=classes_list, output_dict=True)
    with open(os.path.join(run_dir, "metrics.json"), "w") as f:
        json.dump(report, f, indent=4)

    all_acc.append(report["accuracy"])
    all_loss.append(min(history.history["val_loss"]))

    f1_vals, prec_vals, rec_vals = [], [], []
    for classe in classes_list:
        f1_vals.append(report[classe]['f1-score'])
        prec_vals.append(report[classe]['precision'])
        rec_vals.append(report[classe]['recall'])

    all_f1.append(f1_vals)
    all_precision.append(prec_vals)
    all_recall.append(rec_vals)

    # Matriz de Confusão
    cm = confusion_matrix(y_true_int, y_pred)
    cm_df = pd.DataFrame(cm, index=classes_list, columns=classes_list)
    sns.heatmap(cm_df, annot=True, fmt='d', cmap='Blues')
    plt.savefig(os.path.join(run_dir, "confusion_matrix.png"))
    plt.close()

    best_run_val_acc = max(history.history["val_accuracy"])
    if best_run_val_acc > best_val_acc:
        best_val_acc = best_run_val_acc
        best_model_path = checkpoint_path

    tf.keras.backend.clear_session()

all_f1 = np.array(all_f1)
all_precision = np.array(all_precision)
all_recall = np.array(all_recall)

summary = {
    "accuracy_mean": float(np.mean(all_acc)),
    "accuracy_std": float(np.std(all_acc)),
    "loss_mean": float(np.mean(all_loss)),
    "loss_std": float(np.std(all_loss)),
    "classes": {}
}

for idx, classe in enumerate(classes_list):
    summary["classes"][classe] = {
        "f1_mean": float(np.mean(all_f1[:, idx])),
        "f1_std": float(np.std(all_f1[:, idx])),
        "precision_mean": float(np.mean(all_precision[:, idx])),
        "precision_std": float(np.std(all_precision[:, idx])),
        "recall_mean": float(np.mean(all_recall[:, idx])),
        "recall_std": float(np.std(all_recall[:, idx]))
    }

summary["best_model"] = best_model_path

with open(os.path.join(RESULTS_DIR, "summary_metrics.json"), "w") as f:
    json.dump(summary, f, indent=4)

# Gráficos

acc_mean = summary["accuracy_mean"]
acc_std = summary["accuracy_std"]
loss_mean = summary["loss_mean"]
loss_std = summary["loss_std"]

plt.figure(figsize=(6, 5))
plt.bar(["Accuracy"], [acc_mean], yerr=[acc_std], capsize=8)
plt.title("Accuracy Média das 10 Runs")
plt.ylabel("Accuracy")
plt.ylim(0, 1)
plt.grid(axis="y")
plt.savefig(os.path.join(RESULTS_DIR, "accuracy_mean.png"))
plt.close()

plt.figure(figsize=(6, 5))
plt.bar(["Loss"], [loss_mean], yerr=[loss_std], capsize=8)
plt.title("Loss Média das 10 Runs")
plt.ylabel("Loss")
plt.grid(axis="y")
plt.savefig(os.path.join(RESULTS_DIR, "loss_mean.png"))
plt.close()

# Matriz de Confusão

print("\nGerando matriz de confusão da melhor run...")
best_run_dir = os.path.dirname(best_model_path)


def build_model_for_loading(num_classes=NUM_CLASSES):
    base_model = MobileNetV2(weights=None, include_top=False, input_shape=(IMG_HEIGHT, IMG_WIDTH, 3))
    x = GlobalAveragePooling2D()(base_model.output)
    x = Dropout(0.3)(x)
    preds = Dense(num_classes, activation='softmax')(x)
    return Model(inputs=base_model.input, outputs=preds)


model_best = build_model_for_loading()
model_best.load_weights(best_model_path)

test_ds_conf = tf.keras.utils.image_dataset_from_directory(
    os.path.join(DATASET_DIR, "test"),
    labels='inferred',
    label_mode='int',
    image_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=1,
    shuffle=False
)
test_ds_conf_processed = test_ds_conf.map(lambda image, label: (tf.cast(image, tf.float32) / 255.0, label))

y_true_int = np.concatenate([y.numpy() for x, y in test_ds_conf], axis=0)
y_pred_prob = model_best.predict(test_ds_conf_processed)
y_pred = np.argmax(y_pred_prob, axis=1)

cm = confusion_matrix(y_true_int, y_pred)
cm_df = pd.DataFrame(cm, index=classes_list, columns=classes_list)

plt.figure(figsize=(10, 8))
sns.heatmap(cm_df, annot=True, fmt='d', cmap='Blues')
plt.title("Matriz de Confusão – Melhor Run")
plt.ylabel("Classe Real")
plt.xlabel("Classe Predita")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(os.path.join(best_run_dir, "confusion_matrix_best_run.png"))
plt.close()

print("Treino Concluido.")