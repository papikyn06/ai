import os
import warnings

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator

warnings.filterwarnings("ignore")

# ----------------------------
# CONFIG
# ----------------------------
DATASET_PATH = r"C:\Users\Сергей\.cache\kagglehub\datasets\meowmeowmeowmeowmeow\gtsrb-german-traffic-sign\versions\1"
TRAIN_PATH = os.path.join(DATASET_PATH, "Train")
TEST_PATH = os.path.join(DATASET_PATH, "Test")

IMG_SIZE = 48
NUM_CLASSES = 43
SEED = 42
BATCH_SIZE = 64
EPOCHS = 40


def load_images_train_style(path: str, img_size: int = 48):
    images, labels = [], []

    if not os.path.exists(path):
        raise FileNotFoundError(f"Train path not found: {path}")

    for class_dir in sorted(os.listdir(path)):
        class_path = os.path.join(path, class_dir)
        if not os.path.isdir(class_path):
            continue
        if not class_dir.isdigit():
            continue

        class_num = int(class_dir)

        for img_file in os.listdir(class_path):
            if not img_file.lower().endswith((".ppm", ".png", ".jpg", ".jpeg")):
                continue
            img_path = os.path.join(class_path, img_file)

            try:
                img = Image.open(img_path).convert("RGB").resize((img_size, img_size), Image.BILINEAR)
                arr = np.asarray(img, dtype=np.float32) / 255.0
                images.append(arr)
                labels.append(class_num)
            except:
                continue

    X = np.asarray(images, dtype=np.float32)
    y = np.asarray(labels, dtype=np.int32)
    return X, y


print("Загрузка датасета GTSRB...")
print(f"DATASET_PATH: {DATASET_PATH}")
print(f"TRAIN_PATH:   {TRAIN_PATH}")
if os.path.exists(TEST_PATH):
    print(f"TEST sample:  {os.listdir(TEST_PATH)[:5]}")

X_all, y_all = load_images_train_style(TRAIN_PATH, IMG_SIZE)
print(f"Всего Train: {X_all.shape}, классов: {len(np.unique(y_all))}, y_min={y_all.min()}, y_max={y_all.max()}")
print("X stats:", float(X_all.min()), float(X_all.max()), float(X_all.mean()))


X_train, X_test, y_train, y_test = train_test_split(
    X_all, y_all, test_size=0.15, random_state=SEED, stratify=y_all
)
X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train, test_size=0.15, random_state=SEED, stratify=y_train
)
print(f"Train: {X_train.shape} | Val: {X_val.shape} | Test: {X_test.shape}")


train_datagen = ImageDataGenerator(
    rotation_range=10,
    width_shift_range=0.08,
    height_shift_range=0.08,
    zoom_range=0.10,
    fill_mode="nearest"
)


model = models.Sequential([
    layers.Conv2D(32, (3, 3), padding="same", activation="relu", input_shape=(IMG_SIZE, IMG_SIZE, 3)),
    layers.BatchNormalization(),
    layers.Conv2D(32, (3, 3), padding="same", activation="relu"),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.15),

    layers.Conv2D(64, (3, 3), padding="same", activation="relu"),
    layers.BatchNormalization(),
    layers.Conv2D(64, (3, 3), padding="same", activation="relu"),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.20),

    layers.Conv2D(128, (3, 3), padding="same", activation="relu"),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.25),

    layers.Flatten(),
    layers.Dense(256, activation="relu"),
    layers.Dropout(0.40),
    layers.Dense(NUM_CLASSES, activation="softmax"),
])

model.compile(
    optimizer=Adam(learning_rate=1e-3),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

model.summary()


callbacks = [
    ModelCheckpoint("best_traffic_signs_model.keras", monitor="val_accuracy", save_best_only=True, verbose=1),
    ReduceLROnPlateau(monitor="val_loss", factor=0.2, patience=3, min_lr=1e-6, verbose=1),
    EarlyStopping(monitor="val_accuracy", patience=8, restore_best_weights=True, verbose=1),
]


history = model.fit(
    train_datagen.flow(X_train, y_train, batch_size=BATCH_SIZE, shuffle=True),
    epochs=EPOCHS,
    validation_data=(X_val, y_val),
    callbacks=callbacks,
    verbose=1,
)


test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
error_pct = (1.0 - test_acc) * 100.0

print("\n" + "=" * 70)
print(f"TEST ACCURACY : {test_acc * 100:.2f}%")
print(f"TEST ERROR    : {error_pct:.2f}%  (100% - accuracy)")
print(f"TEST LOSS     : {test_loss:.4f}")
print("=" * 70 + "\n")


with open("metrics_summary.txt", "w", encoding="utf-8") as f:
    f.write(f"TEST ACCURACY : {test_acc * 100:.2f}%\n")
    f.write(f"TEST ERROR    : {error_pct:.2f}%\n")
    f.write(f"TEST LOSS     : {test_loss:.4f}\n")


fig, axes = plt.subplots(1, 2, figsize=(14, 4))

axes[0].plot(history.history["accuracy"], label="Train")
axes[0].plot(history.history["val_accuracy"], label="Val")
axes[0].set_title("Accuracy")
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Accuracy")
axes[0].legend()
axes[0].grid(True)

axes[1].plot(history.history["loss"], label="Train")
axes[1].plot(history.history["val_loss"], label="Val")
axes[1].set_title("Loss")
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("Loss")
axes[1].legend()
axes[1].grid(True)

plt.tight_layout()
plt.savefig("training_history.png", dpi=200)
plt.show()


y_prob = model.predict(X_test, verbose=0)
y_pred = np.argmax(y_prob, axis=1)

cm = confusion_matrix(y_test, y_pred, labels=list(range(NUM_CLASSES)))


plt.figure(figsize=(10, 9))
plt.imshow(cm, interpolation="nearest")
plt.title("Confusion Matrix (GTSRB)")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.colorbar()
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=200)
plt.show()


per_class_total = cm.sum(axis=1)
per_class_correct = np.diag(cm)
per_class_acc = np.divide(per_class_correct, np.maximum(per_class_total, 1))
per_class_err = 1.0 - per_class_acc


top_k = 10
worst = np.argsort(per_class_err)[-top_k:][::-1]

print(f"ТОП-{top_k} классов с наибольшей ошибкой (по тесту):")
for c in worst:
    print(f"Class {c:2d}: acc={per_class_acc[c]*100:6.2f}% | err={per_class_err[c]*100:6.2f}% | samples={per_class_total[c]}")


report = classification_report(y_test, y_pred, digits=4)
with open("classification_report.txt", "w", encoding="utf-8") as f:
    f.write(report)

print("\n✓ Сохранено:")
print("  - best_traffic_signs_model.keras")
print("  - traffic_signs_model.keras (ниже)")
print("  - training_history.png")
print("  - confusion_matrix.png")
print("  - metrics_summary.txt")
print("  - classification_report.txt")


model.save("traffic_signs_model.keras")
print("\n✓ Модель сохранена: traffic_signs_model.keras")
print("✓ Лучшая модель: best_traffic_signs_model.keras")
