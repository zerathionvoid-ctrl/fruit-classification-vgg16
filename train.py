import os
import json
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ReduceLROnPlateau,
    ModelCheckpoint
)

# ====================================================
# CONFIG
# ====================================================

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 64
EPOCHS = 6

TRAIN_DIR = "dataset/train"
VALID_DIR = "dataset/valid"
TEST_DIR = "dataset/test"

SAVE_MODEL = "saved_model/fruit_vgg16.keras"
CLASS_JSON = "saved_model/class_names.json"

os.makedirs("saved_model", exist_ok=True)
os.makedirs("logs", exist_ok=True)

print("=" * 50)
print("Fruit Classification Using VGG16")
print("=" * 50)

# ====================================================
# LOAD DATASET
# ====================================================

train_dataset = tf.keras.preprocessing.image_dataset_from_directory(
    TRAIN_DIR,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=True
)

valid_dataset = tf.keras.preprocessing.image_dataset_from_directory(
    VALID_DIR,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False
)

test_dataset = tf.keras.preprocessing.image_dataset_from_directory(
    TEST_DIR,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False
)

class_names = train_dataset.class_names
num_classes = len(class_names)

print("\nClass Names:")
print(class_names)

with open(CLASS_JSON, "w") as f:
    json.dump(class_names, f)

AUTOTUNE = tf.data.AUTOTUNE

train_dataset = train_dataset.prefetch(AUTOTUNE)
valid_dataset = valid_dataset.prefetch(AUTOTUNE)
test_dataset = test_dataset.prefetch(AUTOTUNE)

# ====================================================
# BUILD MODEL
# ====================================================

base_model = VGG16(
    weights="imagenet",
    include_top=False,
    input_shape=(224,224,3)
)

base_model.trainable = False

x = base_model.output
x = GlobalAveragePooling2D()(x)

x = Dense(256, activation="relu")(x)
x = Dropout(0.5)(x)

outputs = Dense(num_classes, activation="softmax")(x)

model = Model(inputs=base_model.input, outputs=outputs)

model.compile(
    optimizer=Adam(learning_rate=0.0001),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# ====================================================
# CALLBACK
# ====================================================

early_stop = EarlyStopping(
    monitor="val_loss",
    patience=2,
    restore_best_weights=True
)

reduce_lr = ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.3,
    patience=2,
    verbose=1
)

checkpoint = ModelCheckpoint(
    SAVE_MODEL,
    monitor="val_accuracy",
    save_best_only=True,
    verbose=1
)

callbacks = [
    early_stop,
    reduce_lr,
    checkpoint
]

# ====================================================
# TRAIN
# ====================================================

history = model.fit(
    train_dataset,
    validation_data=valid_dataset,
    epochs=EPOCHS,
    callbacks=callbacks
)

# ====================================================
# FINE TUNING
# ====================================================

print("\nFine Tuning...\n")

base_model.trainable = True

for layer in base_model.layers[:-4]:
    layer.trainable = False

model.compile(
    optimizer=Adam(learning_rate=1e-5),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

history_fine = model.fit(
    train_dataset,
    validation_data=valid_dataset,
    epochs=2,
    callbacks=callbacks
)

# ====================================================
# SAVE MODEL
# ====================================================

model.save(SAVE_MODEL)

print("\nModel saved!")

# ====================================================
# TEST
# ====================================================

loss, acc = model.evaluate(test_dataset)

print("\n============================")
print("Test Accuracy :", round(acc*100,2),"%")
print("Test Loss :", round(loss,4))
print("============================")

# ====================================================
# PREDICTION
# ====================================================

y_true = []
y_pred = []

for images, labels in test_dataset:

    pred = model.predict(images, verbose=0)

    pred = np.argmax(pred, axis=1)

    y_true.extend(labels.numpy())
    y_pred.extend(pred)

print("\nClassification Report\n")

report = classification_report(
    y_true,
    y_pred,
    target_names=class_names
)

print(report)

with open("logs/classification_report.txt","w") as f:
    f.write(report)

# ====================================================
# CONFUSION MATRIX
# ====================================================

cm = confusion_matrix(y_true,y_pred)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=class_names
)

fig, ax = plt.subplots(figsize=(8,8))
disp.plot(ax=ax)

plt.title("Confusion Matrix")

plt.savefig("logs/confusion_matrix.png")

# ====================================================
# GRAPH
# ====================================================

acc = history.history["accuracy"] + history_fine.history["accuracy"]
val_acc = history.history["val_accuracy"] + history_fine.history["val_accuracy"]

loss = history.history["loss"] + history_fine.history["loss"]
val_loss = history.history["val_loss"] + history_fine.history["val_loss"]

plt.figure(figsize=(10,5))

plt.plot(acc,label="Train Accuracy")
plt.plot(val_acc,label="Validation Accuracy")

plt.legend()

plt.grid()

plt.title("Accuracy")

plt.savefig("logs/accuracy.png")

plt.figure(figsize=(10,5))

plt.plot(loss,label="Train Loss")
plt.plot(val_loss,label="Validation Loss")

plt.legend()

plt.grid()

plt.title("Loss")

plt.savefig("logs/loss.png")

print("\nTraining Finished Successfully!")
print("\nSaved Files:")
print("- fruit_vgg16.keras")
print("- class_names.json")
print("- accuracy.png")
print("- loss.png")
print("- confusion_matrix.png")
print("- classification_report.txt")