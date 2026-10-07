"""
Brain Tumor Classification - Training Script
Uses EfficientNetB0 with Transfer Learning to achieve 90%+ accuracy.
Dataset: 4 classes (glioma, meningioma, notumor, pituitary)
"""

import os
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import classification_report, confusion_matrix

# ============================================================
# Configuration
# ============================================================
TRAIN_DIR = "dataset/Training"
TEST_DIR = "dataset/Testing"
MODEL_DIR = "models"
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
INITIAL_EPOCHS = 15      # Phase 1: Train top layers only
FINE_TUNE_EPOCHS = 25    # Phase 2: Fine-tune EfficientNet layers
NUM_CLASSES = 4

os.makedirs(MODEL_DIR, exist_ok=True)

# ============================================================
# Data Generators with Augmentation
# ============================================================
print("=" * 60)
print("BRAIN TUMOR CLASSIFICATION - MODEL TRAINING")
print("=" * 60)
print(f"\nImage Size: {IMG_SIZE}")
print(f"Batch Size: {BATCH_SIZE}")
print(f"Training Directory: {TRAIN_DIR}")
print(f"Testing Directory: {TEST_DIR}")

# Training data: heavy augmentation to prevent overfitting
train_datagen = ImageDataGenerator(
    preprocessing_function=tf.keras.applications.efficientnet.preprocess_input,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    vertical_flip=False,
    brightness_range=[0.8, 1.2],
    fill_mode='nearest',
    validation_split=0.15  # 15% of training data for validation
)

# Test/validation data: only preprocessing, no augmentation
test_datagen = ImageDataGenerator(
    preprocessing_function=tf.keras.applications.efficientnet.preprocess_input
)

print("\nLoading training data...")
train_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training',
    shuffle=True,
    seed=42
)

print("\nLoading validation data...")
val_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation',
    shuffle=False,
    seed=42
)

print("\nLoading test data...")
test_generator = test_datagen.flow_from_directory(
    TEST_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)

# Save class indices
class_indices = train_generator.class_indices
print(f"\nClass Indices: {class_indices}")
with open(os.path.join(MODEL_DIR, "class_indices.json"), "w", encoding="utf-8") as f:
    json.dump(class_indices, f, indent=2)

class_names = list(class_indices.keys())
print(f"Classes: {class_names}")
print(f"Training samples: {train_generator.samples}")
print(f"Validation samples: {val_generator.samples}")
print(f"Test samples: {test_generator.samples}")

# ============================================================
# Build Model - EfficientNetB0 with Transfer Learning
# ============================================================
print("\n" + "=" * 60)
print("PHASE 1: Training Top Layers (Feature Extraction)")
print("=" * 60)

# Load EfficientNetB0 pretrained on ImageNet
base_model = EfficientNetB0(
    weights='imagenet',
    include_top=False,
    input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3)
)

# Freeze the base model
base_model.trainable = False

# Add custom classification head
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = BatchNormalization()(x)
x = Dropout(0.3)(x)
x = Dense(256, activation='relu')(x)
x = BatchNormalization()(x)
x = Dropout(0.3)(x)
predictions = Dense(NUM_CLASSES, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)

# Compile
model.compile(
    optimizer=Adam(learning_rate=1e-3),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# Callbacks
callbacks_phase1 = [
    ModelCheckpoint(
        os.path.join(MODEL_DIR, "best_brain_tumor_model.h5"),
        monitor='val_accuracy',
        save_best_only=True,
        mode='max',
        verbose=1
    ),
    EarlyStopping(
        monitor='val_accuracy',
        patience=5,
        restore_best_weights=True,
        verbose=1
    ),
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=3,
        min_lr=1e-7,
        verbose=1
    )
]

# Phase 1: Train top layers
history1 = model.fit(
    train_generator,
    epochs=INITIAL_EPOCHS,
    validation_data=val_generator,
    callbacks=callbacks_phase1,
    verbose=1
)

# ============================================================
# Phase 2: Fine-Tuning
# ============================================================
print("\n" + "=" * 60)
print("PHASE 2: Fine-Tuning EfficientNet Layers")
print("=" * 60)

# Unfreeze the last ~20 layers of EfficientNet
base_model.trainable = True
fine_tune_from = len(base_model.layers) - 20
for layer in base_model.layers[:fine_tune_from]:
    layer.trainable = False

trainable_count = sum(1 for layer in model.layers if layer.trainable)
print(f"Total layers: {len(model.layers)}")
print(f"Trainable layers: {trainable_count}")

# Recompile with a lower learning rate
model.compile(
    optimizer=Adam(learning_rate=1e-4),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

callbacks_phase2 = [
    ModelCheckpoint(
        os.path.join(MODEL_DIR, "best_brain_tumor_model.h5"),
        monitor='val_accuracy',
        save_best_only=True,
        mode='max',
        verbose=1
    ),
    EarlyStopping(
        monitor='val_accuracy',
        patience=7,
        restore_best_weights=True,
        verbose=1
    ),
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.3,
        patience=3,
        min_lr=1e-8,
        verbose=1
    )
]

# Phase 2: Fine-tune
history2 = model.fit(
    train_generator,
    epochs=FINE_TUNE_EPOCHS,
    validation_data=val_generator,
    callbacks=callbacks_phase2,
    verbose=1
)

# Save final model
model.save(os.path.join(MODEL_DIR, "brain_tumor_model.h5"))
print(f"\nFinal model saved to {MODEL_DIR}/brain_tumor_model.h5")

# ============================================================
# Evaluation on Test Set
# ============================================================
print("\n" + "=" * 60)
print("EVALUATION ON TEST SET")
print("=" * 60)

# Evaluate
test_loss, test_accuracy = model.evaluate(test_generator, verbose=1)
print(f"\n{'=' * 40}")
print(f"  TEST ACCURACY: {test_accuracy * 100:.2f}%")
print(f"  TEST LOSS: {test_loss:.4f}")
print(f"{'=' * 40}")

if test_accuracy >= 0.90:
    print("\n✅ TARGET ACHIEVED! Model accuracy >= 90%")
else:
    print(f"\n⚠️  Model accuracy is {test_accuracy*100:.2f}%. Consider training for more epochs.")

# Classification Report
test_generator.reset()
predictions = model.predict(test_generator, verbose=1)
y_pred = np.argmax(predictions, axis=1)
y_true = test_generator.classes

print("\nClassification Report:")
print("-" * 60)
report = classification_report(y_true, y_pred, target_names=class_names)
print(report)

# Save classification report
with open(os.path.join(MODEL_DIR, "classification_report.txt"), "w") as f:
    f.write(f"Test Accuracy: {test_accuracy * 100:.2f}%\n")
    f.write(f"Test Loss: {test_loss:.4f}\n\n")
    f.write(report)

# ============================================================
# Plots
# ============================================================

# Combine histories
acc = history1.history['accuracy'] + history2.history['accuracy']
val_acc = history1.history['val_accuracy'] + history2.history['val_accuracy']
loss = history1.history['loss'] + history2.history['loss']
val_loss = history1.history['val_loss'] + history2.history['val_loss']
epochs_range = range(1, len(acc) + 1)

# Accuracy and Loss curves
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

ax1.plot(epochs_range, acc, 'b-', label='Training Accuracy')
ax1.plot(epochs_range, val_acc, 'r-', label='Validation Accuracy')
ax1.axvline(x=len(history1.history['accuracy']), color='gray', linestyle='--', label='Fine-tuning Start')
ax1.set_title('Training & Validation Accuracy')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Accuracy')
ax1.legend()
ax1.grid(True, alpha=0.3)

ax2.plot(epochs_range, loss, 'b-', label='Training Loss')
ax2.plot(epochs_range, val_loss, 'r-', label='Validation Loss')
ax2.axvline(x=len(history1.history['loss']), color='gray', linestyle='--', label='Fine-tuning Start')
ax2.set_title('Training & Validation Loss')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Loss')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(MODEL_DIR, "training_curves.png"), dpi=150)
print(f"\nTraining curves saved to {MODEL_DIR}/training_curves.png")

# Confusion Matrix
fig, ax = plt.subplots(figsize=(8, 6))
cm = confusion_matrix(y_true, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names, yticklabels=class_names, ax=ax)
ax.set_title(f'Confusion Matrix (Accuracy: {test_accuracy*100:.1f}%)')
ax.set_xlabel('Predicted')
ax.set_ylabel('Actual')
plt.tight_layout()
plt.savefig(os.path.join(MODEL_DIR, "confusion_matrix.png"), dpi=150)
print(f"Confusion matrix saved to {MODEL_DIR}/confusion_matrix.png")

print("\n" + "=" * 60)
print("TRAINING COMPLETE!")
print("=" * 60)
