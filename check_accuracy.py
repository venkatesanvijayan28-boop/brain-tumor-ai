"""
Brain Tumor AI - Accuracy Evaluation Program
Checks the model's accuracy on the testing and training datasets.
Generates per-class metrics, confusion matrix, and visual reports.
"""

import os
import sys
import json
import numpy as np

# Ensure UTF-8 output encoding for Windows command prompts
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


def evaluate_model(model_path=None, test_dir="dataset/Testing", img_size=(224, 224), batch_size=32):
    """
    Evaluate the brain tumor classification model on the dataset.
    """
    # 1. Determine model path
    if model_path is None:
        if os.path.exists("models/best_brain_tumor_model.h5"):
            model_path = "models/best_brain_tumor_model.h5"
        elif os.path.exists("models/brain_tumor_model.h5"):
            model_path = "models/brain_tumor_model.h5"
        else:
            print("[ERROR] No trained model found in 'models/' directory.")
            return None

    print("\n" + "=" * 65)
    print("        BRAIN TUMOR AI -- ACCURACY EVALUATION TOOL")
    print("=" * 65)
    print(f"  * Model File:        {model_path}")
    print(f"  * Test Dataset:      {test_dir}")
    print(f"  * Target Image Size: {img_size}")
    print("=" * 65)

    # 2. Load Model
    print("\nLoading trained neural network model...")
    model = load_model(model_path)
    print("Model loaded successfully!")

    # 3. Load Test Data
    print(f"\nScanning test dataset from '{test_dir}'...")
    test_datagen = ImageDataGenerator(
        preprocessing_function=tf.keras.applications.efficientnet.preprocess_input
    )

    test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False
    )

    total_images = test_generator.samples
    num_classes = len(test_generator.class_indices)
    class_names = list(test_generator.class_indices.keys())

    print(f"Found {total_images} test images across {num_classes} categories: {class_names}")

    # 4. Evaluate Overall Loss and Accuracy
    print("\nCalculating accuracy across test batches...")
    loss, accuracy = model.evaluate(test_generator, verbose=1)

    # 5. Predict for Detailed Class-by-Class Metrics
    test_generator.reset()
    predictions = model.predict(test_generator, verbose=0)
    y_pred = np.argmax(predictions, axis=1)
    y_true = test_generator.classes

    acc_percent = accuracy * 100
    cm = confusion_matrix(y_true, y_pred)

    # Display Results Summary
    print("\n" + "+" + "-" * 63 + "+")
    print(f"| {'FINAL EVALUATION RESULTS':^61} |")
    print("+" + "-" * 63 + "+")
    print(f"|  Overall Accuracy : {acc_percent:6.2f}%{' ':35}|")
    print(f"|  Overall Loss     : {loss:6.4f}{' ':36}|")
    print(f"|  Total Evaluated  : {total_images} images{' ':32}|")

    if acc_percent >= 90.0:
        grade = "EXCELLENT (>= 90% Target Met)"
    elif acc_percent >= 80.0:
        grade = "VERY GOOD (80% - 90%)"
    elif acc_percent >= 70.0:
        grade = "MODERATE (70% - 80%)"
    else:
        grade = "NEEDS RETRAINING (< 70%)"
    print(f"|  Status           : {grade:<36}|")
    print("+" + "-" * 63 + "+")

    # 6. Detailed Class Breakdown Table
    print("\nDetailed Per-Class Breakdown:")
    print("-" * 65)
    print(f"{'Class Name':<15} | {'Correct':<9} | {'Total':<6} | {'Accuracy':<10} | {'Status'}")
    print("-" * 65)
    for i, name in enumerate(class_names):
        total_for_class = int(np.sum(y_true == i))
        correct_for_class = int(cm[i][i])
        cls_acc = (correct_for_class / total_for_class * 100) if total_for_class > 0 else 0
        status_tag = "[PASS]" if cls_acc >= 85 else ("[WARN]" if cls_acc >= 60 else "[FAIL]")
        print(f"{name.capitalize():<15} | {correct_for_class:>4}/{total_for_class:<4} | {total_for_class:<6} | {cls_acc:6.2f}%    | {status_tag}")
    print("-" * 65)

    # 7. Classification Report (Precision, Recall, F1)
    print("\nScikit-Learn Classification Report:")
    print(classification_report(y_true, y_pred, target_names=[c.capitalize() for c in class_names]))

    # 8. Plot & Save Confusion Matrix
    output_dir = os.path.dirname(os.path.abspath(model_path))
    cm_path = os.path.join(output_dir, "evaluation_confusion_matrix.png")
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=[c.capitalize() for c in class_names],
        yticklabels=[c.capitalize() for c in class_names]
    )
    plt.title(f'Test Confusion Matrix - Accuracy: {acc_percent:.2f}%', fontsize=14, pad=15)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Ground Truth Label', fontsize=12)
    plt.tight_layout()
    plt.savefig(cm_path, dpi=200)
    plt.close()
    print(f"Confusion matrix plot saved to: '{cm_path}'\n")

    return {
        "accuracy": acc_percent,
        "loss": loss,
        "class_report": classification_report(y_true, y_pred, target_names=class_names, output_dict=True),
        "confusion_matrix": cm.tolist()
    }


if __name__ == "__main__":
    evaluate_model()
