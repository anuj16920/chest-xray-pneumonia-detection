"""
Generate comprehensive test set metrics
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, 
    f1_score, roc_auc_score, confusion_matrix,
    classification_report
)

print("="*70)
print("COMPREHENSIVE TEST SET METRICS")
print("="*70)

# Load model
MODEL_PATH = 'models/best_model.h5'
print(f"\n📦 Loading model: {MODEL_PATH}")
model = load_model(MODEL_PATH)
print("✓ Model loaded\n")

# Prepare test data
DATA_DIR = 'chest_xray'
IMG_SIZE = (224, 224)
BATCH_SIZE = 32

test_datagen = ImageDataGenerator(rescale=1./255)
test_generator = test_datagen.flow_from_directory(
    os.path.join(DATA_DIR, 'test'),
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary',
    shuffle=False
)

print(f"📊 Test set: {test_generator.samples} images")
print(f"   Classes: {test_generator.class_indices}\n")

# Get predictions
print("🔮 Generating predictions...")
predictions_prob = model.predict(test_generator, verbose=1)
predictions_class = (predictions_prob > 0.5).astype(int).flatten()
true_labels = test_generator.classes

print("\n" + "="*70)
print("METRICS SUMMARY")
print("="*70)

# Calculate metrics
accuracy = accuracy_score(true_labels, predictions_class)
precision = precision_score(true_labels, predictions_class)
recall = recall_score(true_labels, predictions_class)
f1 = f1_score(true_labels, predictions_class)
auc_roc = roc_auc_score(true_labels, predictions_prob)

# Confusion matrix
cm = confusion_matrix(true_labels, predictions_class)
tn, fp, fn, tp = cm.ravel()

# Specificity
specificity = tn / (tn + fp)

print(f"\n📈 OVERALL METRICS:")
print(f"   Accuracy:    {accuracy:.4f} ({accuracy*100:.2f}%)")
print(f"   Precision:   {precision:.4f} ({precision*100:.2f}%)")
print(f"   Recall:      {recall:.4f} ({recall*100:.2f}%)")
print(f"   F1-Score:    {f1:.4f} ({f1*100:.2f}%)")
print(f"   AUC-ROC:     {auc_roc:.4f} ({auc_roc*100:.2f}%)")
print(f"   Specificity: {specificity:.4f} ({specificity*100:.2f}%)")

print(f"\n📊 CONFUSION MATRIX:")
print(f"   True Negatives (TN):  {tn:4d} (Correctly predicted NORMAL)")
print(f"   False Positives (FP): {fp:4d} (NORMAL predicted as PNEUMONIA)")
print(f"   False Negatives (FN): {fn:4d} (PNEUMONIA predicted as NORMAL)")
print(f"   True Positives (TP):  {tp:4d} (Correctly predicted PNEUMONIA)")

print(f"\n🎯 INTERPRETATION:")
print(f"   • Model correctly classifies {accuracy*100:.1f}% of X-rays")
print(f"   • When it predicts PNEUMONIA, it's right {precision*100:.1f}% of the time")
print(f"   • It catches {recall*100:.1f}% of actual pneumonia cases")
print(f"   • F1-Score balances precision and recall: {f1*100:.1f}%")
print(f"   • AUC-ROC shows overall discrimination ability: {auc_roc*100:.1f}%")

print("\n" + "="*70)
print("PER-CLASS BREAKDOWN")
print("="*70)

# Per-class metrics
report = classification_report(
    true_labels, 
    predictions_class,
    target_names=['NORMAL', 'PNEUMONIA'],
    digits=4
)
print(f"\n{report}")

# Save to file
with open('results/test_metrics_summary.txt', 'w') as f:
    f.write("="*70 + "\n")
    f.write("COMPREHENSIVE TEST SET METRICS\n")
    f.write("="*70 + "\n\n")
    f.write(f"Model: {MODEL_PATH}\n")
    f.write(f"Test samples: {test_generator.samples}\n\n")
    f.write(f"OVERALL METRICS:\n")
    f.write(f"  Accuracy:    {accuracy:.4f} ({accuracy*100:.2f}%)\n")
    f.write(f"  Precision:   {precision:.4f} ({precision*100:.2f}%)\n")
    f.write(f"  Recall:      {recall:.4f} ({recall*100:.2f}%)\n")
    f.write(f"  F1-Score:    {f1:.4f} ({f1*100:.2f}%)\n")
    f.write(f"  AUC-ROC:     {auc_roc:.4f} ({auc_roc*100:.2f}%)\n")
    f.write(f"  Specificity: {specificity:.4f} ({specificity*100:.2f}%)\n\n")
    f.write(f"CONFUSION MATRIX:\n")
    f.write(f"  TN: {tn:4d}  FP: {fp:4d}\n")
    f.write(f"  FN: {fn:4d}  TP: {tp:4d}\n\n")
    f.write("PER-CLASS BREAKDOWN:\n")
    f.write(report)
    f.write("\n" + "="*70 + "\n")

print("\n✅ Metrics saved to 'results/test_metrics_summary.txt'")
print("="*70)
