"""
Fine-tune the trained model for better performance
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt

# GPU Configuration
physical_devices = tf.config.list_physical_devices('GPU')
if len(physical_devices) > 0:
    for gpu in physical_devices:
        tf.config.experimental.set_memory_growth(gpu, True)
    print(f"✓ GPU detected and configured\n")

# Configuration
DATA_DIR = 'chest_xray'
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
FINE_TUNE_EPOCHS = 20

print("""
╔══════════════════════════════════════════════════════════════════╗
║          FINE-TUNING MODEL FOR BETTER PERFORMANCE                ║
╚══════════════════════════════════════════════════════════════════╝
""")

# Load the trained model
print("📦 Loading trained model...")
model = load_model('models/best_model.h5')
print("✓ Model loaded\n")

# Unfreeze the last 20 layers of ResNet50 for fine-tuning
print("🔓 Unfreezing last 20 layers of base model...")
base_model = model.layers[0]
base_model.trainable = True

# Freeze all layers except the last 20
for layer in base_model.layers[:-20]:
    layer.trainable = False

print(f"✓ Trainable layers: {sum([1 for layer in model.layers if layer.trainable])}")

# Recompile with lower learning rate
model.compile(
    optimizer=Adam(learning_rate=0.00001),  # Lower learning rate for fine-tuning
    loss='binary_crossentropy',
    metrics=[
        'accuracy',
        tf.keras.metrics.Precision(name='precision'),
        tf.keras.metrics.Recall(name='recall'),
        tf.keras.metrics.AUC(name='auc')
    ]
)
print("✓ Model recompiled with lr=0.00001\n")

# Prepare data generators
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

val_test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    os.path.join(DATA_DIR, 'train'),
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary',
    shuffle=True,
    seed=42
)

test_generator = val_test_datagen.flow_from_directory(
    os.path.join(DATA_DIR, 'test'),
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary',
    shuffle=False
)

# Calculate class weights
class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(train_generator.classes),
    y=train_generator.classes
)
class_weight_dict = {i: weight for i, weight in enumerate(class_weights)}

# Callbacks
callbacks = [
    ModelCheckpoint(
        filepath='models/fine_tuned_model.h5',
        monitor='val_accuracy',
        save_best_only=True,
        mode='max',
        verbose=1
    ),
    EarlyStopping(
        monitor='val_loss',
        patience=8,
        restore_best_weights=True,
        verbose=1
    ),
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=4,
        min_lr=1e-8,
        verbose=1
    )
]

print(f"🚀 Starting fine-tuning for {FINE_TUNE_EPOCHS} epochs...\n")

# Fine-tune
history = model.fit(
    train_generator,
    epochs=FINE_TUNE_EPOCHS,
    validation_data=test_generator,
    class_weight=class_weight_dict,
    callbacks=callbacks,
    verbose=1
)

print("\n✅ Fine-tuning completed!")
print("✓ Fine-tuned model saved to 'models/fine_tuned_model.h5'")

# Evaluate
print("\n📊 Evaluating fine-tuned model...")
results = model.evaluate(test_generator, verbose=0)
print(f"\nTest Accuracy:  {results[1]*100:.2f}%")
print(f"Test Precision: {results[2]:.4f}")
print(f"Test Recall:    {results[3]:.4f}")
print(f"Test AUC:       {results[4]:.4f}")

print("\n✅ Done! Use 'models/fine_tuned_model.h5' for predictions")
