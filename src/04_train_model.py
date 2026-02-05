"""
Chest X-Ray Pneumonia Detection - Model Training
This is the MAIN training script that brings everything together
"""

import os
import sys
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Dense, Dropout, GlobalAveragePooling2D, BatchNormalization
)
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import (
    ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
)
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt
from datetime import datetime

# ============= GPU CONFIGURATION =============
# Enable GPU memory growth to prevent TensorFlow from allocating all GPU memory at once
physical_devices = tf.config.list_physical_devices('GPU')
if len(physical_devices) > 0:
    print(f"\n🎮 GPU DETECTED: {len(physical_devices)} GPU(s) available")
    for gpu in physical_devices:
        print(f"   • {gpu}")
        # Enable memory growth
        tf.config.experimental.set_memory_growth(gpu, True)
    print("✓ GPU memory growth enabled")
    print("✓ TensorFlow will use GPU for training\n")
else:
    print("\n⚠️  No GPU detected. Training will use CPU (slower)")
    print("   Consider using Google Colab for free GPU access\n")

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

class PneumoniaDetector:
    def __init__(self, data_dir, img_size=(224, 224), batch_size=32, epochs=25):
        """
        Initialize Pneumonia Detector
        
        Args:
            data_dir: Path to dataset directory
            img_size: Image size for training
            batch_size: Batch size
            epochs: Number of training epochs
        """
        self.data_dir = data_dir
        self.img_size = img_size
        self.batch_size = batch_size
        self.epochs = epochs
        
        self.train_dir = os.path.join(data_dir, 'train')
        self.test_dir = os.path.join(data_dir, 'test')
        self.val_dir = os.path.join(data_dir, 'val')
        
        self.history = None
        self.model = None
        
    def prepare_data(self):
        """Prepare data generators with augmentation"""
        print("\n" + "="*70)
        print("STEP 1: DATA PREPARATION")
        print("="*70)
        
        # Training data with augmentation
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
        
        # Validation/Test data without augmentation
        val_test_datagen = ImageDataGenerator(rescale=1./255)
        
        # Create generators
        self.train_generator = train_datagen.flow_from_directory(
            self.train_dir,
            target_size=self.img_size,
            batch_size=self.batch_size,
            class_mode='binary',
            shuffle=True,
            seed=42
        )
        
        self.test_generator = val_test_datagen.flow_from_directory(
            self.test_dir,
            target_size=self.img_size,
            batch_size=self.batch_size,
            class_mode='binary',
            shuffle=False
        )
        
        # Use validation set if exists, otherwise use test set
        if os.path.exists(self.val_dir):
            self.val_generator = val_test_datagen.flow_from_directory(
                self.val_dir,
                target_size=self.img_size,
                batch_size=self.batch_size,
                class_mode='binary',
                shuffle=False
            )
        else:
            print("⚠️  No validation set found. Using test set for validation.")
            self.val_generator = self.test_generator
        
        # Calculate class weights for imbalanced data
        class_weights = compute_class_weight(
            class_weight='balanced',
            classes=np.unique(self.train_generator.classes),
            y=self.train_generator.classes
        )
        self.class_weight_dict = {i: weight for i, weight in enumerate(class_weights)}
        
        print(f"\n✓ Training samples:   {self.train_generator.samples:,}")
        print(f"✓ Validation samples: {self.val_generator.samples:,}")
        print(f"✓ Test samples:       {self.test_generator.samples:,}")
        print(f"✓ Image size:         {self.img_size}")
        print(f"✓ Batch size:         {self.batch_size}")
        print(f"✓ Class mapping:      {self.train_generator.class_indices}")
        print(f"✓ Class weights:      {self.class_weight_dict}")
        
    def build_model(self):
        """Build transfer learning model with ResNet50"""
        print("\n" + "="*70)
        print("STEP 2: MODEL BUILDING")
        print("="*70)
        
        # Load pre-trained ResNet50
        base_model = ResNet50(
            weights='imagenet',
            include_top=False,
            input_shape=(*self.img_size, 3)
        )
        
        # Freeze base model
        base_model.trainable = False
        
        # Build complete model
        self.model = Sequential([
            base_model,
            GlobalAveragePooling2D(),
            Dense(512, activation='relu'),
            BatchNormalization(),
            Dropout(0.5),
            Dense(256, activation='relu'),
            BatchNormalization(),
            Dropout(0.3),
            Dense(1, activation='sigmoid')
        ])
        
        # Compile model
        self.model.compile(
            optimizer=Adam(learning_rate=0.00005),  # Lower learning rate
            loss='binary_crossentropy',
            metrics=[
                'accuracy',
                tf.keras.metrics.Precision(name='precision'),
                tf.keras.metrics.Recall(name='recall'),
                tf.keras.metrics.AUC(name='auc')
            ]
        )
        
        trainable_params = sum([tf.size(w).numpy() for w in self.model.trainable_weights])
        
        print(f"\n✓ Base Model: ResNet50 (pre-trained on ImageNet)")
        print(f"✓ Transfer Learning: Base layers frozen")
        print(f"✓ Custom top layers: Added")
        print(f"✓ Trainable parameters: {trainable_params:,}")
        print(f"✓ Optimizer: Adam (lr=0.0001)")
        print(f"✓ Loss: Binary Crossentropy")
        print(f"✓ Metrics: Accuracy, Precision, Recall, AUC")
        
    def train_model(self):
        """Train the model"""
        print("\n" + "="*70)
        print("STEP 3: MODEL TRAINING")
        print("="*70)
        
        # Create directories
        os.makedirs('models', exist_ok=True)
        os.makedirs('results', exist_ok=True)
        
        # Setup callbacks
        callbacks = [
            ModelCheckpoint(
                filepath='models/best_model.h5',
                monitor='val_accuracy',
                save_best_only=True,
                mode='max',
                verbose=1,
            ),
            EarlyStopping(
                monitor='val_loss',
                patience=20,  # More patience
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=7,  # More patience
                min_lr=1e-7,
                verbose=1
            )
        ]
        
        print(f"\n🚀 Starting training for {self.epochs} epochs...")
        print(f"⏱️  Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        
        # Train the model
        self.history = self.model.fit(
            self.train_generator,
            epochs=self.epochs,
            validation_data=self.val_generator,
            class_weight=self.class_weight_dict,
            callbacks=callbacks,
            verbose=1
        )
        
        print(f"\n✓ Training completed!")
        print(f"⏱️  End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Save final model
        self.model.save('models/final_model.h5')
        print(f"✓ Final model saved to 'models/final_model.h5'")
        print(f"✓ Best model saved to 'models/best_model.h5'")
        
    def plot_training_history(self):
        """Plot training history"""
        print("\n" + "="*70)
        print("STEP 4: VISUALIZING TRAINING HISTORY")
        print("="*70)
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Plot accuracy
        axes[0, 0].plot(self.history.history['accuracy'], label='Train Accuracy', linewidth=2)
        axes[0, 0].plot(self.history.history['val_accuracy'], label='Val Accuracy', linewidth=2)
        axes[0, 0].set_title('Model Accuracy', fontsize=14, fontweight='bold')
        axes[0, 0].set_xlabel('Epoch', fontsize=12)
        axes[0, 0].set_ylabel('Accuracy', fontsize=12)
        axes[0, 0].legend(fontsize=10)
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot loss
        axes[0, 1].plot(self.history.history['loss'], label='Train Loss', linewidth=2)
        axes[0, 1].plot(self.history.history['val_loss'], label='Val Loss', linewidth=2)
        axes[0, 1].set_title('Model Loss', fontsize=14, fontweight='bold')
        axes[0, 1].set_xlabel('Epoch', fontsize=12)
        axes[0, 1].set_ylabel('Loss', fontsize=12)
        axes[0, 1].legend(fontsize=10)
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot precision
        axes[1, 0].plot(self.history.history['precision'], label='Train Precision', linewidth=2)
        axes[1, 0].plot(self.history.history['val_precision'], label='Val Precision', linewidth=2)
        axes[1, 0].set_title('Model Precision', fontsize=14, fontweight='bold')
        axes[1, 0].set_xlabel('Epoch', fontsize=12)
        axes[1, 0].set_ylabel('Precision', fontsize=12)
        axes[1, 0].legend(fontsize=10)
        axes[1, 0].grid(True, alpha=0.3)
        
        # Plot recall
        axes[1, 1].plot(self.history.history['recall'], label='Train Recall', linewidth=2)
        axes[1, 1].plot(self.history.history['val_recall'], label='Val Recall', linewidth=2)
        axes[1, 1].set_title('Model Recall', fontsize=14, fontweight='bold')
        axes[1, 1].set_xlabel('Epoch', fontsize=12)
        axes[1, 1].set_ylabel('Recall', fontsize=12)
        axes[1, 1].legend(fontsize=10)
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('results/training_history.png', dpi=300, bbox_inches='tight')
        print("\n✓ Training history plot saved to 'results/training_history.png'")
        plt.show()
        
    def evaluate_model(self):
        """Evaluate model on test set"""
        print("\n" + "="*70)
        print("STEP 5: MODEL EVALUATION")
        print("="*70)
        
        # Load best model
        from tensorflow.keras.models import load_model
        best_model = load_model('models/best_model.h5')
        
        # Evaluate on test set
        test_results = best_model.evaluate(self.test_generator, verbose=0)
        
        print("\n📊 Test Set Performance:")
        print("-"*70)
        print(f"   Test Loss:      {test_results[0]:.4f}")
        print(f"   Test Accuracy:  {test_results[1]:.4f} ({test_results[1]*100:.2f}%)")
        print(f"   Test Precision: {test_results[2]:.4f}")
        print(f"   Test Recall:    {test_results[3]:.4f}")
        print(f"   Test AUC:       {test_results[4]:.4f}")
        
        # Calculate F1 Score
        f1_score = 2 * (test_results[2] * test_results[3]) / (test_results[2] + test_results[3])
        print(f"   Test F1-Score:  {f1_score:.4f}")
        print("-"*70)
        
    def run_complete_training(self):
        """Run complete training pipeline"""
        print("""
        ╔══════════════════════════════════════════════════════════════════╗
        ║     Chest X-Ray Pneumonia Detection - COMPLETE TRAINING         ║
        ╚══════════════════════════════════════════════════════════════════╝
        """)
        
        try:
            # Step 1: Prepare data
            self.prepare_data()
            
            # Step 2: Build model
            self.build_model()
            
            # Step 3: Train model
            self.train_model()
            
            # Step 4: Plot history
            self.plot_training_history()
            
            # Step 5: Evaluate model
            self.evaluate_model()
            
            print("\n" + "="*70)
            print("✅ TRAINING PIPELINE COMPLETED SUCCESSFULLY!")
            print("="*70)
            print("\n📁 Generated Files:")
            print("   • models/best_model.h5 - Best model (highest validation accuracy)")
            print("   • models/final_model.h5 - Final model after all epochs")
            print("   • results/training_history.png - Training visualization")
            print("\n📝 Next Steps:")
            print("   • Run prediction script to test on new X-rays")
            print("   • Check results/ folder for visualizations")
            print("="*70)
            
        except Exception as e:
            print(f"\n❌ ERROR during training: {str(e)}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    # ============= CONFIGURATION =============
    DATA_DIR = 'chest_xray'  # Dataset in root directory
    IMG_SIZE = (224, 224)
    BATCH_SIZE = 16  # Reduced for better gradient updates
    EPOCHS = 50  # Increased for better convergence
    # =========================================
    
    # Check if data exists
    if not os.path.exists(DATA_DIR):
        print(f"\n❌ ERROR: Dataset not found at '{DATA_DIR}'")
        print("\n📝 Instructions:")
        print("1. Download 'Chest X-Ray Images (Pneumonia)' from Kaggle")
        print("2. Extract to a folder")
        print("3. Update DATA_DIR variable in this script")
        print("\nExpected structure:")
        print("  chest_xray/")
        print("    ├── train/NORMAL/")
        print("    ├── train/PNEUMONIA/")
        print("    ├── test/NORMAL/")
        print("    ├── test/PNEUMONIA/")
        print("    └── val/NORMAL/")
        print("        └── val/PNEUMONIA/")
        sys.exit(1)
    
    # Run training
    detector = PneumoniaDetector(
        data_dir=DATA_DIR,
        img_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS
    )
    
    detector.run_complete_training()
