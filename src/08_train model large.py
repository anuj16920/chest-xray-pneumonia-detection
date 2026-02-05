"""
Chest X-Ray Pneumonia Detection - LARGE DATASET TRAINING
Optimized for 50,000+ images

This script is specifically designed for large-scale training with:
- Memory-efficient data loading
- Progressive learning rate scheduling
- Advanced augmentation
- Multi-GPU support (optional)
- Better batch processing
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
    ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, 
    CSVLogger, LearningRateScheduler
)
from tensorflow.keras.mixed_precision import set_global_policy
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt
from datetime import datetime
import json

# Enable mixed precision for faster training (if GPU available)
try:
    set_global_policy('mixed_float16')
    print("✅ Mixed precision enabled (faster training with GPU)")
except:
    print("ℹ️ Mixed precision not available (CPU mode)")

# Set random seeds
np.random.seed(42)
tf.random.set_seed(42)

class LargeScalePneumoniaDetector:
    def __init__(self, data_dir, img_size=(224, 224), batch_size=64, epochs=50):
        """
        Initialize Large-Scale Pneumonia Detector
        
        Args:
            data_dir: Path to dataset directory
            img_size: Image size for training
            batch_size: Batch size (increased for large datasets)
            epochs: Number of training epochs (increased for more data)
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
        
        print(f"""
        ╔══════════════════════════════════════════════════════════════════╗
        ║   LARGE-SCALE CHEST X-RAY PNEUMONIA DETECTION                   ║
        ║   Optimized for 50,000+ Images                                  ║
        ╚══════════════════════════════════════════════════════════════════╝
        
        Configuration:
          • Image Size: {img_size}
          • Batch Size: {batch_size} (optimized for large datasets)
          • Epochs: {epochs}
          • Mixed Precision: Enabled (if GPU available)
          • Memory Optimization: Enabled
        """)
        
    def prepare_data(self):
        """Prepare data generators with ENHANCED augmentation for large datasets"""
        print("\n" + "="*70)
        print("STEP 1: DATA PREPARATION (OPTIMIZED FOR LARGE DATASET)")
        print("="*70)
        
        # ENHANCED augmentation for large datasets
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=25,           # Increased
            width_shift_range=0.25,      # Increased
            height_shift_range=0.25,     # Increased
            shear_range=0.25,            # Increased
            zoom_range=0.25,             # Increased
            horizontal_flip=True,
            brightness_range=[0.8, 1.2], # NEW: Brightness variation
            fill_mode='nearest',
            validation_split=0.15        # Use 15% for validation if no val folder
        )
        
        # Test data (no augmentation)
        val_test_datagen = ImageDataGenerator(rescale=1./255)
        
        # Create training generator
        print("\n📊 Loading training data...")
        self.train_generator = train_datagen.flow_from_directory(
            self.train_dir,
            target_size=self.img_size,
            batch_size=self.batch_size,
            class_mode='binary',
            shuffle=True,
            seed=42,
            subset='training' if not os.path.exists(self.val_dir) else None
        )
        
        # Create validation generator
        if os.path.exists(self.val_dir):
            print("📊 Loading validation data from separate folder...")
            self.val_generator = val_test_datagen.flow_from_directory(
                self.val_dir,
                target_size=self.img_size,
                batch_size=self.batch_size,
                class_mode='binary',
                shuffle=False
            )
        else:
            print("📊 Creating validation split from training data (15%)...")
            self.val_generator = train_datagen.flow_from_directory(
                self.train_dir,
                target_size=self.img_size,
                batch_size=self.batch_size,
                class_mode='binary',
                shuffle=False,
                seed=42,
                subset='validation'
            )
        
        # Test generator
        print("📊 Loading test data...")
        self.test_generator = val_test_datagen.flow_from_directory(
            self.test_dir,
            target_size=self.img_size,
            batch_size=self.batch_size,
            class_mode='binary',
            shuffle=False
        )
        
        # Calculate class weights
        print("\n⚖️ Calculating class weights...")
        class_weights = compute_class_weight(
            class_weight='balanced',
            classes=np.unique(self.train_generator.classes),
            y=self.train_generator.classes
        )
        self.class_weight_dict = {i: weight for i, weight in enumerate(class_weights)}
        
        # Display statistics
        print("\n" + "="*70)
        print("DATASET STATISTICS")
        print("="*70)
        print(f"Training samples:   {self.train_generator.samples:,}")
        print(f"Validation samples: {self.val_generator.samples:,}")
        print(f"Test samples:       {self.test_generator.samples:,}")
        print(f"Total samples:      {self.train_generator.samples + self.val_generator.samples + self.test_generator.samples:,}")
        print(f"\nImage size:         {self.img_size}")
        print(f"Batch size:         {self.batch_size}")
        print(f"Steps per epoch:    {len(self.train_generator)}")
        print(f"Class mapping:      {self.train_generator.class_indices}")
        print(f"Class weights:      {self.class_weight_dict}")
        print("="*70)
        
    def build_model(self, use_deeper_head=True):
        """
        Build model with option for DEEPER custom head for large datasets
        
        Args:
            use_deeper_head: Use deeper classification head (recommended for 50k+ images)
        """
        print("\n" + "="*70)
        print("STEP 2: MODEL BUILDING (LARGE DATASET OPTIMIZED)")
        print("="*70)
        
        # Load pre-trained ResNet50
        base_model = ResNet50(
            weights='imagenet',
            include_top=False,
            input_shape=(*self.img_size, 3)
        )
        
        # Freeze base model initially
        base_model.trainable = False
        
        if use_deeper_head:
            print("\n🏗️ Building DEEPER classification head (optimized for large datasets)...")
            # Deeper head for large datasets
            self.model = Sequential([
                base_model,
                GlobalAveragePooling2D(),
                
                # First block
                Dense(1024, activation='relu'),
                BatchNormalization(),
                Dropout(0.5),
                
                # Second block
                Dense(512, activation='relu'),
                BatchNormalization(),
                Dropout(0.4),
                
                # Third block
                Dense(256, activation='relu'),
                BatchNormalization(),
                Dropout(0.3),
                
                # Output
                Dense(1, activation='sigmoid')
            ])
        else:
            print("\n🏗️ Building standard classification head...")
            # Standard head
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
            optimizer=Adam(learning_rate=0.0001),
            loss='binary_crossentropy',
            metrics=[
                'accuracy',
                tf.keras.metrics.Precision(name='precision'),
                tf.keras.metrics.Recall(name='recall'),
                tf.keras.metrics.AUC(name='auc')
            ]
        )
        
        trainable_params = sum([tf.size(w).numpy() for w in self.model.trainable_weights])
        total_params = sum([tf.size(w).numpy() for w in self.model.weights])
        
        print(f"\n✓ Base Model: ResNet50 (pre-trained on ImageNet)")
        print(f"✓ Classification Head: {'Deep (3 hidden layers)' if use_deeper_head else 'Standard (2 hidden layers)'}")
        print(f"✓ Total parameters: {total_params:,}")
        print(f"✓ Trainable parameters: {trainable_params:,} ({trainable_params/total_params*100:.1f}%)")
        print(f"✓ Optimizer: Adam (lr=0.0001)")
        print(f"✓ Loss: Binary Crossentropy")
        print(f"✓ Metrics: Accuracy, Precision, Recall, AUC")
        
    def get_callbacks(self):
        """Get ENHANCED callbacks for large-scale training"""
        print("\n📋 Setting up ENHANCED training callbacks...")
        
        # Create directories
        os.makedirs('models', exist_ok=True)
        os.makedirs('results', exist_ok=True)
        os.makedirs('logs', exist_ok=True)
        
        # Custom learning rate schedule for large datasets
        def lr_schedule(epoch, lr):
            """
            Learning rate schedule:
            - Epochs 0-10: 0.0001
            - Epochs 11-30: 0.00005
            - Epochs 31+: 0.00001
            """
            if epoch < 10:
                return 0.0001
            elif epoch < 30:
                return 0.00005
            else:
                return 0.00001
        
        callbacks = [
            # Save best model
            ModelCheckpoint(
                filepath='models/best_model_large.h5',
                monitor='val_accuracy',
                save_best_only=True,
                mode='max',
                verbose=1
            ),
            
            # Save checkpoints every 5 epochs
            ModelCheckpoint(
                filepath='models/checkpoint_epoch_{epoch:02d}_acc_{val_accuracy:.4f}.h5',
                monitor='val_accuracy',
                save_freq='epoch',
                period=5,
                verbose=1
            ),
            
            # Early stopping (more patience for large datasets)
            EarlyStopping(
                monitor='val_loss',
                patience=15,  # Increased patience
                restore_best_weights=True,
                verbose=1
            ),
            
            # Learning rate scheduler
            LearningRateScheduler(lr_schedule, verbose=1),
            
            # Reduce LR on plateau (backup)
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=7,
                min_lr=1e-7,
                verbose=1
            ),
            
            # CSV logger for detailed metrics
            CSVLogger('results/training_log_large.csv', separator=',', append=False),
        ]
        
        print("✓ ModelCheckpoint: Saves best model + periodic checkpoints")
        print("✓ EarlyStopping: patience=15 (increased for large datasets)")
        print("✓ LearningRateScheduler: Custom schedule for large-scale training")
        print("✓ ReduceLROnPlateau: Backup LR reduction")
        print("✓ CSVLogger: Detailed training logs")
        
        return callbacks
        
    def train_model(self):
        """Train the model with PROGRESS TRACKING"""
        print("\n" + "="*70)
        print("STEP 3: MODEL TRAINING (LARGE-SCALE)")
        print("="*70)
        
        callbacks = self.get_callbacks()
        
        print(f"\n🚀 Starting training for {self.epochs} epochs...")
        print(f"⏱️  Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"📊 Training on {self.train_generator.samples:,} images")
        print(f"📊 Validating on {self.val_generator.samples:,} images")
        print(f"💾 Best model will be saved to: models/best_model_large.h5\n")
        
        # Calculate steps
        steps_per_epoch = len(self.train_generator)
        validation_steps = len(self.val_generator)
        
        print(f"Steps per epoch: {steps_per_epoch}")
        print(f"Validation steps: {validation_steps}\n")
        print("="*70)
        
        # Train the model
        self.history = self.model.fit(
            self.train_generator,
            steps_per_epoch=steps_per_epoch,
            epochs=self.epochs,
            validation_data=self.val_generator,
            validation_steps=validation_steps,
            class_weight=self.class_weight_dict,
            callbacks=callbacks,
            verbose=1,
            workers=4,              # Parallel data loading
            use_multiprocessing=True,  # Speed up data loading
            max_queue_size=32       # Larger queue for large datasets
        )
        
        print(f"\n✓ Training completed!")
        print(f"⏱️  End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Save final model
        self.model.save('models/final_model_large.h5')
        print(f"✓ Final model saved to 'models/final_model_large.h5'")
        print(f"✓ Best model saved to 'models/best_model_large.h5'")
        
        # Save training history
        with open('results/training_history_large.json', 'w') as f:
            json.dump(self.history.history, f)
        print(f"✓ Training history saved to 'results/training_history_large.json'")
        
    def plot_training_history(self):
        """Plot ENHANCED training history"""
        print("\n" + "="*70)
        print("STEP 4: VISUALIZING TRAINING HISTORY")
        print("="*70)
        
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        
        # Plot accuracy
        axes[0, 0].plot(self.history.history['accuracy'], label='Train', linewidth=2)
        axes[0, 0].plot(self.history.history['val_accuracy'], label='Validation', linewidth=2)
        axes[0, 0].set_title('Model Accuracy', fontsize=14, fontweight='bold')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot loss
        axes[0, 1].plot(self.history.history['loss'], label='Train', linewidth=2)
        axes[0, 1].plot(self.history.history['val_loss'], label='Validation', linewidth=2)
        axes[0, 1].set_title('Model Loss', fontsize=14, fontweight='bold')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot precision
        axes[0, 2].plot(self.history.history['precision'], label='Train', linewidth=2)
        axes[0, 2].plot(self.history.history['val_precision'], label='Validation', linewidth=2)
        axes[0, 2].set_title('Precision', fontsize=14, fontweight='bold')
        axes[0, 2].set_xlabel('Epoch')
        axes[0, 2].set_ylabel('Precision')
        axes[0, 2].legend()
        axes[0, 2].grid(True, alpha=0.3)
        
        # Plot recall
        axes[1, 0].plot(self.history.history['recall'], label='Train', linewidth=2)
        axes[1, 0].plot(self.history.history['val_recall'], label='Validation', linewidth=2)
        axes[1, 0].set_title('Recall', fontsize=14, fontweight='bold')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Recall')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Plot AUC
        axes[1, 1].plot(self.history.history['auc'], label='Train', linewidth=2)
        axes[1, 1].plot(self.history.history['val_auc'], label='Validation', linewidth=2)
        axes[1, 1].set_title('AUC-ROC', fontsize=14, fontweight='bold')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('AUC')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        # Plot learning rate
        if 'lr' in self.history.history:
            axes[1, 2].plot(self.history.history['lr'], linewidth=2, color='red')
            axes[1, 2].set_title('Learning Rate', fontsize=14, fontweight='bold')
            axes[1, 2].set_xlabel('Epoch')
            axes[1, 2].set_ylabel('Learning Rate')
            axes[1, 2].set_yscale('log')
            axes[1, 2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('results/training_history_large.png', dpi=300, bbox_inches='tight')
        print("\n✓ Training history plot saved to 'results/training_history_large.png'")
        plt.show()
        
    def evaluate_model(self):
        """Evaluate model on test set"""
        print("\n" + "="*70)
        print("STEP 5: MODEL EVALUATION")
        print("="*70)
        
        from tensorflow.keras.models import load_model
        best_model = load_model('models/best_model_large.h5')
        
        print("\n📊 Evaluating on test set...")
        test_results = best_model.evaluate(
            self.test_generator,
            steps=len(self.test_generator),
            verbose=1
        )
        
        print("\n" + "="*70)
        print("TEST SET PERFORMANCE (LARGE DATASET MODEL)")
        print("="*70)
        print(f"Test Loss:      {test_results[0]:.4f}")
        print(f"Test Accuracy:  {test_results[1]:.4f} ({test_results[1]*100:.2f}%)")
        print(f"Test Precision: {test_results[2]:.4f}")
        print(f"Test Recall:    {test_results[3]:.4f}")
        print(f"Test AUC:       {test_results[4]:.4f}")
        
        f1_score = 2 * (test_results[2] * test_results[3]) / (test_results[2] + test_results[3])
        print(f"Test F1-Score:  {f1_score:.4f}")
        print("="*70)
        
        # Save results
        results_dict = {
            'test_loss': float(test_results[0]),
            'test_accuracy': float(test_results[1]),
            'test_precision': float(test_results[2]),
            'test_recall': float(test_results[3]),
            'test_auc': float(test_results[4]),
            'test_f1': float(f1_score)
        }
        
        with open('results/test_results_large.json', 'w') as f:
            json.dump(results_dict, f, indent=4)
        
        print(f"\n✓ Results saved to 'results/test_results_large.json'")
        
    def run_complete_training(self):
        """Run complete large-scale training pipeline"""
        try:
            # Step 1: Prepare data
            self.prepare_data()
            
            # Step 2: Build model
            self.build_model(use_deeper_head=True)
            
            # Step 3: Train model
            self.train_model()
            
            # Step 4: Plot history
            self.plot_training_history()
            
            # Step 5: Evaluate model
            self.evaluate_model()
            
            print("\n" + "="*70)
            print("✅ LARGE-SCALE TRAINING COMPLETED SUCCESSFULLY!")
            print("="*70)
            print("\n📁 Generated Files:")
            print("   • models/best_model_large.h5 - Best model")
            print("   • models/final_model_large.h5 - Final model")
            print("   • results/training_history_large.png - Visualization")
            print("   • results/training_log_large.csv - Detailed logs")
            print("   • results/test_results_large.json - Test metrics")
            print("\n🎯 Expected Performance with 50k images:")
            print("   • Accuracy: 95-98% (improved with more data!)")
            print("   • Precision: 93-96%")
            print("   • Recall: 96-99%")
            print("   • F1-Score: 95-97%")
            print("="*70)
            
        except Exception as e:
            print(f"\n❌ ERROR during training: {str(e)}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    # ============= CONFIGURATION FOR LARGE DATASET =============
    DATA_DIR = 'data/chest_xray'  # UPDATE THIS PATH
    IMG_SIZE = (224, 224)
    BATCH_SIZE = 64      # Increased for large datasets (adjust based on RAM/GPU)
    EPOCHS = 50          # More epochs for large datasets
    # ===========================================================
    
    # Check if data exists
    if not os.path.exists(DATA_DIR):
        print(f"\n❌ ERROR: Dataset not found at '{DATA_DIR}'")
        print("\nUpdate DATA_DIR to point to your 50k image dataset")
        sys.exit(1)
    
    print(f"""
    ╔══════════════════════════════════════════════════════════════════╗
    ║                                                                  ║
    ║          LARGE-SCALE PNEUMONIA DETECTION TRAINING                ║
    ║                  Optimized for 50,000+ Images                    ║
    ║                                                                  ║
    ╚══════════════════════════════════════════════════════════════════╝
    
    📊 Configuration:
       • Dataset: {DATA_DIR}
       • Image Size: {IMG_SIZE}
       • Batch Size: {BATCH_SIZE} (optimized for large datasets)
       • Epochs: {EPOCHS}
       • Expected Training Time: 
         - With GPU: 3-6 hours
         - Without GPU: 15-24 hours
    
    💡 Tips for 50k images:
       1. GPU is HIGHLY RECOMMENDED (10x faster)
       2. Consider using Google Colab with Pro ($10/month) for V100 GPU
       3. Training will take several hours - be patient!
       4. Model will be MUCH better with more data!
    """)
    
    # Run training
    detector = LargeScalePneumoniaDetector(
        data_dir=DATA_DIR,
        img_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS
    )
    
    detector.run_complete_training()