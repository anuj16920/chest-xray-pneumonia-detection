A"""
Chest X-Ray Pneumonia Detection - Model Architecture
This script defines the CNN model using Transfer Learning
"""

import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (
    Dense, Dropout, GlobalAveragePooling2D, 
    Conv2D, MaxPooling2D, Flatten, BatchNormalization
)
from tensorflow.keras.applications import ResNet50, VGG16, EfficientNetB0
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import (
    ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, TensorBoard
)
import os

class ModelBuilder:
    def __init__(self, img_size=(224, 224, 3)):
        """
        Initialize Model Builder
        
        Args:
            img_size: Input image shape (height, width, channels)
        """
        self.img_size = img_size
        
    def build_transfer_learning_model(self, base_model_name='ResNet50'):
        """
        Build model using Transfer Learning
        
        Args:
            base_model_name: Pre-trained model to use ('ResNet50', 'VGG16', or 'EfficientNetB0')
        """
        print(f"\n🏗️  Building Transfer Learning Model with {base_model_name}...")
        print("="*60)
        
        # Load pre-trained base model
        if base_model_name == 'ResNet50':
            base_model = ResNet50(
                weights='imagenet',
                include_top=False,
                input_shape=self.img_size
            )
        elif base_model_name == 'VGG16':
            base_model = VGG16(
                weights='imagenet',
                include_top=False,
                input_shape=self.img_size
            )
        elif base_model_name == 'EfficientNetB0':
            base_model = EfficientNetB0(
                weights='imagenet',
                include_top=False,
                input_shape=self.img_size
            )
        else:
            raise ValueError(f"Unknown base model: {base_model_name}")
        
        # Freeze the base model layers
        base_model.trainable = False
        
        # Build the model
        model = Sequential([
            base_model,
            GlobalAveragePooling2D(),
            Dense(512, activation='relu'),
            BatchNormalization(),
            Dropout(0.5),
            Dense(256, activation='relu'),
            BatchNormalization(),
            Dropout(0.3),
            Dense(1, activation='sigmoid')  # Binary classification
        ])
        
        print(f"✓ Base model: {base_model_name} (pre-trained on ImageNet)")
        print(f"✓ Base model layers frozen: {len(base_model.layers)}")
        print(f"✓ Custom top layers added")
        print(f"✓ Total trainable parameters: {self.count_trainable_params(model):,}")
        print("="*60)
        
        return model
    
    def build_custom_cnn_model(self):
        """
        Build custom CNN model from scratch (alternative approach)
        """
        print("\n🏗️  Building Custom CNN Model...")
        print("="*60)
        
        model = Sequential([
            # First Convolutional Block
            Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=self.img_size),
            BatchNormalization(),
            Conv2D(32, (3, 3), activation='relu', padding='same'),
            BatchNormalization(),
            MaxPooling2D((2, 2)),
            Dropout(0.25),
            
            # Second Convolutional Block
            Conv2D(64, (3, 3), activation='relu', padding='same'),
            BatchNormalization(),
            Conv2D(64, (3, 3), activation='relu', padding='same'),
            BatchNormalization(),
            MaxPooling2D((2, 2)),
            Dropout(0.25),
            
            # Third Convolutional Block
            Conv2D(128, (3, 3), activation='relu', padding='same'),
            BatchNormalization(),
            Conv2D(128, (3, 3), activation='relu', padding='same'),
            BatchNormalization(),
            MaxPooling2D((2, 2)),
            Dropout(0.25),
            
            # Fourth Convolutional Block
            Conv2D(256, (3, 3), activation='relu', padding='same'),
            BatchNormalization(),
            Conv2D(256, (3, 3), activation='relu', padding='same'),
            BatchNormalization(),
            MaxPooling2D((2, 2)),
            Dropout(0.25),
            
            # Fully Connected Layers
            Flatten(),
            Dense(512, activation='relu'),
            BatchNormalization(),
            Dropout(0.5),
            Dense(256, activation='relu'),
            BatchNormalization(),
            Dropout(0.5),
            Dense(1, activation='sigmoid')
        ])
        
        print(f"✓ Custom CNN architecture created")
        print(f"✓ Total trainable parameters: {self.count_trainable_params(model):,}")
        print("="*60)
        
        return model
    
    def compile_model(self, model, learning_rate=0.0001):
        """
        Compile the model with optimizer and loss function
        
        Args:
            model: Keras model to compile
            learning_rate: Learning rate for optimizer
        """
        print("\n⚙️  Compiling Model...")
        
        model.compile(
            optimizer=Adam(learning_rate=learning_rate),
            loss='binary_crossentropy',
            metrics=[
                'accuracy',
                tf.keras.metrics.Precision(name='precision'),
                tf.keras.metrics.Recall(name='recall'),
                tf.keras.metrics.AUC(name='auc')
            ]
        )
        
        print(f"✓ Optimizer: Adam (lr={learning_rate})")
        print(f"✓ Loss function: Binary Crossentropy")
        print(f"✓ Metrics: Accuracy, Precision, Recall, AUC")
        
        return model
    
    def get_callbacks(self, model_save_path='models/best_model.h5'):
        """
        Create callbacks for training
        
        Args:
            model_save_path: Path to save the best model
        """
        print("\n📋 Setting up Training Callbacks...")
        
        os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
        os.makedirs('logs', exist_ok=True)
        
        callbacks = [
            # Save best model
            ModelCheckpoint(
                filepath=model_save_path,
                monitor='val_accuracy',
                save_best_only=True,
                mode='max',
                verbose=1
            ),
            
            # Early stopping
            EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True,
                verbose=1
            ),
            
            # Reduce learning rate on plateau
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-7,
                verbose=1
            ),
            
            # TensorBoard logging
            TensorBoard(
                log_dir='logs',
                histogram_freq=1
            )
        ]
        
        print(f"✓ ModelCheckpoint: Saves best model to '{model_save_path}'")
        print(f"✓ EarlyStopping: Stops if val_loss doesn't improve for 10 epochs")
        print(f"✓ ReduceLROnPlateau: Reduces learning rate if val_loss plateaus")
        print(f"✓ TensorBoard: Logs training metrics to 'logs/' directory")
        
        return callbacks
    
    def count_trainable_params(self, model):
        """Count trainable parameters in the model"""
        return sum([tf.size(w).numpy() for w in model.trainable_weights])
    
    def print_model_summary(self, model):
        """Print detailed model summary"""
        print("\n📊 Model Architecture Summary:")
        print("="*60)
        model.summary()
        print("="*60)


if __name__ == "__main__":
    print("""
    ╔══════════════════════════════════════════════════════════╗
    ║   Chest X-Ray Pneumonia Detection - Model Architecture  ║
    ╚══════════════════════════════════════════════════════════╝
    """)
    
    # Configuration
    IMG_SIZE = (224, 224, 3)
    
    # Initialize model builder
    builder = ModelBuilder(IMG_SIZE)
    
    # Build transfer learning model (RECOMMENDED)
    print("\n🎯 Building TRANSFER LEARNING Model (Recommended):")
    model_tl = builder.build_transfer_learning_model('ResNet50')
    model_tl = builder.compile_model(model_tl, learning_rate=0.0001)
    builder.print_model_summary(model_tl)
    
    # Build custom CNN model (alternative)
    print("\n\n🎯 Building CUSTOM CNN Model (Alternative):")
    model_custom = builder.build_custom_cnn_model()
    model_custom = builder.compile_model(model_custom, learning_rate=0.001)
    
    # Get callbacks
    callbacks = builder.get_callbacks('models/xray_pneumonia_model.h5')
    
    print("\n✅ Model architecture ready!")
    print("📝 Recommendation: Use Transfer Learning model for better accuracy")
    print("📝 Next step: Run the training script")
