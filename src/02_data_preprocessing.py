"""
Chest X-Ray Pneumonia Detection - Data Preprocessing
This script handles data preprocessing and augmentation
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

class DataPreprocessor:
    def __init__(self, data_dir, img_size=(224, 224), batch_size=32):
        """
        Initialize Data Preprocessor
        
        Args:
            data_dir: Path to dataset directory
            img_size: Target image size (height, width)
            batch_size: Batch size for training
        """
        self.data_dir = data_dir
        self.img_size = img_size
        self.batch_size = batch_size
        
        self.train_dir = os.path.join(data_dir, 'train')
        self.test_dir = os.path.join(data_dir, 'test')
        self.val_dir = os.path.join(data_dir, 'val')
        
    def create_data_generators(self):
        """
        Create data generators with augmentation for training
        and without augmentation for validation/test
        """
        print("\n🔧 Creating Data Generators...")
        print("="*60)
        
        # Training data generator with augmentation
        train_datagen = ImageDataGenerator(
            rescale=1./255,                    # Normalize pixel values to [0,1]
            rotation_range=20,                 # Random rotation ±20 degrees
            width_shift_range=0.2,             # Random horizontal shift
            height_shift_range=0.2,            # Random vertical shift
            shear_range=0.2,                   # Shear transformation
            zoom_range=0.2,                    # Random zoom
            horizontal_flip=True,              # Random horizontal flip
            fill_mode='nearest'                # Fill mode for new pixels
        )
        
        # Validation and test generators (no augmentation, only rescaling)
        val_test_datagen = ImageDataGenerator(rescale=1./255)
        
        # Create train generator
        train_generator = train_datagen.flow_from_directory(
            self.train_dir,
            target_size=self.img_size,
            batch_size=self.batch_size,
            class_mode='binary',               # Binary classification: 0=Normal, 1=Pneumonia
            shuffle=True,
            seed=42
        )
        
        # Create validation generator
        if os.path.exists(self.val_dir):
            val_generator = val_test_datagen.flow_from_directory(
                self.val_dir,
                target_size=self.img_size,
                batch_size=self.batch_size,
                class_mode='binary',
                shuffle=False
            )
        else:
            print("⚠️  Validation directory not found. Using test set for validation.")
            val_generator = None
        
        # Create test generator
        test_generator = val_test_datagen.flow_from_directory(
            self.test_dir,
            target_size=self.img_size,
            batch_size=self.batch_size,
            class_mode='binary',
            shuffle=False
        )
        
        print("\n✅ Data generators created successfully!")
        print(f"   Training samples:   {train_generator.samples}")
        if val_generator:
            print(f"   Validation samples: {val_generator.samples}")
        print(f"   Test samples:       {test_generator.samples}")
        print(f"   Image size:         {self.img_size}")
        print(f"   Batch size:         {self.batch_size}")
        print(f"   Classes:            {train_generator.class_indices}")
        print("="*60)
        
        return train_generator, val_generator, test_generator
    
    def visualize_augmentation(self, train_generator):
        """
        Visualize data augmentation on sample images
        """
        print("\n📊 Visualizing Data Augmentation...")
        
        # Get one batch of images
        images, labels = next(train_generator)
        
        # Plot 8 augmented images
        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        axes = axes.ravel()
        
        for idx in range(8):
            axes[idx].imshow(images[idx], cmap='gray')
            label = "PNEUMONIA" if labels[idx] == 1 else "NORMAL"
            color = 'red' if labels[idx] == 1 else 'green'
            axes[idx].set_title(f'{label}', fontsize=12, fontweight='bold', color=color)
            axes[idx].axis('off')
        
        plt.suptitle('Augmented Training Images (After Preprocessing)', 
                     fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig('results/augmented_samples.png', dpi=300, bbox_inches='tight')
        print("✓ Augmentation visualization saved to 'results/augmented_samples.png'")
        plt.show()
    
    def get_class_weights(self, train_generator):
        """
        Calculate class weights to handle class imbalance
        """
        from sklearn.utils.class_weight import compute_class_weight
        
        # Get class distribution
        class_counts = np.bincount(train_generator.classes)
        total_samples = len(train_generator.classes)
        
        print("\n⚖️  Calculating Class Weights for Imbalanced Data...")
        print("="*60)
        print(f"   Normal:    {class_counts[0]:,} samples ({class_counts[0]/total_samples*100:.1f}%)")
        print(f"   Pneumonia: {class_counts[1]:,} samples ({class_counts[1]/total_samples*100:.1f}%)")
        
        # Calculate weights
        class_weights = compute_class_weight(
            class_weight='balanced',
            classes=np.unique(train_generator.classes),
            y=train_generator.classes
        )
        
        class_weight_dict = {i: weight for i, weight in enumerate(class_weights)}
        
        print(f"\n   Class Weights: {class_weight_dict}")
        print("   (Higher weight for minority class to balance training)")
        print("="*60)
        
        return class_weight_dict


if __name__ == "__main__":
    print("""
    ╔══════════════════════════════════════════════════════════╗
    ║   Chest X-Ray Pneumonia Detection - Data Preprocessing  ║
    ╚══════════════════════════════════════════════════════════╝
    """)
    
    # Configuration
    DATA_DIR = 'chest_xray'
    IMG_SIZE = (224, 224)
    BATCH_SIZE = 32
    
    if not os.path.exists(DATA_DIR):
        print(f"\n❌ ERROR: Data directory not found at '{DATA_DIR}'")
        print("Please update DATA_DIR to point to your dataset location.")
    else:
        # Create results directory
        os.makedirs('results', exist_ok=True)
        
        # Initialize preprocessor
        preprocessor = DataPreprocessor(DATA_DIR, IMG_SIZE, BATCH_SIZE)
        
        # Create generators
        train_gen, val_gen, test_gen = preprocessor.create_data_generators()
        
        # Visualize augmentation
        preprocessor.visualize_augmentation(train_gen)
        
        # Calculate class weights
        class_weights = preprocessor.get_class_weights(train_gen)
        
        print("\n✅ Preprocessing complete!")
        print("📝 Next step: Run the model training script")
