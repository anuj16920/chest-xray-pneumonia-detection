"""
Chest X-Ray Pneumonia Detection - Data Exploration
This script explores the dataset and visualizes the X-ray images
"""

import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from PIL import Image

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

class DataExplorer:
    def __init__(self, data_dir):
        """
        Initialize Data Explorer
        
        Args:
            data_dir: Path to the main dataset directory
        """
        self.data_dir = data_dir
        self.train_dir = os.path.join(data_dir, 'train')
        self.test_dir = os.path.join(data_dir, 'test')
        self.val_dir = os.path.join(data_dir, 'val')
        
    def count_images(self):
        """Count images in each category"""
        print("="*60)
        print("DATASET STATISTICS")
        print("="*60)
        
        stats = {}
        
        for split in ['train', 'test', 'val']:
            split_dir = os.path.join(self.data_dir, split)
            if not os.path.exists(split_dir):
                print(f"\nWarning: {split} directory not found!")
                continue
                
            print(f"\n{split.upper()} SET:")
            print("-"*40)
            
            normal_dir = os.path.join(split_dir, 'NORMAL')
            pneumonia_dir = os.path.join(split_dir, 'PNEUMONIA')
            
            normal_count = len([f for f in os.listdir(normal_dir) if f.endswith(('.jpeg', '.jpg', '.png'))]) if os.path.exists(normal_dir) else 0
            pneumonia_count = len([f for f in os.listdir(pneumonia_dir) if f.endswith(('.jpeg', '.jpg', '.png'))]) if os.path.exists(pneumonia_dir) else 0
            
            total = normal_count + pneumonia_count
            
            print(f"  Normal:    {normal_count:,} images")
            print(f"  Pneumonia: {pneumonia_count:,} images")
            print(f"  Total:     {total:,} images")
            
            if total > 0:
                print(f"  Balance:   {normal_count/total*100:.1f}% Normal, {pneumonia_count/total*100:.1f}% Pneumonia")
            
            stats[split] = {
                'normal': normal_count,
                'pneumonia': pneumonia_count,
                'total': total
            }
        
        print("\n" + "="*60)
        return stats
    
    def plot_class_distribution(self, stats):
        """Plot class distribution across splits"""
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        for idx, split in enumerate(['train', 'test', 'val']):
            if split not in stats or stats[split]['total'] == 0:
                continue
                
            categories = ['Normal', 'Pneumonia']
            counts = [stats[split]['normal'], stats[split]['pneumonia']]
            colors = ['#2ecc71', '#e74c3c']
            
            axes[idx].bar(categories, counts, color=colors, alpha=0.7, edgecolor='black')
            axes[idx].set_title(f'{split.upper()} Set Distribution', fontsize=14, fontweight='bold')
            axes[idx].set_ylabel('Number of Images', fontsize=12)
            axes[idx].grid(axis='y', alpha=0.3)
            
            # Add count labels on bars
            for i, count in enumerate(counts):
                axes[idx].text(i, count + max(counts)*0.02, str(count), 
                             ha='center', va='bottom', fontsize=11, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig('results/class_distribution.png', dpi=300, bbox_inches='tight')
        print("\n✓ Class distribution plot saved to 'results/class_distribution.png'")
        plt.show()
    
    def visualize_samples(self, num_samples=8):
        """Visualize sample images from both classes"""
        fig, axes = plt.subplots(2, num_samples, figsize=(20, 6))
        
        # Get sample images
        normal_dir = os.path.join(self.train_dir, 'NORMAL')
        pneumonia_dir = os.path.join(self.train_dir, 'PNEUMONIA')
        
        normal_images = [f for f in os.listdir(normal_dir) if f.endswith(('.jpeg', '.jpg', '.png'))][:num_samples]
        pneumonia_images = [f for f in os.listdir(pneumonia_dir) if f.endswith(('.jpeg', '.jpg', '.png'))][:num_samples]
        
        # Plot normal images
        for idx, img_name in enumerate(normal_images):
            img_path = os.path.join(normal_dir, img_name)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            axes[0, idx].imshow(img, cmap='gray')
            axes[0, idx].axis('off')
            if idx == 0:
                axes[0, idx].set_title('NORMAL', fontsize=14, fontweight='bold', color='green')
        
        # Plot pneumonia images
        for idx, img_name in enumerate(pneumonia_images):
            img_path = os.path.join(pneumonia_dir, img_name)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            axes[1, idx].imshow(img, cmap='gray')
            axes[1, idx].axis('off')
            if idx == 0:
                axes[1, idx].set_title('PNEUMONIA', fontsize=14, fontweight='bold', color='red')
        
        plt.suptitle('Sample Chest X-Ray Images', fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()
        plt.savefig('results/sample_images.png', dpi=300, bbox_inches='tight')
        print("✓ Sample images saved to 'results/sample_images.png'")
        plt.show()
    
    def analyze_image_dimensions(self):
        """Analyze image dimensions and sizes"""
        print("\n" + "="*60)
        print("IMAGE DIMENSION ANALYSIS")
        print("="*60)
        
        dimensions = []
        
        # Sample 100 random images from train set
        normal_dir = os.path.join(self.train_dir, 'NORMAL')
        pneumonia_dir = os.path.join(self.train_dir, 'PNEUMONIA')
        
        normal_images = [f for f in os.listdir(normal_dir) if f.endswith(('.jpeg', '.jpg', '.png'))][:50]
        pneumonia_images = [f for f in os.listdir(pneumonia_dir) if f.endswith(('.jpeg', '.jpg', '.png'))][:50]
        
        for img_name in normal_images + pneumonia_images:
            if img_name in normal_images:
                img_path = os.path.join(normal_dir, img_name)
            else:
                img_path = os.path.join(pneumonia_dir, img_name)
            
            img = cv2.imread(img_path)
            if img is not None:
                dimensions.append(img.shape[:2])  # height, width
        
        dimensions = np.array(dimensions)
        
        print(f"\nAnalyzed {len(dimensions)} sample images:")
        print(f"  Average dimensions: {dimensions.mean(axis=0).astype(int)} (H x W)")
        print(f"  Min dimensions:     {dimensions.min(axis=0)} (H x W)")
        print(f"  Max dimensions:     {dimensions.max(axis=0)} (H x W)")
        print(f"  Std dimensions:     {dimensions.std(axis=0).astype(int)} (H x W)")
        
        # Recommendation
        print(f"\n💡 Recommendation: Resize all images to 224x224 for training")
        print("="*60)
        
        return dimensions
    
    def run_full_exploration(self):
        """Run complete data exploration"""
        print("\n🔍 Starting Data Exploration...\n")
        
        # Create results directory
        os.makedirs('results', exist_ok=True)
        
        # Count images
        stats = self.count_images()
        
        # Plot distribution
        self.plot_class_distribution(stats)
        
        # Visualize samples
        self.visualize_samples()
        
        # Analyze dimensions
        self.analyze_image_dimensions()
        
        print("\n✅ Data exploration complete!")
        print("="*60)


if __name__ == "__main__":
    # IMPORTANT: Update this path to your dataset location
    # Example: '/path/to/chest_xray' or 'C:/Users/YourName/chest_xray'
    DATA_DIR = 'chest_xray'
    
    print("""
    ╔══════════════════════════════════════════════════════════╗
    ║   Chest X-Ray Pneumonia Detection - Data Exploration    ║
    ╚══════════════════════════════════════════════════════════╝
    """)
    
    # Check if data directory exists
    if not os.path.exists(DATA_DIR):
        print(f"\n❌ ERROR: Data directory not found at '{DATA_DIR}'")
        print("\n📝 Instructions:")
        print("1. Download the 'Chest X-Ray Images (Pneumonia)' dataset from Kaggle")
        print("2. Extract the dataset")
        print("3. Update the DATA_DIR variable in this script to point to your dataset")
        print("\nDataset structure should be:")
        print("  chest_xray/")
        print("    ├── train/")
        print("    │   ├── NORMAL/")
        print("    │   └── PNEUMONIA/")
        print("    ├── test/")
        print("    │   ├── NORMAL/")
        print("    │   └── PNEUMONIA/")
        print("    └── val/")
        print("        ├── NORMAL/")
        print("        └── PNEUMONIA/")
    else:
        explorer = DataExplorer(DATA_DIR)
        explorer.run_full_exploration()
