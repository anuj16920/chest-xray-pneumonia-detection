"""
Chest X-Ray Pneumonia Detection - Model Evaluation
Detailed evaluation with confusion matrix, ROC curve, and metrics
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import (
    confusion_matrix, classification_report, 
    roc_curve, auc, precision_recall_curve
)
import pandas as pd

class ModelEvaluator:
    def __init__(self, model_path, data_dir, img_size=(224, 224), batch_size=32):
        """
        Initialize Model Evaluator
        
        Args:
            model_path: Path to saved model
            data_dir: Path to dataset directory
            img_size: Image size
            batch_size: Batch size
        """
        self.model_path = model_path
        self.data_dir = data_dir
        self.img_size = img_size
        self.batch_size = batch_size
        self.test_dir = os.path.join(data_dir, 'test')
        
        # Load model
        print(f"\n📦 Loading model from '{model_path}'...")
        self.model = load_model(model_path)
        print("✓ Model loaded successfully!")
        
        # Prepare test generator
        test_datagen = ImageDataGenerator(rescale=1./255)
        self.test_generator = test_datagen.flow_from_directory(
            self.test_dir,
            target_size=self.img_size,
            batch_size=self.batch_size,
            class_mode='binary',
            shuffle=False
        )
        
        self.class_names = ['NORMAL', 'PNEUMONIA']
        
    def get_predictions(self):
        """Get model predictions on test set"""
        print("\n🔮 Generating predictions on test set...")
        
        # Get predictions
        predictions = self.model.predict(self.test_generator, verbose=1)
        predicted_classes = (predictions > 0.5).astype(int).flatten()
        
        # Get true labels
        true_classes = self.test_generator.classes
        
        print(f"✓ Generated {len(predictions)} predictions")
        
        return predictions, predicted_classes, true_classes
    
    def plot_confusion_matrix(self, true_labels, predicted_labels):
        """Plot confusion matrix"""
        print("\n📊 Creating confusion matrix...")
        
        cm = confusion_matrix(true_labels, predicted_labels)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=self.class_names,
                   yticklabels=self.class_names,
                   cbar_kws={'label': 'Count'},
                   annot_kws={'size': 16, 'weight': 'bold'})
        
        plt.title('Confusion Matrix', fontsize=16, fontweight='bold', pad=20)
        plt.xlabel('Predicted Label', fontsize=14, fontweight='bold')
        plt.ylabel('True Label', fontsize=14, fontweight='bold')
        
        # Add accuracy text
        accuracy = (cm[0,0] + cm[1,1]) / cm.sum()
        plt.text(1, 2.3, f'Accuracy: {accuracy:.2%}', 
                fontsize=14, fontweight='bold', ha='center')
        
        plt.tight_layout()
        plt.savefig('results/confusion_matrix.png', dpi=300, bbox_inches='tight')
        print("✓ Confusion matrix saved to 'results/confusion_matrix.png'")
        plt.show()
        
        return cm
    
    def plot_roc_curve(self, true_labels, predictions):
        """Plot ROC curve"""
        print("\n📈 Creating ROC curve...")
        
        # Calculate ROC curve
        fpr, tpr, thresholds = roc_curve(true_labels, predictions)
        roc_auc = auc(fpr, tpr)
        
        plt.figure(figsize=(10, 8))
        plt.plot(fpr, tpr, color='#e74c3c', lw=3, 
                label=f'ROC Curve (AUC = {roc_auc:.4f})')
        plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--', 
                label='Random Classifier')
        
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=14, fontweight='bold')
        plt.ylabel('True Positive Rate', fontsize=14, fontweight='bold')
        plt.title('Receiver Operating Characteristic (ROC) Curve', 
                 fontsize=16, fontweight='bold')
        plt.legend(loc="lower right", fontsize=12)
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('results/roc_curve.png', dpi=300, bbox_inches='tight')
        print("✓ ROC curve saved to 'results/roc_curve.png'")
        plt.show()
        
        return roc_auc
    
    def plot_precision_recall_curve(self, true_labels, predictions):
        """Plot Precision-Recall curve"""
        print("\n📉 Creating Precision-Recall curve...")
        
        precision, recall, thresholds = precision_recall_curve(true_labels, predictions)
        pr_auc = auc(recall, precision)
        
        plt.figure(figsize=(10, 8))
        plt.plot(recall, precision, color='#3498db', lw=3,
                label=f'PR Curve (AUC = {pr_auc:.4f})')
        
        plt.xlabel('Recall', fontsize=14, fontweight='bold')
        plt.ylabel('Precision', fontsize=14, fontweight='bold')
        plt.title('Precision-Recall Curve', fontsize=16, fontweight='bold')
        plt.legend(loc="lower left", fontsize=12)
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('results/precision_recall_curve.png', dpi=300, bbox_inches='tight')
        print("✓ Precision-Recall curve saved to 'results/precision_recall_curve.png'")
        plt.show()
        
        return pr_auc
    
    def generate_classification_report(self, true_labels, predicted_labels):
        """Generate and save classification report"""
        print("\n📋 Generating classification report...")
        
        report = classification_report(
            true_labels, 
            predicted_labels,
            target_names=self.class_names,
            digits=4
        )
        
        print("\n" + "="*60)
        print("CLASSIFICATION REPORT")
        print("="*60)
        print(report)
        print("="*60)
        
        # Save report
        with open('results/classification_report.txt', 'w') as f:
            f.write("="*60 + "\n")
            f.write("CHEST X-RAY PNEUMONIA DETECTION - CLASSIFICATION REPORT\n")
            f.write("="*60 + "\n\n")
            f.write(report)
            f.write("\n" + "="*60 + "\n")
        
        print("\n✓ Classification report saved to 'results/classification_report.txt'")
        
        return report
    
    def analyze_misclassifications(self, predictions, predicted_classes, true_classes):
        """Analyze misclassified samples"""
        print("\n🔍 Analyzing misclassifications...")
        
        # Find misclassified indices
        misclassified_idx = np.where(predicted_classes != true_classes)[0]
        
        false_positives = []
        false_negatives = []
        
        for idx in misclassified_idx:
            if true_classes[idx] == 0 and predicted_classes[idx] == 1:
                false_positives.append((idx, predictions[idx][0]))
            elif true_classes[idx] == 1 and predicted_classes[idx] == 0:
                false_negatives.append((idx, predictions[idx][0]))
        
        print(f"\n📊 Misclassification Analysis:")
        print(f"   Total misclassified: {len(misclassified_idx)}")
        print(f"   False Positives (Normal → Pneumonia): {len(false_positives)}")
        print(f"   False Negatives (Pneumonia → Normal): {len(false_negatives)}")
        
        # Visualize some misclassified samples
        self.visualize_misclassified(false_positives[:4], false_negatives[:4])
        
    def visualize_misclassified(self, false_positives, false_negatives):
        """Visualize misclassified samples"""
        if len(false_positives) == 0 and len(false_negatives) == 0:
            print("✓ No misclassifications to visualize!")
            return
        
        print("\n📸 Visualizing misclassified samples...")
        
        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        
        # Get file paths
        filepaths = self.test_generator.filepaths
        
        # Plot false positives
        for idx, (img_idx, pred_prob) in enumerate(false_positives[:4]):
            img_path = filepaths[img_idx]
            img = plt.imread(img_path)
            
            axes[0, idx].imshow(img, cmap='gray')
            axes[0, idx].set_title(f'FP: Predicted Pneumonia\nConfidence: {pred_prob:.2%}', 
                                  fontsize=10, color='red', fontweight='bold')
            axes[0, idx].axis('off')
        
        # Plot false negatives
        for idx, (img_idx, pred_prob) in enumerate(false_negatives[:4]):
            img_path = filepaths[img_idx]
            img = plt.imread(img_path)
            
            axes[1, idx].imshow(img, cmap='gray')
            axes[1, idx].set_title(f'FN: Predicted Normal\nConfidence: {1-pred_prob:.2%}', 
                                  fontsize=10, color='red', fontweight='bold')
            axes[1, idx].axis('off')
        
        plt.suptitle('Misclassified Samples', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig('results/misclassified_samples.png', dpi=300, bbox_inches='tight')
        print("✓ Misclassified samples saved to 'results/misclassified_samples.png'")
        plt.show()
    
    def run_complete_evaluation(self):
        """Run complete evaluation pipeline"""
        print("""
        ╔══════════════════════════════════════════════════════════════════╗
        ║     Chest X-Ray Pneumonia Detection - MODEL EVALUATION          ║
        ╚══════════════════════════════════════════════════════════════════╝
        """)
        
        os.makedirs('results', exist_ok=True)
        
        # Get predictions
        predictions, predicted_classes, true_classes = self.get_predictions()
        
        # Confusion matrix
        cm = self.plot_confusion_matrix(true_classes, predicted_classes)
        
        # ROC curve
        roc_auc = self.plot_roc_curve(true_classes, predictions)
        
        # Precision-Recall curve
        pr_auc = self.plot_precision_recall_curve(true_classes, predictions)
        
        # Classification report
        report = self.generate_classification_report(true_classes, predicted_classes)
        
        # Analyze misclassifications
        self.analyze_misclassifications(predictions, predicted_classes, true_classes)
        
        print("\n" + "="*70)
        print("✅ EVALUATION COMPLETED!")
        print("="*70)
        print("\n📁 Generated Files:")
        print("   • results/confusion_matrix.png")
        print("   • results/roc_curve.png")
        print("   • results/precision_recall_curve.png")
        print("   • results/classification_report.txt")
        print("   • results/misclassified_samples.png")
        print("="*70)


if __name__ == "__main__":
    # Configuration
    MODEL_PATH = 'models/best_model.h5'
    DATA_DIR = 'chest_xray'
    IMG_SIZE = (224, 224)
    BATCH_SIZE = 32
    
    # Check if model exists
    if not os.path.exists(MODEL_PATH):
        print(f"\n❌ ERROR: Model not found at '{MODEL_PATH}'")
        print("Please train the model first by running '04_train_model.py'")
    elif not os.path.exists(DATA_DIR):
        print(f"\n❌ ERROR: Dataset not found at '{DATA_DIR}'")
        print("Please update DATA_DIR to point to your dataset location")
    else:
        evaluator = ModelEvaluator(MODEL_PATH, DATA_DIR, IMG_SIZE, BATCH_SIZE)
        evaluator.run_complete_evaluation()
