"""
Chest X-Ray Pneumonia Detection - Prediction on New Images
Use this script to diagnose new chest X-ray images
"""

import os
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt
from PIL import Image

class XRayPredictor:
    def __init__(self, model_path='models/best_model.h5', img_size=(224, 224)):
        """
        Initialize X-Ray Predictor
        
        Args:
            model_path: Path to trained model
            img_size: Target image size
        """
        self.model_path = model_path
        self.img_size = img_size
        
        print(f"\n📦 Loading model from '{model_path}'...")
        self.model = load_model(model_path)
        print("✓ Model loaded successfully!\n")
        
        self.class_names = {0: 'NORMAL', 1: 'PNEUMONIA'}
        
    def preprocess_image(self, img_path):
        """
        Preprocess image for prediction
        
        Args:
            img_path: Path to image file
        """
        # Load image
        img = image.load_img(img_path, target_size=self.img_size)
        
        # Convert to array and normalize
        img_array = image.img_to_array(img)
        img_array = img_array / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        return img_array
    
    def predict_single_image(self, img_path):
        """
        Predict on a single image
        
        Args:
            img_path: Path to X-ray image
        """
        # Preprocess
        img_array = self.preprocess_image(img_path)
        
        # Predict
        prediction = self.model.predict(img_array, verbose=0)[0][0]
        
        # Get class and confidence
        predicted_class = 1 if prediction > 0.5 else 0
        confidence = prediction if predicted_class == 1 else 1 - prediction
        
        diagnosis = self.class_names[predicted_class]
        
        return diagnosis, confidence, prediction
    
    def predict_and_visualize(self, img_path, save_path=None):
        """
        Predict and visualize result
        
        Args:
            img_path: Path to X-ray image
            save_path: Path to save visualization (optional)
        """
        # Get prediction
        diagnosis, confidence, raw_prediction = self.predict_single_image(img_path)
        
        # Load original image for display
        original_img = cv2.imread(img_path)
        original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
        
        # Create visualization
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # Display image
        axes[0].imshow(original_img, cmap='gray')
        axes[0].set_title('Chest X-Ray Image', fontsize=14, fontweight='bold')
        axes[0].axis('off')
        
        # Display prediction
        color = '#e74c3c' if diagnosis == 'PNEUMONIA' else '#2ecc71'
        axes[1].barh(['Prediction'], [confidence], color=color, alpha=0.7, height=0.4)
        axes[1].set_xlim([0, 1])
        axes[1].set_xlabel('Confidence', fontsize=12, fontweight='bold')
        axes[1].set_title(f'Diagnosis: {diagnosis}', fontsize=14, fontweight='bold', color=color)
        axes[1].grid(axis='x', alpha=0.3)
        
        # Add confidence text
        axes[1].text(confidence + 0.02, 0, f'{confidence*100:.2f}%', 
                    va='center', fontsize=12, fontweight='bold')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Visualization saved to '{save_path}'")
        
        plt.show()
        
        return diagnosis, confidence
    
    def predict_batch(self, image_folder, output_folder='sample_predictions'):
        """
        Predict on multiple images in a folder
        
        Args:
            image_folder: Folder containing X-ray images
            output_folder: Folder to save predictions
        """
        print(f"\n🔮 Running batch prediction on images in '{image_folder}'...")
        print("="*70)
        
        os.makedirs(output_folder, exist_ok=True)
        
        # Get all image files
        image_files = [f for f in os.listdir(image_folder) 
                      if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        if len(image_files) == 0:
            print(f"❌ No images found in '{image_folder}'")
            return
        
        results = []
        
        for idx, img_file in enumerate(image_files, 1):
            img_path = os.path.join(image_folder, img_file)
            
            # Predict
            diagnosis, confidence, raw_pred = self.predict_single_image(img_path)
            
            # Store result
            results.append({
                'filename': img_file,
                'diagnosis': diagnosis,
                'confidence': f'{confidence*100:.2f}%',
                'raw_score': f'{raw_pred:.4f}'
            })
            
            # Print result
            color_code = '\033[91m' if diagnosis == 'PNEUMONIA' else '\033[92m'
            reset_code = '\033[0m'
            print(f"{idx}. {img_file:30s} → {color_code}{diagnosis:10s}{reset_code} (Confidence: {confidence*100:.2f}%)")
        
        # Save results to CSV
        import pandas as pd
        df = pd.DataFrame(results)
        csv_path = os.path.join(output_folder, 'batch_predictions.csv')
        df.to_csv(csv_path, index=False)
        
        print("\n" + "="*70)
        print(f"✅ Batch prediction completed!")
        print(f"✓ Processed {len(image_files)} images")
        print(f"✓ Results saved to '{csv_path}'")
        print("="*70)
        
        return results
    
    def interactive_prediction(self):
        """
        Interactive prediction mode - user provides image path
        """
        print("""
        ╔══════════════════════════════════════════════════════════════════╗
        ║         Chest X-Ray Pneumonia Detection - PREDICTION            ║
        ╚══════════════════════════════════════════════════════════════════╝
        """)
        
        print("\n🩺 Interactive Prediction Mode")
        print("Enter the path to an X-ray image, or 'quit' to exit.\n")
        
        while True:
            img_path = input("📁 X-Ray image path: ").strip()
            
            if img_path.lower() in ['quit', 'exit', 'q']:
                print("\n👋 Exiting prediction mode. Goodbye!")
                break
            
            if not os.path.exists(img_path):
                print(f"❌ File not found: {img_path}\n")
                continue
            
            print("\n🔮 Analyzing X-ray...")
            diagnosis, confidence = self.predict_and_visualize(
                img_path,
                save_path=f'sample_predictions/prediction_{os.path.basename(img_path)}'
            )
            
            print("\n" + "="*70)
            print("DIAGNOSIS RESULT")
            print("="*70)
            print(f"   Diagnosis:  {diagnosis}")
            print(f"   Confidence: {confidence*100:.2f}%")
            
            if diagnosis == 'PNEUMONIA':
                print("\n⚠️  PNEUMONIA DETECTED")
                print("   Recommendation: Consult a healthcare professional immediately.")
            else:
                print("\n✓ NORMAL")
                print("   No signs of pneumonia detected in this X-ray.")
            
            print("="*70)
            print()


if __name__ == "__main__":
    # Configuration
    MODEL_PATH = 'models/fine_tuned_model.h5'  # Use fine-tuned model
    
    # Fallback to best_model.h5 if fine-tuned doesn't exist
    if not os.path.exists(MODEL_PATH):
        MODEL_PATH = 'models/best_model.h5'
        print("⚠️  Fine-tuned model not found, using best_model.h5")
    
    if not os.path.exists(MODEL_PATH):
        print(f"\n❌ ERROR: Model not found at '{MODEL_PATH}'")
        print("Please train the model first by running '04_train_model.py'")
    else:
        # Initialize predictor
        predictor = XRayPredictor(MODEL_PATH)
        
        # Choose mode
        print("\n📋 Select Mode:")
        print("1. Interactive mode (enter image paths one by one)")
        print("2. Batch mode (predict on all images in a folder)")
        print("3. Single prediction")
        
        mode = input("\nEnter choice (1/2/3): ").strip()
        
        if mode == '1':
            predictor.interactive_prediction()
            
        elif mode == '2':
            folder_path = input("\n📁 Enter folder path containing X-ray images: ").strip()
            if os.path.exists(folder_path):
                predictor.predict_batch(folder_path)
            else:
                print(f"❌ Folder not found: {folder_path}")
                
        elif mode == '3':
            img_path = input("\n📁 Enter X-ray image path: ").strip()
            if os.path.exists(img_path):
                os.makedirs('sample_predictions', exist_ok=True)
                diagnosis, confidence = predictor.predict_and_visualize(
                    img_path,
                    save_path='sample_predictions/single_prediction.png'
                )
                
                print("\n" + "="*70)
                print("DIAGNOSIS RESULT")
                print("="*70)
                print(f"   Diagnosis:  {diagnosis}")
                print(f"   Confidence: {confidence*100:.2f}%")
                
                if diagnosis == 'PNEUMONIA':
                    print("\n⚠️  PNEUMONIA DETECTED")
                else:
                    print("\n✓ NORMAL - No pneumonia detected")
                print("="*70)
            else:
                print(f"❌ File not found: {img_path}")
        else:
            print("❌ Invalid choice")
