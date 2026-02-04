"""
SIMPLE DEMO - Chest X-Ray Pneumonia Detection
Run this AFTER training the model to quickly test it!
"""

import os
import sys

def check_requirements():
    """Check if all requirements are met"""
    print("="*70)
    print("CHECKING REQUIREMENTS")
    print("="*70)
    
    # Check if model exists
    if not os.path.exists('models/best_model.h5'):
        print("\n❌ ERROR: Trained model not found!")
        print("\n📝 Please train the model first:")
        print("   python src/04_train_model.py")
        return False
    
    print("✓ Model found: models/best_model.h5")
    
    # Check if TensorFlow is installed
    try:
        import tensorflow as tf
        print(f"✓ TensorFlow {tf.__version__} installed")
    except ImportError:
        print("❌ TensorFlow not installed!")
        print("   Run: pip install tensorflow==2.15.0")
        return False
    
    # Check other dependencies
    try:
        import numpy
        import matplotlib
        from PIL import Image
        print("✓ All dependencies installed")
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        print("   Run: pip install -r requirements.txt")
        return False
    
    print("="*70)
    return True

def demo_prediction():
    """Run a simple prediction demo"""
    import numpy as np
    from tensorflow.keras.models import load_model
    from tensorflow.keras.preprocessing import image
    import matplotlib.pyplot as plt
    from PIL import Image as PILImage
    
    print("\n" + "="*70)
    print("CHEST X-RAY PNEUMONIA DETECTOR - DEMO")
    print("="*70)
    
    # Load model
    print("\n📦 Loading trained model...")
    model = load_model('models/best_model.h5')
    print("✓ Model loaded successfully!")
    
    # Get image path from user
    print("\n📁 Enter the path to a chest X-ray image:")
    print("   (or press Enter to skip)")
    img_path = input("   Path: ").strip()
    
    if not img_path:
        print("\n⚠️  No image provided. Demo ended.")
        return
    
    if not os.path.exists(img_path):
        print(f"\n❌ File not found: {img_path}")
        return
    
    # Load and preprocess image
    print(f"\n🔍 Analyzing: {os.path.basename(img_path)}...")
    
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    # Make prediction
    prediction = model.predict(img_array, verbose=0)[0][0]
    
    # Determine diagnosis
    if prediction > 0.5:
        diagnosis = "PNEUMONIA"
        confidence = prediction
        color = 'red'
        emoji = "⚠️"
    else:
        diagnosis = "NORMAL"
        confidence = 1 - prediction
        color = 'green'
        emoji = "✓"
    
    # Display results
    print("\n" + "="*70)
    print("DIAGNOSIS RESULT")
    print("="*70)
    print(f"\n{emoji}  Diagnosis: {diagnosis}")
    print(f"   Confidence: {confidence*100:.2f}%")
    print(f"   Raw Score: {prediction:.4f}")
    
    if diagnosis == "PNEUMONIA":
        print("\n⚠️  PNEUMONIA DETECTED")
        print("   Recommendation: Consult a healthcare professional immediately.")
    else:
        print("\n✓ NORMAL")
        print("   No signs of pneumonia detected.")
    
    print("="*70)
    
    # Visualize result
    print("\n📊 Generating visualization...")
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Show original image
    original_img = PILImage.open(img_path)
    axes[0].imshow(original_img, cmap='gray')
    axes[0].set_title('Chest X-Ray Image', fontsize=12, fontweight='bold')
    axes[0].axis('off')
    
    # Show prediction
    axes[1].barh(['Prediction'], [confidence], color=color, alpha=0.7, height=0.5)
    axes[1].set_xlim([0, 1])
    axes[1].set_xlabel('Confidence', fontsize=11, fontweight='bold')
    axes[1].set_title(f'Diagnosis: {diagnosis}', fontsize=12, fontweight='bold', color=color)
    axes[1].text(confidence + 0.02, 0, f'{confidence*100:.1f}%', 
                va='center', fontsize=11, fontweight='bold')
    axes[1].grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    
    # Save visualization
    os.makedirs('sample_predictions', exist_ok=True)
    output_path = 'sample_predictions/demo_result.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✓ Visualization saved to: {output_path}")
    
    plt.show()
    
    # Ask if user wants to predict another
    print("\n" + "="*70)
    print("Want to predict another X-ray? Run this script again!")
    print("="*70)

def main():
    """Main demo function"""
    print("""
    ╔══════════════════════════════════════════════════════════════════╗
    ║      CHEST X-RAY PNEUMONIA DETECTION - SIMPLE DEMO              ║
    ╚══════════════════════════════════════════════════════════════════╝
    """)
    
    # Check requirements
    if not check_requirements():
        print("\n⚠️  Please fix the issues above and try again.")
        sys.exit(1)
    
    # Run demo
    demo_prediction()
    
    print("\n💡 TIP: For a better experience, try the web interface:")
    print("   streamlit run app.py")
    print("\n👋 Thanks for using Chest X-Ray Pneumonia Detector!")

if __name__ == "__main__":
    main()
