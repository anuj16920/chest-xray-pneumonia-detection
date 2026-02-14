# 🩺 Chest X-Ray Pneumonia Detection

An AI-powered deep learning system that automatically detects pneumonia from chest X-ray images using Transfer Learning with ResNet50.



## 🎯 Overview

This project implements a deep learning model for pneumonia detection from chest X-ray images. The model uses Transfer Learning with ResNet50 (pre-trained on ImageNet) and achieves high accuracy in distinguishing between normal and pneumonia cases.

**Key Highlights:**
- 🤖 Transfer Learning with ResNet50
- 📊 Comprehensive data preprocessing and augmentation
- 📈 Detailed model evaluation with metrics
- 🖥️ Interactive web interface using Streamlit
- 🔮 Batch and single image prediction capabilities

## ✨ Features

- **Automated Detection**: Quickly analyze chest X-rays for signs of pneumonia
- **High Accuracy**: Leverages state-of-the-art deep learning architecture
- **Data Augmentation**: Robust training with image transformations
- **Comprehensive Evaluation**: Confusion matrix, ROC curve, classification report
- **Easy to Use**: Simple command-line interface and web app
- **Batch Processing**: Analyze multiple X-rays at once
- **Visualization**: Clear visual results with confidence scores

## 📦 Dataset

This project uses the **Chest X-Ray Images (Pneumonia)** dataset from Kaggle:
- **Source**: [Kaggle - Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)
- **Total Images**: ~5,800 chest X-ray images
- **Classes**: NORMAL and PNEUMONIA
- **Format**: JPEG images

### Dataset Structure
```
chest_xray/
├── train/
│   ├── NORMAL/      (~1,300 images)
│   └── PNEUMONIA/   (~3,900 images)
├── test/
│   ├── NORMAL/      (~234 images)
│   └── PNEUMONIA/   (~390 images)
└── val/
    ├── NORMAL/      (~8 images)
    └── PNEUMONIA/   (~8 images)
```

## 🚀 Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager
- (Optional) CUDA for GPU acceleration

### Step 1: Clone the Repository
```bash
git clone <repository-url>
cd chest-xray-pneumonia-detection
```

### Step 2: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 3: Download Dataset
1. Go to [Kaggle Dataset](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)
2. Download the dataset
3. Extract it to `data/chest_xray/` directory

Your directory should look like:
```
chest-xray-pneumonia-detection/
├── data/
│   └── chest_xray/
│       ├── train/
│       ├── test/
│       └── val/
```

## 📁 Project Structure

```
chest-xray-pneumonia-detection/
│
├── data/                          # Dataset directory
│   └── chest_xray/               # Kaggle dataset
│
├── src/                          # Source code
│   ├── 01_data_exploration.py   # Data analysis and visualization
│   ├── 02_data_preprocessing.py # Data preprocessing and augmentation
│   ├── 03_model_architecture.py # Model building
│   ├── 04_train_model.py        # Main training script ⭐
│   ├── 05_evaluate_model.py     # Model evaluation
│   └── 06_predict.py            # Prediction on new images
│
├── models/                       # Saved models
│   ├── best_model.h5            # Best model (highest val accuracy)
│   └── final_model.h5           # Final model after training
│
├── results/                      # Training results and visualizations
│   ├── training_history.png
│   ├── confusion_matrix.png
│   ├── roc_curve.png
│   └── classification_report.txt
│
├── sample_predictions/           # Sample prediction outputs
│
├── app.py                        # Streamlit web interface
├── requirements.txt              # Python dependencies
└── README.md                     # This file
```

## 🎮 Usage

### Option 1: Complete Training Pipeline (Recommended)

Run the main training script that handles everything:

```bash
python src/04_train_model.py
```

This will:
1. ✅ Load and preprocess the dataset
2. ✅ Build the model with Transfer Learning
3. ✅ Train the model (25 epochs by default)
4. ✅ Save the best model
5. ✅ Generate training visualizations

**Training Time**: ~30-60 minutes (depending on hardware)

### Option 2: Step-by-Step Approach

#### Step 1: Data Exploration
```bash
python src/01_data_exploration.py
```
Analyzes the dataset and creates visualizations.

#### Step 2: Data Preprocessing
```bash
python src/02_data_preprocessing.py
```
Sets up data generators with augmentation.

#### Step 3: Model Architecture
```bash
python src/03_model_architecture.py
```
Defines and compiles the model.

#### Step 4: Train Model
```bash
python src/04_train_model.py
```
Trains the complete model.

#### Step 5: Evaluate Model
```bash
python src/05_evaluate_model.py
```
Evaluates trained model and generates detailed metrics.

#### Step 6: Make Predictions
```bash
python src/06_predict.py
```
Predict on new X-ray images.

### Option 3: Web Interface

Launch the Streamlit web app:

```bash
streamlit run app.py
```

Then open your browser to `http://localhost:8501`

## 🏗️ Model Architecture

### Transfer Learning Approach

The model uses **ResNet50** pre-trained on ImageNet as the base:

```
Input (224x224x3)
    ↓
ResNet50 Base (Frozen)
    ↓
Global Average Pooling
    ↓
Dense (512, ReLU)
    ↓
Batch Normalization
    ↓
Dropout (0.5)
    ↓
Dense (256, ReLU)
    ↓
Batch Normalization
    ↓
Dropout (0.3)
    ↓
Dense (1, Sigmoid)
```

### Training Configuration
- **Optimizer**: Adam (lr=0.0001)
- **Loss Function**: Binary Crossentropy
- **Metrics**: Accuracy, Precision, Recall, AUC
- **Batch Size**: 32
- **Epochs**: 25 (with early stopping)
- **Data Augmentation**: Rotation, shift, zoom, flip

### Callbacks
- **ModelCheckpoint**: Saves best model based on validation accuracy
- **EarlyStopping**: Stops training if validation loss doesn't improve (patience=10)
- **ReduceLROnPlateau**: Reduces learning rate on plateau (patience=5)

## 📊 Results

Expected performance metrics (will vary based on training):

| Metric      | Score    |
|-------------|----------|
| Accuracy    | ~92-95%  |
| Precision   | ~90-93%  |
| Recall      | ~94-97%  |
| F1-Score    | ~92-95%  |
| AUC-ROC     | ~95-98%  |

### Generated Outputs

After training and evaluation, you'll get:

1. **Training History Plot** (`results/training_history.png`)
   - Accuracy curves (train vs validation)
   - Loss curves
   - Precision and recall curves

2. **Confusion Matrix** (`results/confusion_matrix.png`)
   - True positives, false positives
   - True negatives, false negatives

3. **ROC Curve** (`results/roc_curve.png`)
   - Area Under Curve (AUC)
   - Model performance visualization

4. **Classification Report** (`results/classification_report.txt`)
   - Precision, recall, F1-score per class
   - Support values

## 🖥️ Web Interface

The Streamlit web app provides an easy-to-use interface:

**Features:**
- 📤 Drag-and-drop X-ray image upload
- 🔍 Instant AI-powered diagnosis
- 📊 Confidence score visualization
- 📥 Downloadable diagnosis report
- 📱 Mobile-friendly responsive design

**To run:**
```bash
streamlit run app.py
```

## 🔮 Making Predictions

### Single Image Prediction
```python
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

# Load model
model = load_model('models/best_model.h5')

# Load and preprocess image
img = image.load_img('xray.jpg', target_size=(224, 224))
img_array = image.img_to_array(img) / 255.0
img_array = np.expand_dims(img_array, axis=0)

# Predict
prediction = model.predict(img_array)[0][0]
diagnosis = "PNEUMONIA" if prediction > 0.5 else "NORMAL"
confidence = prediction if prediction > 0.5 else 1 - prediction

print(f"Diagnosis: {diagnosis}")
print(f"Confidence: {confidence*100:.2f}%")
```

### Batch Prediction
```bash
python src/06_predict.py
# Select option 2 for batch mode
```

## ⚙️ Configuration

You can modify training parameters in `src/04_train_model.py`:

```python
DATA_DIR = 'data/chest_xray'  # Dataset location
IMG_SIZE = (224, 224)          # Image dimensions
BATCH_SIZE = 32                # Batch size
EPOCHS = 25                    # Number of epochs
```

## 🐛 Troubleshooting

### Common Issues

**1. GPU Out of Memory**
- Reduce `BATCH_SIZE` to 16 or 8
- Use CPU instead: `os.environ['CUDA_VISIBLE_DEVICES'] = '-1'`

**2. Dataset Not Found**
- Ensure dataset is extracted to `data/chest_xray/`
- Update `DATA_DIR` path in scripts

**3. Module Not Found**
- Install missing packages: `pip install -r requirements.txt`
- Use virtual environment

**4. Slow Training**
- Use Google Colab with free GPU
- Reduce number of epochs
- Use smaller batch size

## 📈 Future Improvements

- [ ] Multi-class classification (bacterial vs viral pneumonia)
- [ ] Grad-CAM visualization for interpretability
- [ ] Mobile app deployment
- [ ] API endpoint for integration
- [ ] Ensemble model for better accuracy
- [ ] Support for other chest conditions

## ⚠️ Disclaimer

**IMPORTANT**: This project is for **educational and research purposes only**.

- This tool should **NOT** be used as a substitute for professional medical diagnosis
- Always consult qualified healthcare professionals for medical advice
- The model's predictions are probabilistic and may contain errors
- Clinical validation is required before any real-world medical application

## 📝 License

This project is licensed under the MIT License.

## 🙏 Acknowledgments

- Dataset: [Paul Mooney (Kaggle)](https://www.kaggle.com/paultimothymooney)
- ResNet50 Architecture: [Microsoft Research](https://arxiv.org/abs/1512.03385)
- TensorFlow/Keras: Google Brain Team

## 👨‍💻 Author
LOMTE ANUJ

Built with ❤️ using Python, TensorFlow, and Deep Learning

---

**Need Help?** Open an issue or contact the maintainer.

**Happy Training! 🚀**
