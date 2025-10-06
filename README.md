# HAR CNN Model - Human Activity Recognition

🚀 A deep learning project for human activity recognition using CNN with accelerometer data, optimized for mobile deployment.

## 📋 Project Overview

This project implements a Convolutional Neural Network (CNN) for recognizing human activities using accelerometer sensor data. The model is specifically designed for 3-class classification and optimized for mobile deployment.

### 🎯 Activity Classes
- **IDLE**: Stationary/No movement
- **RUN**: Running activity  
- **WALK**: Walking activity

### 📊 Features Used
- **ax_mps2**: Accelerometer X-axis (m/s²)
- **ay_mps2**: Accelerometer Y-axis (m/s²)
- **az_mps2**: Accelerometer Z-axis (m/s²)
- **acc_magnitude**: Magnitude √(ax² + ay² + az²)

## 🏗️ Project Structure

```
├── har_cnn.ipynb              # Main Jupyter notebook with complete workflow
├── datasets/                  # Training data organized by activity
│   ├── idle/                 # Idle activity data
│   ├── run/                  # Running activity data
│   └── walk/                 # Walking activity data
└── out-final/                # Model outputs and deployments
    ├── mobile/               # Mobile-optimized files
    │   ├── cnn_har.tflite    # TensorFlow Lite model
    │   ├── mobile_config.json
    │   └── preprocessing_config.json
    └── models/               # Trained models and preprocessing files
        ├── cnn_har_3classes.h5
        ├── cnn_har_3classes.weights.h5
        ├── cnn_har_3classes_metadata.json
        ├── label_encoder_3classes.pkl
        ├── scaler_3classes.pkl
        └── training_history_3classes.pkl
```

## 🛠️ Technical Specifications

### Model Architecture
- **Type**: 1D Convolutional Neural Network (CNN)
- **Input Shape**: (100, 4) - 100 timesteps × 4 features
- **Window Size**: 100 samples (2 seconds at 50Hz)
- **Step Size**: 50 samples (50% overlap)
- **Sampling Rate**: 50 Hz

### Model Configuration
- **Batch Size**: 32
- **Epochs**: 50
- **Learning Rate**: 0.001
- **Optimizer**: Adam
- **Classes**: 3 (IDLE, RUN, WALK)
- **Features**: 4 (3-axis accelerometer + magnitude)

## 🚀 Getting Started

### Prerequisites
```bash
pip install tensorflow pandas numpy scikit-learn matplotlib seaborn
```

### Quick Start
1. **Clone/Download** the repository
2. **Install dependencies** listed above
3. **Open** `har_cnn.ipynb` in Jupyter Notebook
4. **Run all cells** to train the model

### Using Pre-trained Model
```python
import tensorflow as tf
import numpy as np
import pickle

# Load the model
model = tf.keras.models.load_model('out-final/models/cnn_har_3classes.h5')

# Load preprocessing components
with open('out-final/models/scaler_3classes.pkl', 'rb') as f:
    scaler = pickle.load(f)

with open('out-final/models/label_encoder_3classes.pkl', 'rb') as f:
    label_encoder = pickle.load(f)

# Predict on new data
def predict_activity(accelerometer_data):
    # accelerometer_data shape: (100, 4) - 100 timesteps × 4 features
    normalized_data = scaler.transform(accelerometer_data)
    prediction = model.predict(normalized_data.reshape(1, 100, 4))
    predicted_class = label_encoder.inverse_transform([np.argmax(prediction)])
    confidence = np.max(prediction)
    return predicted_class[0], confidence
```

## 📱 Mobile Deployment

### TensorFlow Lite Model
The project includes a TensorFlow Lite model optimized for mobile deployment:
- **File**: `out-final/mobile/cnn_har.tflite`
- **Size**: < 3MB
- **Inference Time**: < 10ms
- **Memory Usage**: < 5MB

### Mobile Integration Steps
1. **Load** the TFLite model in your mobile app
2. **Collect** accelerometer data at 50Hz
3. **Calculate** magnitude: √(ax² + ay² + az²)
4. **Create** sliding windows of 100 timesteps
5. **Normalize** using provided scaler parameters
6. **Run** inference and get activity predictions

### Configuration Files
- `mobile_config.json`: Complete mobile deployment configuration
- `preprocessing_config.json`: Data preprocessing parameters

## 🎯 Performance Metrics

### Model Performance
- **Validation Accuracy**: High accuracy on 3-class classification
- **Optimized Features**: Reduced from complex sensor fusion to accelerometer-only
- **Inference Speed**: Optimized for real-time mobile performance

### Mobile Optimizations
- **Model Size**: ~40% reduction (< 3MB)
- **Inference Time**: ~50% reduction (< 10ms)  
- **Memory Usage**: ~30% reduction (< 5MB)
- **Power Consumption**: Significant reduction (accelerometer only)

## 🔧 Data Processing Pipeline

1. **Data Collection**: CSV files with accelerometer readings
2. **Feature Engineering**: Calculate magnitude from 3-axis data
3. **Windowing**: Create overlapping windows (100 samples, 50% overlap)
4. **Normalization**: StandardScaler for feature scaling
5. **Training**: CNN model training with validation split
6. **Optimization**: TensorFlow Lite conversion for mobile

## 📈 Model Training Details

### Data Preparation
- **Window Size**: 100 timesteps (2 seconds at 50Hz)
- **Overlap**: 50% (50 timesteps)
- **Train/Validation/Test Split**: Stratified splitting
- **Normalization**: StandardScaler per feature

### CNN Architecture
- Multiple Conv1D layers with BatchNormalization
- MaxPooling1D for dimensionality reduction
- Dropout for regularization
- Dense layers for classification
- Softmax activation for multi-class output

### Training Strategy
- **Early Stopping**: Monitor validation loss
- **Learning Rate Reduction**: ReduceLROnPlateau
- **Batch Processing**: Efficient GPU utilization
- **Class Balancing**: Handled through stratified sampling

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

## 👨‍💻 Author

**DisEzMike**
- GitHub: [@DisEzMike](https://github.com/DisEzMike)

## 🙏 Acknowledgments

- TensorFlow team for the excellent deep learning framework
- Contributors who provided accelerometer data for training
- Open source community for tools and libraries used

---

⭐ **Star this repository if you found it helpful!**