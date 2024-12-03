# AML_22

Emotion Recongition System

- Comprehensive emotino recognition that creates an ensemble method with CNN, MLP, and logistic regression. CLIP feature extraction is used for the MLP and LogReg training

Overview
- Ensemble based emtion system that can classify based on
- Angry, Disgust, Fear, Happy, Sad, Surprise, Neutral

Features
Multi-Model Architecture:
- Convolutional Neural Network (CNN) for direct image processing
- Multi-Layer Perceptron (MLP) using CLIP features
- Ensemble Logistic Regression using CLIP features

Advanced Ensemble Methods:
- Weighted voting combination
- Hierarchical decision making
- Dynamic confidence-based weight adjustment

CLIP Integration: Utilizes OpenAI's CLIP model for robust feature extraction

Requirements:
torch
torchvision
numpy
pandas
scikit-learn
PIL
clip
ftfy
regex
tqdm

Model Architecture

CNN Model
- Three convolutional blocks with batch normalization and dropout
- Max pooling layers
- Fully connected layers with dropout
- BatchNorm and ReLU activation

Feature MLP
- Multi-layer architecture with skip connections
- Batch normalization and GELU activation
- Dropout for regularization
- Dynamic learning rate scheduling

Ensemble Logistic Regression
- Multiple logistic regression models (3% performance improvement from standard LogReg)
- Feature selection using variance threshold
- Batch processing for memory efficiency
- Class weight balancing

Performance
The system achieves the following accuracies on the validation set:
Through empircal testing, CNN performed better on stock images so was given preference in hierarchical model.
- MLP with CLIP features: ~69%
- Logistic Regression: ~65%
- CNN: ~63%
- Weighted Voting Ensemble: ~70%
- Hierarchical Ensemble: ~69%

Model weights are saved with timestamps:
- mlp_model_TIMESTAMP.pth
- logreg_model_TIMESTAMP.pkl
- cnn_model_TIMESTAMP.pth
- ensemble_weights_TIMESTAMP.pkl