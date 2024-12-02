import streamlit as st
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import pickle
import clip
from torchvision import transforms
import io
import base64
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold


#***LLM based tools were used for conceptual assistance, clarification, and syntactic reference


# Constants
EMOTION_LABELS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TIMESTAMP = "20241202_0613"

class EnsembleLogisticRegression:
    def __init__(self, n_estimators=5, max_iter=1000, batch_size=5000):
        self.n_estimators = n_estimators
        self.max_iter = max_iter
        self.models = []
        self.scaler = StandardScaler()
        self.feature_selector = VarianceThreshold(threshold=0.01)
        self.batch_size = batch_size

    def predict_proba(self, X):
        n_samples = len(X)
        n_classes = len(self.models[0].classes_)
        predictions = np.zeros((n_samples, n_classes))

        # Transform features once
        X_scaled = self.scaler.transform(X)
        X_selected = self.feature_selector.transform(X_scaled)

        # Process in batches
        for start_idx in range(0, n_samples, self.batch_size):
            end_idx = min(start_idx + self.batch_size, n_samples)
            X_batch = X_selected[start_idx:end_idx]

            # Get predictions from all models
            batch_predictions = np.zeros((end_idx - start_idx, n_classes))
            for model in self.models:
                batch_predictions += model.predict_proba(X_batch)

            # Store averaged predictions
            predictions[start_idx:end_idx] = batch_predictions / len(self.models)

        return predictions

    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1)

# Model architectures
class FeatureMLP(nn.Module):
    def __init__(self, feature_dim=512, num_classes=7):
        super().__init__()
        
        # Wider architecture with skip connections
        self.input_layer = nn.Linear(feature_dim, 1024)
        self.bn_input = nn.BatchNorm1d(1024)
        
        self.hidden1 = nn.Linear(1024, 512)
        self.bn1 = nn.BatchNorm1d(512)
        
        self.hidden2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        
        self.hidden3 = nn.Linear(256, 128)
        self.bn3 = nn.BatchNorm1d(128)
        
        # Skip connections
        self.skip1 = nn.Linear(1024, 512)
        self.skip2 = nn.Linear(512, 256)
        self.skip3 = nn.Linear(256, 128)
        
        self.output = nn.Linear(128, num_classes)
        
        # Dropouts
        self.dropout1 = nn.Dropout(0.5)
        self.dropout2 = nn.Dropout(0.4)
        self.dropout3 = nn.Dropout(0.3)
        
        self.gelu = nn.GELU()

    def forward(self, x):
        x1 = self.gelu(self.bn_input(self.input_layer(x)))
        x1 = self.dropout1(x1)
        
        x2_main = self.gelu(self.bn1(self.hidden1(x1)))
        x2_skip = self.skip1(x1)
        x2 = x2_main + x2_skip
        x2 = self.dropout2(x2)
        
        x3_main = self.gelu(self.bn2(self.hidden2(x2)))
        x3_skip = self.skip2(x2)
        x3 = x3_main + x3_skip
        x3 = self.dropout3(x3)
        
        x4_main = self.gelu(self.bn3(self.hidden3(x3)))
        x4_skip = self.skip3(x3)
        x4 = x4_main + x4_skip
        
        return self.output(x4)

class EmotionCNN(nn.Module):
    def __init__(self, num_classes=7):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.25),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.25),
            
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.25),
        )
        self.classifier = nn.Sequential(
            nn.Linear(256 * 6 * 6, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.reshape(x.size(0), -1)
        x = self.classifier(x)
        return x

class EnhancedEnsembleModel:
    def __init__(self, mlp_model, logreg_model, cnn_model, model_weights=None):
        self.mlp_model = mlp_model
        self.logreg_model = logreg_model
        self.cnn_model = cnn_model
        self.device = DEVICE
        self.model_weights = model_weights or {
            'mlp': 0.4,
            'logreg': 0.3,
            'cnn': 0.3
        }

    def predict_mlp(self, clip_features):
        self.mlp_model.eval()
        with torch.no_grad():
            features = torch.FloatTensor(clip_features).to(self.device)
            logits = self.mlp_model(features)
            probs = torch.softmax(logits, dim=1)
            preds = logits.argmax(dim=1).cpu().numpy()
            return preds, probs.cpu().numpy()

    def predict_logreg(self, clip_features):
        preds = self.logreg_model.predict(clip_features)
        probs = self.logreg_model.predict_proba(clip_features)
        return preds, probs

    def predict_cnn(self, images):
        self.cnn_model.eval()
        with torch.no_grad():
            inputs = torch.FloatTensor(images).permute(0, 3, 1, 2).to(self.device) / 255.0
            logits = self.cnn_model(inputs)
            probs = torch.softmax(logits, dim=1)
            preds = logits.argmax(dim=1).cpu().numpy()
            return preds, probs.cpu().numpy()

    def predict_all(self, images, clip_features):
        results = {}
        results['mlp'] = self.predict_mlp(clip_features)
        results['logreg'] = self.predict_logreg(clip_features)
        results['cnn'] = self.predict_cnn(images)

        # Get weighted ensemble predictions
        weighted_probs = (
            self.model_weights['mlp'] * results['mlp'][1] +
            self.model_weights['logreg'] * results['logreg'][1] +
            self.model_weights['cnn'] * results['cnn'][1]
        )
        weighted_preds = np.argmax(weighted_probs, axis=1)
        results['ensemble'] = (weighted_preds, weighted_probs)

        return results

@st.cache_resource
def load_models():
    # Load CLIP
    clip_model, preprocess = clip.load("ViT-B/32", device=DEVICE)
    
    # Load MLP
    mlp_model = FeatureMLP().to(DEVICE)
    mlp_model.load_state_dict(torch.load('mlp_model_20241202_0613.pth', map_location=DEVICE))
    mlp_model.eval()

    # Load Logistic Regression
    with open('logreg_model_20241202_0613.pkl', 'rb') as f:
        logreg_model = pickle.load(f)

    # Load CNN
    cnn_model = EmotionCNN().to(DEVICE)
    cnn_model.load_state_dict(torch.load('cnn_model_20241202_0613.pth', map_location=DEVICE))
    cnn_model.eval()

    # Load ensemble weights
    with open('ensemble_weights_20241202_0613.pkl', 'rb') as f:
        weights = pickle.load(f)

    # Create ensemble
    ensemble_model = EnhancedEnsembleModel(mlp_model, logreg_model, cnn_model, weights['model_weights'])
    
    return clip_model, preprocess, ensemble_model

def process_image(image, clip_model, preprocess, ensemble_model):
    # Convert to grayscale and resize to 48x48
    img_gray = image.convert('L').resize((48, 48))
    img_array = np.array(img_gray)
    img_array_cnn = img_array.reshape(1, 48, 48, 1)

    # Prepare image for CLIP
    img_rgb = Image.new('RGB', image.size, (0, 0, 0))
    img_rgb.paste(img_gray)
    img_clip = preprocess(img_rgb).unsqueeze(0).to(DEVICE)

    # Extract CLIP features
    with torch.no_grad():
        features = clip_model.encode_image(img_clip)
        features = features / features.norm(dim=-1, keepdim=True)
        clip_features = features.cpu().numpy()

    # Get predictions
    return ensemble_model.predict_all(img_array_cnn, clip_features)

def main():
    st.set_page_config(
        page_title="Enhanced Emotion Detection App",
        page_icon="😊",
        layout="wide"
    )
    
    st.title("Enhanced Emotion Detection App")
    st.markdown("""
    This app uses an ensemble of three advanced models to detect emotions in facial expressions:
    - **Enhanced MLP** with skip connections and GELU activation
    - **Ensemble Logistic Regression** with feature selection
    - **Deep CNN** with batch normalization and dropout
    """)

    # Load models
    with st.spinner("Loading models..."):
        clip_model, preprocess, ensemble_model = load_models()

    # File uploader
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Display images
        col1, col2 = st.columns(2)
        
        image = Image.open(uploaded_file)
        with col1:
            st.subheader("Original Image")
            st.image(image, use_container_width=True)
        
        with col2:
            st.subheader("Processed Image")
            img_gray = image.convert('L').resize((48, 48))
            st.image(img_gray, use_container_width=True)

        # Process image and get predictions
        with st.spinner("Analyzing image..."):
            results = process_image(image, clip_model, preprocess, ensemble_model)

        # Create tabs for different models
        tabs = st.tabs(["Ensemble Results", "Individual Models", "Confidence Analysis"])
        
        with tabs[0]:
            # Display ensemble results prominently
            ensemble_preds, ensemble_probs = results['ensemble']
            pred_emotion = EMOTION_LABELS[ensemble_preds[0]]
            confidence = np.max(ensemble_probs[0]) * 100
            
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f"### Final Prediction: **{pred_emotion}**")
                st.markdown(f"### Confidence: **{confidence:.2f}%**")
            
            with col2:
                # Display emotion emoji based on prediction
                emojis = {"Angry": "😠", "Disgust": "🤢", "Fear": "😨", 
                        "Happy": "😊", "Sad": "😢", "Surprise": "😲", "Neutral": "😐"}
                st.markdown(f"# {emojis[pred_emotion]}")
            
            # Add the logic diagram
            st.markdown("### Model Logic")
            st.image("mermaid_logic.png", use_container_width=True, caption="Ensemble Model Architecture")

        # In the tabs[1] block, modify the model weights display:
        with tabs[1]:
            # Individual model results
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### Model Predictions")
                for model_name in ['mlp', 'logreg', 'cnn']:
                    preds, probs = results[model_name]
                    pred_emotion = EMOTION_LABELS[preds[0]]
                    confidence = float(np.max(probs[0]) * 100)  # Convert to native float
                    st.markdown(f"**{model_name.upper()}**: {pred_emotion} ({confidence:.2f}%)")
            
            with col2:
                st.markdown("### Model Weights")
                for model, weight in ensemble_model.model_weights.items():
                    st.progress(float(weight))  # Convert to native float
                    st.markdown(f"**{model.upper()}**: {weight*100:.1f}%")

        # And in the confidence analysis tab:
        with tabs[2]:
            st.markdown("### Confidence Analysis for Each Model")
            
            for model_name, (preds, probs) in results.items():
                if model_name != 'ensemble':
                    st.markdown(f"#### {model_name.upper()}")
                    top3_idx = np.argsort(probs[0])[-3:][::-1]
                    
                    for idx in top3_idx:
                        confidence = float(probs[0][idx] * 100)  # Convert to native float
                        st.progress(confidence/100)
                        st.markdown(f"{EMOTION_LABELS[idx]}: {confidence:.2f}%")
                    st.markdown("---")

if __name__ == "__main__":
    main()