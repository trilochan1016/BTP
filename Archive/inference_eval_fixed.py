"""
Advanced Inference and Evaluation Script for Deepfake Detection
Error-free production version
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np
import pandas as pd
from sklearn.metrics import (precision_recall_fscore_support, roc_auc_score, 
                           roc_curve, precision_recall_curve, average_precision_score,
                           accuracy_score, classification_report, confusion_matrix)
from sklearn.preprocessing import LabelBinarizer
import matplotlib.pyplot as plt
import seaborn as sns
from torchvision import transforms
from scipy.stats import skew, kurtosis, mode
from skimage.measure import shannon_entropy
import os
import warnings
warnings.filterwarnings('ignore')

# Import model architectures (they must be available)
from torch.nn import TransformerEncoder, TransformerEncoderLayer


# ==================== MODEL ARCHITECTURES (Copy from training script) ====================

class PatchEmbedding(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_channels=3, embed_dim=768):
        super().__init__()
        self.n_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        
    def forward(self, x):
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        return x


class CustomViT(nn.Module):
    def __init__(self, img_size=224, patch_size=16, embed_dim=768, depth=6, n_heads=8):
        super().__init__()
        
        self.patch_embed = PatchEmbedding(img_size, patch_size, 3, embed_dim)
        n_patches = self.patch_embed.n_patches
        
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, n_patches + 1, embed_dim))
        self.dropout = nn.Dropout(0.1)
        
        encoder_layer = TransformerEncoderLayer(
            d_model=embed_dim, nhead=n_heads,
            dim_feedforward=int(embed_dim * 4),
            dropout=0.1, batch_first=True
        )
        self.transformer = TransformerEncoder(encoder_layer, num_layers=depth)
        
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, 512)
        
    def forward(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)
        
        cls_token = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_token, x), dim=1)
        x = x + self.pos_embed
        x = self.dropout(x)
        
        x = self.transformer(x)
        x = self.norm(x)
        x = x[:, 0]
        x = self.head(x)
        return x


class CustomVGG(nn.Module):
    def __init__(self, num_classes=5):
        super(CustomVGG, self).__init__()
        
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2), nn.Dropout2d(0.1),
            
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2), nn.Dropout2d(0.1),
            
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2), nn.Dropout2d(0.2),
            
            nn.Conv2d(256, 512, 3, padding=1), nn.BatchNorm2d(512), nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1), nn.BatchNorm2d(512), nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1), nn.BatchNorm2d(512), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2), nn.Dropout2d(0.2),
        )
        
        self.adaptive_pool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096), nn.ReLU(inplace=True), nn.Dropout(0.5),
            nn.Linear(4096, 2048), nn.ReLU(inplace=True), nn.Dropout(0.5),
            nn.Linear(2048, 512),
        )
        
    def forward(self, x):
        x = self.features(x)
        x = self.adaptive_pool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


class TemporalNet(nn.Module):
    def __init__(self, input_dim=8, hidden_dim=128, num_layers=2):
        super(TemporalNet, self).__init__()
        
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, 
                           batch_first=True, dropout=0.3 if num_layers > 1 else 0, 
                           bidirectional=True)
        
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers, 
                         batch_first=True, dropout=0.3 if num_layers > 1 else 0, 
                         bidirectional=True)
        
        self.attention = nn.MultiheadAttention(hidden_dim * 4, 8, batch_first=True)
        
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 4, 256), nn.ReLU(), nn.Dropout(0.4),
            nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, 64)
        )
        
    def forward(self, x):
        if len(x.shape) == 2:
            x = x.unsqueeze(1)
        
        lstm_out, _ = self.lstm(x)
        gru_out, _ = self.gru(x)
        
        combined = torch.cat([lstm_out, gru_out], dim=-1)
        attended, _ = self.attention(combined, combined, combined)
        pooled = torch.mean(attended, dim=1)
        output = self.fusion(pooled)
        
        return output


class FrequencyNet(nn.Module):
    def __init__(self, input_dim=5, hidden_dim=64):
        super(FrequencyNet, self).__init__()
        
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.BatchNorm1d(hidden_dim), 
            nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim * 2), nn.BatchNorm1d(hidden_dim * 2), 
            nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(hidden_dim * 2, hidden_dim * 4), nn.BatchNorm1d(hidden_dim * 4), 
            nn.ReLU(), nn.Dropout(0.4),
            nn.Linear(hidden_dim * 4, hidden_dim * 2), nn.BatchNorm1d(hidden_dim * 2), 
            nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(hidden_dim * 2, 64),
        )
        
    def forward(self, x):
        return self.feature_extractor(x)


class MultiDomainDeepfakeDetector(nn.Module):
    def __init__(self, num_classes=5, use_vit=True):
        super(MultiDomainDeepfakeDetector, self).__init__()
        
        if use_vit:
            self.spatial_net = CustomViT()
        else:
            self.spatial_net = CustomVGG()
        
        self.temporal_net = TemporalNet()
        self.frequency_net = FrequencyNet()
        
        self.fusion_attention = nn.Sequential(
            nn.Linear(512 + 64 + 64, 256), nn.ReLU(),
            nn.Linear(256, 3), nn.Softmax(dim=1)
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(512 + 64 + 64, 512), nn.BatchNorm1d(512), 
            nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(512, 256), nn.BatchNorm1d(256), 
            nn.ReLU(), nn.Dropout(0.4),
            nn.Linear(256, 128), nn.BatchNorm1d(128), 
            nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )
        
    def forward(self, image, temporal_features, frequency_features):
        spatial_features = self.spatial_net(image)
        temporal_features = self.temporal_net(temporal_features)
        frequency_features = self.frequency_net(frequency_features)
        
        combined_features = torch.cat([spatial_features, temporal_features, frequency_features], dim=1)
        attention_weights = self.fusion_attention(combined_features)
        
        weighted_spatial = spatial_features * attention_weights[:, 0:1]
        weighted_temporal = temporal_features * attention_weights[:, 1:2]
        weighted_frequency = frequency_features * attention_weights[:, 2:3]
        
        final_features = torch.cat([weighted_spatial, weighted_temporal, weighted_frequency], dim=1)
        output = self.classifier(final_features)
        
        return output, attention_weights


# ==================== INFERENCE CLASS ====================

class DeepfakeInference:
    def __init__(self, model_path='./Output/best_model.pth'):
        """Initialize the inference pipeline"""
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Load the trained model
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found at {model_path}")
        
        print(f"Loading model from {model_path}...")
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # Initialize model architecture
        self.model = MultiDomainDeepfakeDetector(num_classes=5).to(self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        # Load preprocessing objects
        self.label_encoder = checkpoint['label_encoder']
        self.scaler_temporal = checkpoint['scaler_temporal']
        self.scaler_frequency = checkpoint['scaler_frequency']
        
        # Image preprocessing
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        print("✅ Model loaded successfully!")
        print(f"Classes: {self.label_encoder.classes_}")
    
    def compute_frame_features(self, frame):
        """Compute temporal and frequency features for a single frame"""
        
        try:
            if frame is None or frame.size == 0:
                return None, None
            
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            pixels = gray.flatten().astype(np.float64)
            
            # Temporal features
            temporal_features = {
                "mean_intensity": float(np.mean(pixels)),
                "variance": float(np.var(pixels)),
                "std_dev": float(np.std(pixels)),
                "skewness": float(skew(pixels)),
                "kurtosis": float(kurtosis(pixels)),
                "median_intensity": float(np.median(pixels)),
                "mode_intensity": float(mode(pixels, keepdims=True).mode[0]),
                "entropy": float(shannon_entropy(gray))
            }
            
            # DFT features
            dft = np.fft.fft2(gray)
            dft_shift = np.fft.fftshift(dft)
            magnitude_spectrum = 20 * np.log(np.abs(dft_shift) + 1)
            
            frequency_features = {
                "dft_mean": float(np.mean(magnitude_spectrum)),
                "dft_variance": float(np.var(magnitude_spectrum)),
                "dft_std": float(np.std(magnitude_spectrum)),
                "dft_median": float(np.median(magnitude_spectrum)),
                "dft_max": float(np.max(magnitude_spectrum))
            }
            
            return temporal_features, frequency_features
            
        except Exception as e:
            print(f"⚠️ Error computing features: {e}")
            return None, None
    
    def predict_single_image(self, image_path):
        """Predict on a single image"""
        
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        # Load and preprocess image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Extract features
        temporal_features, frequency_features = self.compute_frame_features(image)
        
        if temporal_features is None or frequency_features is None:
            raise ValueError("Failed to compute features")
        
        # Convert to tensors and normalize
        temporal_array = np.array(list(temporal_features.values())).reshape(1, -1)
        frequency_array = np.array(list(frequency_features.values())).reshape(1, -1)
        
        temporal_normalized = self.scaler_temporal.transform(temporal_array)
        frequency_normalized = self.scaler_frequency.transform(frequency_array)
        
        temporal_tensor = torch.tensor(temporal_normalized, dtype=torch.float32).to(self.device)
        frequency_tensor = torch.tensor(frequency_normalized, dtype=torch.float32).to(self.device)
        image_tensor = self.transform(image_rgb).unsqueeze(0).to(self.device)
        
        # Predict
        with torch.no_grad():
            outputs, attention_weights = self.model(image_tensor, temporal_tensor, frequency_tensor)
            probabilities = F.softmax(outputs, dim=1)
            predicted_class = torch.argmax(probabilities, dim=1).item()
            confidence = torch.max(probabilities).item()
        
        # Decode prediction
        predicted_label = self.label_encoder.inverse_transform([predicted_class])[0]
        
        return {
            'predicted_class': predicted_label,
            'confidence': confidence,
            'probabilities': probabilities.cpu().numpy()[0],
            'attention_weights': attention_weights.cpu().numpy()[0],
            'all_classes': self.label_encoder.classes_.tolist()
        }
    
    def predict_video(self, video_path, frame_interval=1000, max_frames=30):
        """Predict on video by extracting frames"""
        
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video not found: {video_path}")
        
        vidcap = cv2.VideoCapture(video_path)
        if not vidcap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        fps = vidcap.get(cv2.CAP_PROP_FPS)
        frame_count = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if fps == 0 or frame_count == 0:
            vidcap.release()
            raise ValueError(f"Invalid video: {video_path}")
        
        duration_ms = (frame_count / fps) * 1000
        
        predictions = []
        timestamp_ms = 0
        frame_counter = 0
        
        # Create temp directory
        temp_dir = './Output/temp_frames'
        os.makedirs(temp_dir, exist_ok=True)
        
        while timestamp_ms <= duration_ms and frame_counter < max_frames:
            vidcap.set(cv2.CAP_PROP_POS_MSEC, timestamp_ms)
            ret, frame = vidcap.read()
            
            if not ret or frame is None:
                break
            
            # Save frame temporarily
            temp_path = os.path.join(temp_dir, f'temp_frame_{frame_counter}.jpg')
            cv2.imwrite(temp_path, frame)
            
            try:
                # Predict on frame
                result = self.predict_single_image(temp_path)
                result['timestamp_ms'] = timestamp_ms
                result['frame_number'] = frame_counter
                predictions.append(result)
            except Exception as e:
                print(f"⚠️ Error predicting frame {frame_counter}: {e}")
            finally:
                # Clean up temp file
                if os.path.exists(temp_path):
                    os.remove(temp_path)
            
            timestamp_ms += frame_interval
            frame_counter += 1
        
        vidcap.release()
        
        # Clean up temp directory
        if os.path.exists(temp_dir):
            import shutil
            shutil.rmtree(temp_dir)
        
        if not predictions:
            raise ValueError("No frames could be processed from video")
        
        # Aggregate predictions
        all_probs = np.array([p['probabilities'] for p in predictions])
        avg_probs = np.mean(all_probs, axis=0)
        final_prediction = self.label_encoder.classes_[np.argmax(avg_probs)]
        final_confidence = np.max(avg_probs)
        
        # Calculate prediction consistency
        pred_classes = [p['predicted_class'] for p in predictions]
        most_common = max(set(pred_classes), key=pred_classes.count)
        consistency = pred_classes.count(most_common) / len(pred_classes)
        
        return {
            'final_prediction': final_prediction,
            'final_confidence': float(final_confidence),
            'consistency': float(consistency),
            'frames_analyzed': len(predictions),
            'frame_predictions': predictions,
            'average_probabilities': avg_probs.tolist()
        }


# ==================== ADVANCED EVALUATION METRICS ====================

class AdvancedEvaluator:
    def __init__(self, model_path='./Output/best_model.pth'):
        self.inference = DeepfakeInference(model_path)
        
    def evaluate_dataset(self, test_df):
        """Comprehensive evaluation on test dataset"""
        
        print("\n" + "=" * 80)
        print("🔍 STARTING COMPREHENSIVE EVALUATION")
        print("=" * 80)
        
        all_predictions = []
        all_labels = []
        all_probabilities = []
        all_attention_weights = []
        
        total = len(test_df)
        print(f"Total samples to evaluate: {total}")
        
        for idx, row in test_df.iterrows():
            try:
                result = self.inference.predict_single_image(row['image_path'])
                
                all_predictions.append(result['predicted_class'])
                all_labels.append(row['label'])
                all_probabilities.append(result['probabilities'])
                all_attention_weights.append(result['attention_weights'])
                
                if (idx + 1) % 50 == 0:
                    print(f"Processed {idx + 1}/{total} images ({100*(idx+1)/total:.1f}%)")
                    
            except Exception as e:
                print(f"⚠️ Error processing {row['image_path']}: {e}")
                continue
        
        if not all_predictions:
            raise ValueError("No predictions were made successfully!")
        
        print(f"\n✅ Successfully processed {len(all_predictions)}/{total} samples")
        
        # Convert to numpy arrays
        all_probabilities = np.array(all_probabilities)
        all_attention_weights = np.array(all_attention_weights)
        
        # Calculate comprehensive metrics
        print("\n📊 Calculating metrics...")
        metrics = self.calculate_comprehensive_metrics(
            all_labels, all_predictions, all_probabilities
        )
        
        # Attention analysis
        print("\n🎯 Analyzing attention patterns...")
        attention_analysis = self.analyze_attention_patterns(all_attention_weights, all_labels)
        
        # Generate visualizations
        print("\n📈 Creating visualizations...")
        self.create_comprehensive_visualizations(
            all_labels, all_predictions, all_probabilities, 
            all_attention_weights, metrics
        )
        
        return {
            'metrics': metrics,
            'attention_analysis': attention_analysis,
            'predictions': all_predictions,
            'labels': all_labels,
            'probabilities': all_probabilities
        }
    
    def calculate_comprehensive_metrics(self, true_labels, predictions, probabilities):
        """Calculate comprehensive evaluation metrics"""
        
        # Basic metrics
        accuracy = accuracy_score(true_labels, predictions)
        
        # Per-class metrics
        precision, recall, f1, support = precision_recall_fscore_support(
            true_labels, predictions, average=None, 
            labels=self.inference.label_encoder.classes_, zero_division=0
        )
        
        # Macro and micro averages
        macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
            true_labels, predictions, average='macro', zero_division=0
        )
        
        micro_precision, micro_recall, micro_f1, _ = precision_recall_fscore_support(
            true_labels, predictions, average='micro', zero_division=0
        )
        
        # Weighted averages
        weighted_precision, weighted_recall, weighted_f1, _ = precision_recall_fscore_support(
            true_labels, predictions, average='weighted', zero_division=0
        )
        
        # ROC-AUC and PR-AUC (for multiclass)
        try:
            label_binarizer = LabelBinarizer()
            label_binarizer.fit(self.inference.label_encoder.classes_)
            true_binary = label_binarizer.transform(true_labels)
            
            if len(self.inference.label_encoder.classes_) == 2:
                roc_auc = roc_auc_score(true_binary, probabilities[:, 1])
                pr_auc = average_precision_score(true_binary, probabilities[:, 1])
            else:
                roc_auc = roc_auc_score(true_binary, probabilities, multi_class='ovr', average='macro')
                pr_auc = average_precision_score(true_binary, probabilities, average='macro')
        except Exception as e:
            print(f"⚠️ Warning: Could not calculate AUC metrics: {e}")
            roc_auc = 0.0
            pr_auc = 0.0
        
        return {
            'accuracy': accuracy,
            'per_class': {
                'precision': dict(zip(self.inference.label_encoder.classes_, precision)),
                'recall': dict(zip(self.inference.label_encoder.classes_, recall)),
                'f1': dict(zip(self.inference.label_encoder.classes_, f1)),
                'support': dict(zip(self.inference.label_encoder.classes_, support))
            },
            'macro_avg': {
                'precision': macro_precision,
                'recall': macro_recall,
                'f1': macro_f1
            },
            'micro_avg': {
                'precision': micro_precision,
                'recall': micro_recall,
                'f1': micro_f1
            },
            'weighted_avg': {
                'precision': weighted_precision,
                'recall': weighted_recall,
                'f1': weighted_f1
            },
            'roc_auc': roc_auc,
            'pr_auc': pr_auc
        }
    
    def analyze_attention_patterns(self, attention_weights, labels):
        """Analyze attention patterns across different classes"""
        
        # attention_weights shape: (n_samples, 3) for [spatial, temporal, frequency]
        df_attention = pd.DataFrame({
            'spatial_attention': attention_weights[:, 0],
            'temporal_attention': attention_weights[:, 1],
            'frequency_attention': attention_weights[:, 2],
            'label': labels
        })
        
        # Calculate average attention for each class
        attention_by_class = df_attention.groupby('label')[
            ['spatial_attention', 'temporal_attention', 'frequency_attention']
        ].mean()
        
        return {
            'attention_by_class': attention_by_class,
            'attention_dataframe': df_attention
        }
    
    def create_comprehensive_visualizations(self, true_labels, predictions, 
                                         probabilities, attention_weights, metrics):
        """Create comprehensive visualization plots"""
        
        fig = plt.figure(figsize=(20, 15))
        
        classes = self.inference.label_encoder.classes_
        
        # 1. Confusion Matrix
        ax1 = plt.subplot(3, 4, 1)
        cm = confusion_matrix(true_labels, predictions, labels=classes)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=classes, yticklabels=classes, ax=ax1)
        ax1.set_title('Confusion Matrix', fontsize=12, fontweight='bold')
        ax1.set_ylabel('True Label')
        ax1.set_xlabel('Predicted Label')
        plt.setp(ax1.get_xticklabels(), rotation=45, ha='right')
        
        # 2. Per-class Performance
        ax2 = plt.subplot(3, 4, 2)
        per_class = metrics['per_class']
        x_pos = np.arange(len(classes))
        width = 0.25
        
        ax2.bar(x_pos - width, [per_class['precision'][c] for c in classes], 
               width=width, label='Precision', alpha=0.8)
        ax2.bar(x_pos, [per_class['recall'][c] for c in classes], 
               width=width, label='Recall', alpha=0.8)
        ax2.bar(x_pos + width, [per_class['f1'][c] for c in classes], 
               width=width, label='F1-Score', alpha=0.8)
        
        ax2.set_xlabel('Classes')
        ax2.set_ylabel('Score')
        ax2.set_title('Per-class Performance', fontsize=12, fontweight='bold')
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels(classes, rotation=45, ha='right')
        ax2.legend()
        ax2.set_ylim(0, 1.1)
        ax2.grid(True, alpha=0.3)
        
        # 3. ROC Curves
        ax3 = plt.subplot(3, 4, 3)
        try:
            label_binarizer = LabelBinarizer()
            label_binarizer.fit(classes)
            
            for i, class_name in enumerate(classes):
                y_true_binary = (np.array(true_labels) == class_name).astype(int)
                if len(np.unique(y_true_binary)) > 1:
                    fpr, tpr, _ = roc_curve(y_true_binary, probabilities[:, i])
                    auc = roc_auc_score(y_true_binary, probabilities[:, i])
                    ax3.plot(fpr, tpr, label=f'{class_name} (AUC={auc:.3f})', linewidth=2)
        except Exception as e:
            print(f"⚠️ Warning: Could not plot ROC curves: {e}")
        
        ax3.plot([0, 1], [0, 1], 'k--', alpha=0.5, linewidth=1)
        ax3.set_xlabel('False Positive Rate')
        ax3.set_ylabel('True Positive Rate')
        ax3.set_title('ROC Curves', fontsize=12, fontweight='bold')
        ax3.legend(fontsize=8)
        ax3.grid(True, alpha=0.3)
        
        # 4. Precision-Recall Curves
        ax4 = plt.subplot(3, 4, 4)
        try:
            for i, class_name in enumerate(classes):
                y_true_binary = (np.array(true_labels) == class_name).astype(int)
                if len(np.unique(y_true_binary)) > 1:
                    precision_vals, recall_vals, _ = precision_recall_curve(y_true_binary, probabilities[:, i])
                    ap = average_precision_score(y_true_binary, probabilities[:, i])
                    ax4.plot(recall_vals, precision_vals, label=f'{class_name} (AP={ap:.3f})', linewidth=2)
        except Exception as e:
            print(f"⚠️ Warning: Could not plot PR curves: {e}")
        
        ax4.set_xlabel('Recall')
        ax4.set_ylabel('Precision')
        ax4.set_title('Precision-Recall Curves', fontsize=12, fontweight='bold')
        ax4.legend(fontsize=8)
        ax4.grid(True, alpha=0.3)
        
        # 5. Attention Analysis by Class
        ax5 = plt.subplot(3, 4, 5)
        attention_df = pd.DataFrame({
            'Spatial': attention_weights[:, 0],
            'Temporal': attention_weights[:, 1],
            'Frequency': attention_weights[:, 2],
            'Label': true_labels
        })
        
        attention_means = attention_df.groupby('Label')[['Spatial', 'Temporal', 'Frequency']].mean()
        attention_means.plot(kind='bar', ax=ax5, width=0.8)
        ax5.set_title('Avg Attention Weights by Class', fontsize=12, fontweight='bold')
        ax5.set_ylabel('Attention Weight')
        ax5.set_xlabel('')
        plt.setp(ax5.get_xticklabels(), rotation=45, ha='right')
        ax5.legend(fontsize=8)
        ax5.grid(True, alpha=0.3, axis='y')
        
        # 6. Confidence Distribution
        ax6 = plt.subplot(3, 4, 6)
        max_probs = np.max(probabilities, axis=1)
        correct_predictions = (np.array(predictions) == np.array(true_labels))
        
        ax6.hist(max_probs[correct_predictions], alpha=0.7, label='Correct', bins=20, color='green')
        ax6.hist(max_probs[~correct_predictions], alpha=0.7, label='Incorrect', bins=20, color='red')
        ax6.set_xlabel('Confidence')
        ax6.set_ylabel('Frequency')
        ax6.set_title('Confidence Distribution', fontsize=12, fontweight='bold')
        ax6.legend()
        ax6.grid(True, alpha=0.3, axis='y')
        
        # 7. Class Distribution
        ax7 = plt.subplot(3, 4, 7)
        unique_labels, counts = np.unique(true_labels, return_counts=True)
        colors = plt.cm.Set3(np.linspace(0, 1, len(unique_labels)))
        ax7.pie(counts, labels=unique_labels, autopct='%1.1f%%', colors=colors, startangle=90)
        ax7.set_title('Class Distribution', fontsize=12, fontweight='bold')
        
        # 8. Error Rate by Class
        ax8 = plt.subplot(3, 4, 8)
        errors_by_class = {}
        for true_class in classes:
            mask = np.array(true_labels) == true_class
            if np.sum(mask) > 0:
                class_predictions = np.array(predictions)[mask]
                errors = np.sum(class_predictions != true_class)
                total = np.sum(mask)
                error_rate = errors / total
                errors_by_class[true_class] = error_rate
            else:
                errors_by_class[true_class] = 0
        
        ax8.bar(errors_by_class.keys(), errors_by_class.values(), color='coral')
        ax8.set_title('Error Rate by True Class', fontsize=12, fontweight='bold')
        ax8.set_ylabel('Error Rate')
        ax8.set_xlabel('')
        plt.setp(ax8.get_xticklabels(), rotation=45, ha='right')
        ax8.grid(True, alpha=0.3, axis='y')
        
        # 9-11. Domain-specific attention distributions
        domain_names = ['Spatial', 'Temporal', 'Frequency']
        for i, domain in enumerate(domain_names):
            ax = plt.subplot(3, 4, 9 + i)
            for class_name in classes:
                class_mask = np.array(true_labels) == class_name
                class_attention = attention_weights[class_mask, i]
                if len(class_attention) > 0:
                    ax.hist(class_attention, alpha=0.6, label=class_name, bins=15)
            ax.set_title(f'{domain} Attention Distribution', fontsize=12, fontweight='bold')
            ax.set_xlabel('Attention Weight')
            ax.set_ylabel('Frequency')
            ax.legend(fontsize=7)
            ax.grid(True, alpha=0.3, axis='y')
        
        # 12. Overall Metrics Summary
        ax12 = plt.subplot(3, 4, 12)
        ax12.axis('off')
        
        summary_text = f"""
        OVERALL METRICS SUMMARY
        {'='*40}
        
        Accuracy: {metrics['accuracy']:.4f}
        
        Macro Averages:
          Precision: {metrics['macro_avg']['precision']:.4f}
          Recall: {metrics['macro_avg']['recall']:.4f}
          F1-Score: {metrics['macro_avg']['f1']:.4f}
        
        Weighted Averages:
          Precision: {metrics['weighted_avg']['precision']:.4f}
          Recall: {metrics['weighted_avg']['recall']:.4f}
          F1-Score: {metrics['weighted_avg']['f1']:.4f}
        
        ROC-AUC: {metrics['roc_auc']:.4f}
        PR-AUC: {metrics['pr_auc']:.4f}
        """
        
        ax12.text(0.1, 0.9, summary_text, fontsize=10, verticalalignment='top',
                 fontfamily='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        plt.savefig('./Output/comprehensive_evaluation.png', dpi=300, bbox_inches='tight')
        print("\n✅ Visualization saved to: ./Output/comprehensive_evaluation.png")
        
        # Print detailed metrics to console
        self.print_detailed_metrics(metrics, classes)
    
    def print_detailed_metrics(self, metrics, classes):
        """Print detailed metrics to console"""
        
        print("\n" + "=" * 80)
        print("📊 COMPREHENSIVE EVALUATION RESULTS")
        print("=" * 80)
        
        print(f"\n🎯 Overall Accuracy: {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
        
        print(f"\n📈 Macro Averages:")
        print(f"   Precision: {metrics['macro_avg']['precision']:.4f}")
        print(f"   Recall:    {metrics['macro_avg']['recall']:.4f}")
        print(f"   F1-Score:  {metrics['macro_avg']['f1']:.4f}")
        
        print(f"\n📊 Weighted Averages:")
        print(f"   Precision: {metrics['weighted_avg']['precision']:.4f}")
        print(f"   Recall:    {metrics['weighted_avg']['recall']:.4f}")
        print(f"   F1-Score:  {metrics['weighted_avg']['f1']:.4f}")
        
        print(f"\n🎨 AUC Scores:")
        print(f"   ROC-AUC: {metrics['roc_auc']:.4f}")
        print(f"   PR-AUC:  {metrics['pr_auc']:.4f}")
        
        print(f"\n🔍 Per-Class Results:")
        print("-" * 80)
        for class_name in classes:
            print(f"\n   {class_name}:")
            print(f"      Precision: {metrics['per_class']['precision'][class_name]:.4f}")
            print(f"      Recall:    {metrics['per_class']['recall'][class_name]:.4f}")
            print(f"      F1-Score:  {metrics['per_class']['f1'][class_name]:.4f}")
            print(f"      Support:   {int(metrics['per_class']['support'][class_name])}")
        
        print("\n" + "=" * 80)


# ==================== USAGE FUNCTIONS ====================

def run_comprehensive_evaluation():
    """Run comprehensive evaluation on test dataset"""
    
    print("\n" + "=" * 80)
    print("🚀 COMPREHENSIVE EVALUATION PIPELINE")
    print("=" * 80)
    
    # Load test data
    merged_features_path = './Output/Merged_Features/'
    images_path = './Output/Images/'
    
    all_data = []
    subfolders = ['original', 'Face2Face', 'FaceShifter', 'FaceSwap', 'NeuralTextures']
    
    print("\n📂 Loading dataset...")
    for subfolder in subfolders:
        csv_path = os.path.join(merged_features_path, f"{subfolder}_merged_features.csv")
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            df['label'] = subfolder
            df['image_path'] = df['image_name'].apply(
                lambda x: os.path.join(images_path, subfolder, x)
            )
            # Validate paths
            df = df[df['image_path'].apply(os.path.exists)]
            all_data.append(df)
            print(f"   ✓ Loaded {len(df)} samples from {subfolder}")
        else:
            print(f"   ✗ CSV not found: {csv_path}")
    
    if not all_data:
        raise ValueError("No data found! Please run the training pipeline first.")
    
    # Combine and split
    combined_df = pd.concat(all_data, ignore_index=True)
    print(f"\n✅ Total samples loaded: {len(combined_df)}")
    
    from sklearn.model_selection import train_test_split
    
    # Use same split as training
    _, test_df = train_test_split(combined_df, test_size=0.2, 
                                  stratify=combined_df['label'], random_state=42)
    
    print(f"📊 Test set size: {len(test_df)}")
    
    # Run evaluation
    evaluator = AdvancedEvaluator()
    results = evaluator.evaluate_dataset(test_df)
    
    print("\n✅ Evaluation completed successfully!")
    
    return results


def predict_single_sample(input_path, is_video=False):
    """Predict on a single image or video"""
    
    print("\n" + "=" * 80)
    print(f"🔮 SINGLE SAMPLE PREDICTION")
    print("=" * 80)
    
    inference = DeepfakeInference()
    
    if is_video:
        print(f"\n🎬 Analyzing video: {input_path}")
        result = inference.predict_video(input_path)
        
        print(f"\n📊 Video Analysis Results:")
        print(f"   Final Prediction: {result['final_prediction']}")
        print(f"   Confidence: {result['final_confidence']:.4f}")
        print(f"   Consistency: {result['consistency']:.4f}")
        print(f"   Frames Analyzed: {result['frames_analyzed']}")
        
        print(f"\n   Frame-by-frame predictions (first 5):")
        for i, pred in enumerate(result['frame_predictions'][:5]):
            print(f"     Frame {i+1} ({pred['timestamp_ms']}ms): {pred['predicted_class']} "
                  f"(conf: {pred['confidence']:.3f})")
        
        print(f"\n   Average Class Probabilities:")
        for class_name, prob in zip(inference.label_encoder.classes_, result['average_probabilities']):
            print(f"     {class_name}: {prob:.4f}")
        
    else:
        print(f"\n🖼️ Analyzing image: {input_path}")
        result = inference.predict_single_image(input_path)
        
        print(f"\n📊 Image Analysis Results:")
        print(f"   Prediction: {result['predicted_class']}")
        print(f"   Confidence: {result['confidence']:.4f}")
        
        print(f"\n   Attention Weights:")
        print(f"     Spatial:   {result['attention_weights'][0]:.4f}")
        print(f"     Temporal:  {result['attention_weights'][1]:.4f}")
        print(f"     Frequency: {result['attention_weights'][2]:.4f}")
        
        print(f"\n   Class Probabilities:")
        for class_name, prob in zip(result['all_classes'], result['probabilities']):
            bar = '█' * int(prob * 50)
            print(f"     {class_name:15s}: {prob:.4f} {bar}")
    
    print("\n" + "=" * 80)
    return result


# ==================== MAIN EXECUTION ====================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Deepfake Detection Inference and Evaluation')
    parser.add_argument('--mode', type=str, default='evaluate', 
                       choices=['evaluate', 'predict_image', 'predict_video'],
                       help='Execution mode')
    parser.add_argument('--input', type=str, default=None,
                       help='Input file path for prediction mode')
    
    args = parser.parse_args()
    
    try:
        if args.mode == 'evaluate':
            print("\n🚀 Running comprehensive evaluation...")
            results = run_comprehensive_evaluation()
            
        elif args.mode == 'predict_image':
            if args.input is None:
                print("❌ Error: --input argument required for prediction mode")
            else:
                result = predict_single_sample(args.input, is_video=False)
                
        elif args.mode == 'predict_video':
            if args.input is None:
                print("❌ Error: --input argument required for prediction mode")
            else:
                result = predict_single_sample(args.input, is_video=True)
        
        print("\n✅ Execution completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Error during execution: {e}")
        import traceback
        traceback.print_exc()