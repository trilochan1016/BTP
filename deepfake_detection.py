import os
import gc
import cv2
import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis, mode
from skimage.measure import shannon_entropy
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_curve, auc
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for server
import matplotlib.pyplot as plt
import seaborn as sns
from torchvision import transforms
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from PIL import Image
import warnings
import pickle
from tqdm import tqdm
import time

warnings.filterwarnings('ignore')

# ==================== GLOBAL CONFIGURATION ====================

CONFIG = {
    'batch_size': 128,  # Increased for better GPU utilization
    'learning_rate': 0.0001,
    'epochs': 50,
    'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
    'num_classes': 2,
    'image_size': 224,
    'num_workers': 4,  # Parallel data loading
    'pin_memory': True,
    'early_stopping_patience': 10,
}

subfolders = ['original', 'FaceSwap', 'FaceShifter', 'Face2Face', 'NeuralTextures']
REAL_LABEL_NAME = 'original'

# ==================== DATASET CLASS ====================

class DeepfakeDataset(Dataset):
    """Optimized Dataset class with robust feature lookup and image loading"""
    def __init__(self, image_paths, features_df, labels, transform=None):
        self.image_paths = image_paths
        # keep a copy and set index for fast lookup
        self.features_df = features_df.set_index('image_name')
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        try:
            img_path = self.image_paths[idx]
            img_name = os.path.basename(img_path)

            # Load image
            image = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
            if image is None:
                print(f"Warning: Failed to load image {img_path}")
                image_tensor = torch.zeros(3, CONFIG['image_size'], CONFIG['image_size'], dtype=torch.float32)
            else:
                # Handle different channel numbers robustly
                if len(image.shape) == 2:  # Grayscale
                    image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
                elif image.shape[2] == 4:  # BGRA
                    # Convert BGRA -> RGB by discarding alpha and converting BGR->RGB
                    image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                else:
                    # Standard BGR -> RGB
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                # Convert to PIL Image for transforms
                image = Image.fromarray(image)

                if self.transform:
                    image_tensor = self.transform(image)
                else:
                    image_tensor = transforms.ToTensor()(image)

                # Ensure 3 channels
                if image_tensor.shape[0] == 1:
                    image_tensor = image_tensor.repeat(3, 1, 1)
                elif image_tensor.shape[0] > 3:
                    image_tensor = image_tensor[:3, :, :]

            # Get features using index. Guard against duplicate entries for same image_name
            if img_name in self.features_df.index:
                feature_row = self.features_df.loc[img_name]
                # If multiple rows exist for same image_name, take the first
                if isinstance(feature_row, pd.DataFrame):
                    feature_row = feature_row.iloc[0]

                temporal_cols = ['mean_intensity', 'variance', 'std_dev', 'skewness', 
                               'kurtosis', 'median_intensity', 'mode_intensity', 'entropy']
                frequency_cols = ['dft_mean', 'dft_variance', 'dft_std', 'dft_median', 'dft_max']

                # Use .get to avoid KeyError if a column is missing; fallback to zeros
                temporal_vals = []
                for c in temporal_cols:
                    if c in feature_row.index:
                        temporal_vals.append(np.float32(feature_row[c]))
                    else:
                        temporal_vals.append(np.float32(0.0))

                frequency_vals = []
                for c in frequency_cols:
                    if c in feature_row.index:
                        frequency_vals.append(np.float32(feature_row[c]))
                    else:
                        frequency_vals.append(np.float32(0.0))

                temporal_features = torch.tensor(temporal_vals, dtype=torch.float32)
                frequency_features = torch.tensor(frequency_vals, dtype=torch.float32)
            else:
                print(f"Warning: Features not found for {img_name}")
                temporal_features = torch.zeros(8, dtype=torch.float32)
                frequency_features = torch.zeros(5, dtype=torch.float32)

            # label
            lbl = self.labels[idx]
            label = torch.tensor(int(lbl), dtype=torch.long)

            return {
                'image': image_tensor,
                'temporal_features': temporal_features,
                'frequency_features': frequency_features,
                'label': label
            }
        except Exception as e:
            print(f"Error loading sample {idx} ({self.image_paths[idx]}): {e}")
            # Return valid tensors with correct shapes
            return {
                'image': torch.zeros(3, CONFIG['image_size'], CONFIG['image_size'], dtype=torch.float32),
                'temporal_features': torch.zeros(8, dtype=torch.float32),
                'frequency_features': torch.zeros(5, dtype=torch.float32),
                'label': torch.tensor(0, dtype=torch.long)
            }


# ==================== MODEL ARCHITECTURES ====================

class CustomVGG(nn.Module):
    def __init__(self, num_classes=512):
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
        # x should be [batch_size, 8]
        # Reshape to [batch_size, 1, 8] for LSTM/GRU (1 timestep)
        if len(x.shape) == 2:
            x = x.unsqueeze(1)

        lstm_out, _ = self.lstm(x)  # [batch_size, 1, hidden_dim*2]
        gru_out, _ = self.gru(x)    # [batch_size, 1, hidden_dim*2]

        combined = torch.cat([lstm_out, gru_out], dim=-1)  # [batch_size, 1, hidden_dim*4]
        attended, _ = self.attention(combined, combined, combined)

        # Remove sequence dimension
        pooled = attended.squeeze(1)  # [batch_size, hidden_dim*4]
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
    def __init__(self, num_classes=2, use_vit=True):
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
        output = self.classifier(combined_features)

        return output, attention_weights


# ==================== DATA PREPARATION ====================

def prepare_data():
    """Load actual extracted features and prepare data"""
    print("\n" + "=" * 80)
    print("LOADING ACTUAL EXTRACTED DATA")
    print("=" * 80)

    merged_features_path = './Output/Merged_Features/'
    images_path = './Output/Images/'

    # Verify paths exist
    if not os.path.exists(merged_features_path):
        raise FileNotFoundError(f"Merged features path not found: {merged_features_path}")
    if not os.path.exists(images_path):
        raise FileNotFoundError(f"Images path not found: {images_path}")

    all_data = []

    for subfolder in subfolders:
        csv_path = os.path.join(merged_features_path, f"{subfolder}_merged_features.csv")
        image_folder = os.path.join(images_path, subfolder)
        
        if not os.path.exists(csv_path):
            print(f"⚠️ Warning: CSV not found for {subfolder}: {csv_path}")
            continue

        if not os.path.exists(image_folder):
            print(f"⚠️ Warning: Image folder not found for {subfolder}: {image_folder}")
            continue

        print(f"📂 Loading data from {subfolder}...")
        df = pd.read_csv(csv_path)

        # Add image paths
        df['image_path'] = df['image_name'].apply(
            lambda x: os.path.join(image_folder, x)
        )

        # Verify images exist
        existing_images = df['image_path'].apply(os.path.exists)
        missing_count = (~existing_images).sum()
        if missing_count > 0:
            print(f"⚠️ Warning: {missing_count} images missing in {subfolder}")
            df = df[existing_images]

        df['label'] = subfolder
        all_data.append(df)
        print(f"✅ Loaded {len(df)} samples from {subfolder}")

    if not all_data:
        raise ValueError("No data found! Please ensure preprocessing was completed.")

    combined_df = pd.concat(all_data, ignore_index=True)
    print(f"\n📊 Total samples loaded: {len(combined_df)}")

    # Binary mapping
    combined_df['binary_label_name'] = combined_df['label'].apply(
        lambda x: REAL_LABEL_NAME if x == REAL_LABEL_NAME else 'Deepfake'
    )

    label_encoder = LabelEncoder()
    combined_df['label_encoded'] = label_encoder.fit_transform(combined_df['binary_label_name'])

    print(f"Binary Classes: {label_encoder.classes_}")
    print(f"Class distribution:")
    print(combined_df['binary_label_name'].value_counts())

    # Normalize features
    temporal_cols = ['mean_intensity', 'variance', 'std_dev', 'skewness', 
                    'kurtosis', 'median_intensity', 'mode_intensity', 'entropy']
    frequency_cols = ['dft_mean', 'dft_variance', 'dft_std', 'dft_median', 'dft_max']

    scaler_temporal = StandardScaler()
    scaler_frequency = StandardScaler()

    # Fill missing columns with zeros to keep shapes consistent
    for c in temporal_cols:
        if c not in combined_df.columns:
            combined_df[c] = 0.0
    for c in frequency_cols:
        if c not in combined_df.columns:
            combined_df[c] = 0.0

    combined_df[temporal_cols] = scaler_temporal.fit_transform(combined_df[temporal_cols])
    combined_df[frequency_cols] = scaler_frequency.fit_transform(combined_df[frequency_cols])

    return combined_df, label_encoder, scaler_temporal, scaler_frequency


# ==================== VISUALIZATION FUNCTIONS ====================

def plot_class_distribution(df, file_path='./Output/class_distribution.png'):
    plt.figure(figsize=(8, 6), facecolor='white')
    sns.set_style("whitegrid")
    ax = sns.countplot(x='binary_label_name', data=df, palette='viridis')
    
    plt.title('Binary Class Distribution (Real vs. Deepfake)', fontsize=16)
    plt.xlabel('Class Label', fontsize=14)
    plt.ylabel('Count', fontsize=14)
    
    for p in ax.patches:
        ax.annotate(f'{int(p.get_height())}', (p.get_x() + p.get_width() / 2., p.get_height()), 
                    ha='center', va='center', fontsize=12, color='black', xytext=(0, 5), 
                    textcoords='offset points')
    
    plt.tight_layout()
    plt.savefig(file_path, dpi=300)
    print(f"📈 Class distribution saved to: {file_path}")
    plt.close()


def plot_attention_weights(attention_weights, file_path='./Output/attention_weights.png'):
    weights_df = pd.DataFrame(attention_weights, columns=['Spatial', 'Temporal', 'Frequency'])
    
    plt.figure(figsize=(10, 6), facecolor='white')
    sns.set_style("whitegrid")
    
    mean_weights = weights_df.mean().sort_values(ascending=False)
    
    ax = sns.barplot(x=mean_weights.index, y=mean_weights.values, palette='plasma')
    
    plt.title('Mean Domain Attention Weights', fontsize=16)
    plt.ylabel('Mean Attention Weight', fontsize=14)
    plt.xlabel('Feature Domain', fontsize=14)
    
    for i, v in enumerate(mean_weights.values):
        ax.text(i, v + 0.01, f'{v:.3f}', color='black', ha='center', fontsize=12)
        
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig(file_path, dpi=300)
    print(f"📈 Domain Attention weights saved to: {file_path}")
    plt.close()


def plot_roc_curve(all_labels, all_probas, class_names, file_path='./Output/roc_curve.png'):
    if len(class_names) != 2:
        return
    
    positive_class_index = 1 if 'Deepfake' in class_names else 0
    probas = np.array(all_probas)
    y_scores = probas[:, positive_class_index]
    
    fpr, tpr, thresholds = roc_curve(all_labels, y_scores)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 8), facecolor='white')
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=14)
    plt.ylabel('True Positive Rate', fontsize=14)
    plt.title('Receiver Operating Characteristic (ROC) Curve', fontsize=16)
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(file_path, dpi=300)
    print(f"📈 ROC Curve saved to: {file_path}")
    plt.close()


def plot_results(train_losses, val_losses, train_accuracies, val_accuracies, 
                 all_labels, all_preds, class_names):
    fig, axes = plt.subplots(1, 3, figsize=(21, 6), facecolor='white')
    
    # Loss Curves
    axes[0].plot(train_losses, label='Train Loss', color='mediumblue', linewidth=2)
    axes[0].plot(val_losses, label='Validation Loss', color='red', linewidth=2)
    axes[0].set_title('Training and Validation Loss', fontsize=16, fontweight='bold')
    axes[0].set_xlabel('Epoch', fontsize=14)
    axes[0].set_ylabel('Loss', fontsize=14)
    axes[0].legend(fontsize=12)
    axes[0].grid(True, linestyle='--', alpha=0.7)
    
    # Accuracy Curves
    axes[1].plot(train_accuracies, label='Train Accuracy', color='mediumblue', linewidth=2)
    axes[1].plot(val_accuracies, label='Validation Accuracy', color='red', linewidth=2)
    axes[1].set_title('Training and Validation Accuracy', fontsize=16, fontweight='bold')
    axes[1].set_xlabel('Epoch', fontsize=14)
    axes[1].set_ylabel('Accuracy (%)', fontsize=14)
    axes[1].legend(fontsize=12)
    axes[1].grid(True, linestyle='--', alpha=0.7)
    
    # Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds)
    sns.heatmap(cm, annot=True, fmt='d', 
                xticklabels=class_names, yticklabels=class_names, 
                cmap='Blues', cbar=True, ax=axes[2], 
                annot_kws={"size": 16})
    axes[2].set_title('Confusion Matrix', fontsize=16, fontweight='bold')
    axes[2].set_ylabel('True Label', fontsize=14)
    axes[2].set_xlabel('Predicted Label', fontsize=14)
    
    plt.tight_layout(pad=3.0)
    plt.savefig('./Output/training_summary.png', dpi=300, bbox_inches='tight')
    print("\n📈 Training summary saved to: ./Output/training_summary.png")
    plt.close(fig)


# ==================== TRAINING PIPELINE ====================

def train_model():
    """Train the multi-domain model"""
    print("\n" + "=" * 80)
    print("🚀 BINARY DEEPFAKE DETECTION TRAINING")
    print("=" * 80)
    print(f"Device: {CONFIG['device']}")
    print(f"CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    # Prepare data
    try:
        df, label_encoder, scaler_temporal, scaler_frequency = prepare_data()
    except Exception as e:
        print(f"❌ Error preparing data: {e}")
        raise
    
    # Ensure output directory exists
    os.makedirs('./Output', exist_ok=True)
    
    # Class distribution visualization
    plot_class_distribution(df)
    
    encoder_classes = label_encoder.classes_.tolist()
    class_names = encoder_classes
    print(f"Class Labels: {class_names}")
    
    # Split data
    print("\n📊 Splitting data...")
    train_df, test_df = train_test_split(df, test_size=0.2, 
                                         stratify=df['label_encoded'], 
                                         random_state=42)
    train_df, val_df = train_test_split(train_df, test_size=0.25,
                                        stratify=train_df['label_encoded'], 
                                        random_state=42)
    
    print(f"Train samples: {len(train_df)}")
    print(f"Validation samples: {len(val_df)}")
    print(f"Test samples: {len(test_df)}")
    
    # Data transforms - Convert numpy to PIL first, then apply transforms
    train_transform = transforms.Compose([
        transforms.Resize((CONFIG['image_size'], CONFIG['image_size'])),
        transforms.RandomHorizontalFlip(0.5),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((CONFIG['image_size'], CONFIG['image_size'])),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # Create datasets
    train_dataset = DeepfakeDataset(train_df['image_path'].tolist(), train_df, 
                                   train_df['label_encoded'].tolist(), train_transform)
    val_dataset = DeepfakeDataset(val_df['image_path'].tolist(), val_df, 
                                 val_df['label_encoded'].tolist(), val_transform)
    test_dataset = DeepfakeDataset(test_df['image_path'].tolist(), test_df, 
                                  test_df['label_encoded'].tolist(), val_transform)
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=CONFIG['batch_size'], 
                             shuffle=True, num_workers=CONFIG['num_workers'], 
                             pin_memory=CONFIG['pin_memory'])
    val_loader = DataLoader(val_dataset, batch_size=CONFIG['batch_size'], 
                           shuffle=False, num_workers=CONFIG['num_workers'], 
                           pin_memory=CONFIG['pin_memory'])
    test_loader = DataLoader(test_dataset, batch_size=CONFIG['batch_size'], 
                            shuffle=False, num_workers=CONFIG['num_workers'], 
                            pin_memory=CONFIG['pin_memory'])
    
    print("\n🏗️ Building Multi-Domain Detector (ViT-based)...")
    model = MultiDomainDeepfakeDetector(num_classes=CONFIG['num_classes'], use_vit=True).to(CONFIG['device'])
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=CONFIG['learning_rate'], weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CONFIG['epochs'])
    
    # Training tracking
    train_losses, val_losses = [], []
    train_accuracies, val_accuracies = [], []
    best_val_acc = 0.0
    patience_counter = 0
    
    print("\n🔥 Starting training...\n")
    
    for epoch in range(CONFIG['epochs']):
        start_time = time.time()
        
        # Training phase
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0
        
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{CONFIG['epochs']} [Train]", 
                         leave=False, ncols=100)
        
        for batch in train_pbar:
            if batch['label'].size(0) == 0:
                continue
            
            images = batch['image'].to(CONFIG['device'])
            temporal_features = batch['temporal_features'].to(CONFIG['device'])
            frequency_features = batch['frequency_features'].to(CONFIG['device'])
            labels = batch['label'].to(CONFIG['device'])
            
            optimizer.zero_grad()
            outputs, _ = model(images, temporal_features, frequency_features)
            loss = criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
            
            train_pbar.set_postfix({'loss': f'{loss.item():.4f}', 
                                   'acc': f'{100*train_correct/train_total:.2f}%'})
        
        # Validation phase
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        
        val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{CONFIG['epochs']} [Val]", 
                       leave=False, ncols=100)
        
        with torch.no_grad():
            for batch in val_pbar:
                if batch['label'].size(0) == 0:
                    continue
                
                images = batch['image'].to(CONFIG['device'])
                temporal_features = batch['temporal_features'].to(CONFIG['device'])
                frequency_features = batch['frequency_features'].to(CONFIG['device'])
                labels = batch['label'].to(CONFIG['device'])
                
                outputs, _ = model(images, temporal_features, frequency_features)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
                
                val_pbar.set_postfix({'loss': f'{loss.item():.4f}', 
                                     'acc': f'{100*val_correct/val_total:.2f}%'})
        
        # Calculate metrics
        train_acc = 100 * train_correct / train_total if train_total > 0 else 0
        val_acc = 100 * val_correct / val_total if val_total > 0 else 0
        
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        train_accuracies.append(train_acc)
        val_accuracies.append(val_acc)
        
        scheduler.step()
        
        epoch_time = time.time() - start_time
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_acc': val_acc,
                'train_acc': train_acc,
                'val_loss': avg_val_loss,
                'train_loss': avg_train_loss
            }
            torch.save(checkpoint, './Output/best_binary_model_weights.pth')
            
            with open('./Output/preprocessing_objects.pkl', 'wb') as f:
                pickle.dump({
                    'label_encoder': label_encoder,
                    'scaler_temporal': scaler_temporal,
                    'scaler_frequency': scaler_frequency,
                    'class_names': class_names
                }, f)
            
            print(f'✅ [Epoch {epoch+1}] New best model! Val Acc: {val_acc:.2f}% | Val Loss: {avg_val_loss:.4f} | Time: {epoch_time:.1f}s')
        else:
            patience_counter += 1
            print(f'📊 [Epoch {epoch+1}] Train: {train_acc:.2f}% | Val: {val_acc:.2f}% | Best: {best_val_acc:.2f}% | Patience: {patience_counter}/{CONFIG["early_stopping_patience"]} | Time: {epoch_time:.1f}s')
        
        # Memory cleanup
        if (epoch + 1) % 5 == 0:
            torch.cuda.empty_cache()
            gc.collect()
        
        # Early stopping
        if patience_counter >= CONFIG['early_stopping_patience']:
            print(f'\n⚠️ Early stopping triggered after {epoch + 1} epochs')
            break
    
    # Final evaluation
    print("\n" + "=" * 80)
    print("🧪 FINAL EVALUATION ON TEST SET")
    print("=" * 80)
    
    checkpoint = torch.load('./Output/best_binary_model_weights.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Loaded best model from epoch {checkpoint['epoch']+1}")
    model.eval()
    
    all_preds = []
    all_labels = []
    all_probas = []
    all_attention_weights = []
    
    test_pbar = tqdm(test_loader, desc="Testing", ncols=100)
    
    with torch.no_grad():
        for batch in test_pbar:
            if batch['label'].size(0) == 0:
                continue
            
            images = batch['image'].to(CONFIG['device'])
            temporal_features = batch['temporal_features'].to(CONFIG['device'])
            frequency_features = batch['frequency_features'].to(CONFIG['device'])
            labels = batch['label'].to(CONFIG['device'])
            
            outputs, attention_weights = model(images, temporal_features, frequency_features)
            probabilities = F.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probas.extend(probabilities.cpu().numpy())
            all_attention_weights.extend(attention_weights.cpu().numpy())
    
    # Calculate final metrics
    test_acc = accuracy_score(all_labels, all_preds)
    
    print("\n" + "=" * 80)
    print("📊 FINAL RESULTS")
    print("=" * 80)
    print(f"🎯 Test Accuracy: {test_acc * 100:.2f}%")
    print(f"🏆 Best Validation Accuracy: {best_val_acc:.2f}%")
    
    print("\n📋 Classification Report:")
    print(classification_report(all_labels, all_preds, target_names=class_names, 
                               digits=4, zero_division=0))
    
    # Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds)
    print("\n📈 Confusion Matrix:")
    print(f"{'':12} {'Predicted ' + class_names[0]:>20} {'Predicted ' + class_names[1]:>20}")
    for i, true_label in enumerate(class_names):
        print(f"True {true_label:8} {cm[i][0]:>20} {cm[i][1]:>20}")
    
    # Generate visualizations
    print("\n📊 Generating visualizations...")
    plot_results(train_losses, val_losses, train_accuracies, val_accuracies, 
                all_labels, all_preds, class_names)
    plot_roc_curve(all_labels, all_probas, class_names)
    plot_attention_weights(all_attention_weights)
    
    print("\n" + "=" * 80)
    print("✅ TRAINING COMPLETED SUCCESSFULLY!")
    print("=" * 80)
    print(f"💾 Model saved: ./Output/best_binary_model_weights.pth")
    print(f"💾 Preprocessing objects: ./Output/preprocessing_objects.pkl")
    print(f"💾 Visualizations: ./Output/")
    
    return model, label_encoder, scaler_temporal, scaler_frequency


# ==================== MAIN EXECUTION ====================
if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("BINARY DEEPFAKE DETECTION PIPELINE")
    print("Real vs. Deepfake Classification")
    print("=" * 80)
    
    import sys
    sys.setrecursionlimit(3000)
    
    try:
        # Set random seeds for reproducibility
        torch.manual_seed(42)
        np.random.seed(42)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(42)
        
        # Run training
        train_model()
        
    except KeyboardInterrupt:
        print("\n⚠️ Training interrupted by user")
    except Exception as e:
        print(f"\n❌ CRITICAL ERROR: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Cleanup
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        print("\n🧹 Cleanup completed")