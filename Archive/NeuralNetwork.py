import os
import shutil
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
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from torchvision import transforms
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import warnings
warnings.filterwarnings('ignore')

# ==================== NEURAL NETWORK ARCHITECTURES ====================

CONFIG = {
    'batch_size': 16,
    'learning_rate': 0.0001,
    'epochs': 50,
    'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
    'num_classes': 5,
    'image_size': 224,
}


class DeepfakeDataset(Dataset):
    """Dataset class with robust error handling - FIXED VERSION"""
    def __init__(self, image_paths, features_df, labels, transform=None):
        self.image_paths = image_paths
        self.features_df = features_df
        self.labels = labels
        self.transform = transform
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        try:
            img_path = self.image_paths[idx]
            image = cv2.imread(img_path)
            
            if image is None:
                raise ValueError(f"Could not read image: {img_path}")
                
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            if self.transform:
                image = self.transform(image)
            
            img_name = os.path.basename(img_path)
            feature_row = self.features_df[self.features_df['image_name'] == img_name]
            
            if len(feature_row) == 0:
                temporal_features = torch.zeros(8)
                frequency_features = torch.zeros(5)
            else:
                temporal_cols = ['mean_intensity', 'variance', 'std_dev', 'skewness', 
                               'kurtosis', 'median_intensity', 'mode_intensity', 'entropy']
                frequency_cols = ['dft_mean', 'dft_variance', 'dft_std', 'dft_median', 'dft_max']
                
                # FIXED: Proper indexing using .iloc[0]
                temporal_features = torch.tensor(feature_row[temporal_cols].iloc[0].values, dtype=torch.float32)
                frequency_features = torch.tensor(feature_row[frequency_cols].iloc[0].values, dtype=torch.float32)
            
            label = self.labels[idx]
            
            return {
                'image': image,
                'temporal_features': temporal_features,
                'frequency_features': frequency_features,
                'label': label
            }
        except Exception as e:
            print(f"Error loading sample {idx}: {e}")
            # Return a default sample
            return {
                'image': torch.zeros(3, 224, 224),
                'temporal_features': torch.zeros(8),
                'frequency_features': torch.zeros(5),
                'label': 0
            }


class CustomVGG(nn.Module):
    """Custom VGG architecture for spatial feature extraction"""
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


class PatchEmbedding(nn.Module):
    """Patch embedding for Vision Transformer"""
    def __init__(self, img_size=224, patch_size=16, in_channels=3, embed_dim=768):
        super().__init__()
        self.n_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        
    def forward(self, x):
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        return x


class CustomViT(nn.Module):
    """Vision Transformer for spatial analysis"""
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
    """LSTM + GRU hybrid for temporal analysis"""
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
    """Frequency domain analyzer"""
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
    """Multi-domain ensemble architecture"""
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


def prepare_data():
    """Prepare data for training with validation"""
    merged_features_path = './Output/Merged_Features/'
    images_path = './Output/Images/'
    
    all_data = []
    
    for subfolder in subfolders:
        csv_path = os.path.join(merged_features_path, f"{subfolder}_merged_features.csv")
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            df['label'] = subfolder
            df['image_path'] = df['image_name'].apply(
                lambda x: os.path.join(images_path, subfolder, x)
            )
            
            # Validate image paths
            df = df[df['image_path'].apply(os.path.exists)]
            all_data.append(df)
        else:
            print(f"⚠️ Warning: CSV not found for {subfolder}")
    
    if not all_data:
        raise ValueError("No data found! Please run feature extraction first.")
    
    combined_df = pd.concat(all_data, ignore_index=True)
    print(f"Total samples: {len(combined_df)}")
    
    label_encoder = LabelEncoder()
    combined_df['label_encoded'] = label_encoder.fit_transform(combined_df['label'])
    
    temporal_cols = ['mean_intensity', 'variance', 'std_dev', 'skewness', 
                    'kurtosis', 'median_intensity', 'mode_intensity', 'entropy']
    frequency_cols = ['dft_mean', 'dft_variance', 'dft_std', 'dft_median', 'dft_max']
    
    scaler_temporal = StandardScaler()
    scaler_frequency = StandardScaler()
    
    combined_df[temporal_cols] = scaler_temporal.fit_transform(combined_df[temporal_cols])
    combined_df[frequency_cols] = scaler_frequency.fit_transform(combined_df[frequency_cols])
    
    return combined_df, label_encoder, scaler_temporal, scaler_frequency


def train_model():
    """Train the multi-domain model with comprehensive error handling"""
    print("\n" + "=" * 80)
    print("🚀 STARTING MULTI-DOMAIN DEEPFAKE DETECTION TRAINING")
    print("=" * 80)
    print(f"Device: {CONFIG['device']}")
    
    try:
        df, label_encoder, scaler_temporal, scaler_frequency = prepare_data()
    except Exception as e:
        print(f"❌ Error preparing data: {e}")
        return
    
    train_df, test_df = train_test_split(df, test_size=0.2, 
                                         stratify=df['label_encoded'], 
                                         random_state=42)
    train_df, val_df = train_test_split(train_df, test_size=0.25, 
                                        stratify=train_df['label_encoded'], 
                                        random_state=42)
    
    print(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
    
    train_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((CONFIG['image_size'], CONFIG['image_size'])),
        transforms.RandomHorizontalFlip(0.5),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((CONFIG['image_size'], CONFIG['image_size'])),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    train_dataset = DeepfakeDataset(train_df['image_path'].tolist(), train_df, 
                                   train_df['label_encoded'].tolist(), train_transform)
    val_dataset = DeepfakeDataset(val_df['image_path'].tolist(), val_df, 
                                 val_df['label_encoded'].tolist(), val_transform)
    test_dataset = DeepfakeDataset(test_df['image_path'].tolist(), test_df, 
                                  test_df['label_encoded'].tolist(), val_transform)
    
    train_loader = DataLoader(train_dataset, batch_size=CONFIG['batch_size'], 
                             shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=CONFIG['batch_size'], 
                           shuffle=False, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=CONFIG['batch_size'], 
                            shuffle=False, num_workers=2, pin_memory=True)
    
    print("\n🏗️ Building Multi-Domain Architecture...")
    model = MultiDomainDeepfakeDetector(num_classes=CONFIG['num_classes']).to(CONFIG['device'])
    
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=CONFIG['learning_rate'], weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CONFIG['epochs'])
    
    train_losses, val_losses = [], []
    train_accuracies, val_accuracies = [], []
    best_val_acc = 0.0
    patience = 10
    patience_counter = 0
    
    print("\n🔥 Starting training...")
    
    for epoch in range(CONFIG['epochs']):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for batch_idx, batch in enumerate(train_loader):
            try:
                images = batch['image'].to(CONFIG['device'])
                temporal_features = batch['temporal_features'].to(CONFIG['device'])
                frequency_features = batch['frequency_features'].to(CONFIG['device'])
                labels = batch['label'].to(CONFIG['device'])
                
                optimizer.zero_grad()
                outputs, attention_weights = model(images, temporal_features, frequency_features)
                loss = criterion(outputs, labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                
                train_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                train_total += labels.size(0)
                train_correct += (predicted == labels).sum().item()
                
            except Exception as e:
                print(f"⚠️ Error in training batch {batch_idx}: {e}")
                continue
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for batch in val_loader:
                try:
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
                    
                except Exception as e:
                    print(f"⚠️ Error in validation batch: {e}")
                    continue
        
        train_acc = 100 * train_correct / train_total if train_total > 0 else 0
        val_acc = 100 * val_correct / val_total if val_total > 0 else 0
        
        train_losses.append(train_loss / len(train_loader))
        val_losses.append(val_loss / len(val_loader))
        train_accuracies.append(train_acc)
        val_accuracies.append(val_acc)
        
        scheduler.step()
        
        # Save best model - FIXED for PyTorch 2.6
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'train_acc': train_acc
            }, './Output/best_model_weights.pth')
            
            # Save scalers and encoder separately as pickle
            import pickle
            with open('./Output/preprocessing_objects.pkl', 'wb') as f:
                pickle.dump({
                    'label_encoder': label_encoder,
                    'scaler_temporal': scaler_temporal,
                    'scaler_frequency': scaler_frequency
                }, f)
            
            print(f'✅ New best model saved! Val Acc: {val_acc:.2f}%')
        else:
            patience_counter += 1
        
        # Print progress
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f'\nEpoch [{epoch+1}/{CONFIG["epochs"]}]')
            print(f'Train Loss: {train_losses[-1]:.4f}, Train Acc: {train_acc:.2f}%')
            print(f'Val Loss: {val_losses[-1]:.4f}, Val Acc: {val_acc:.2f}%')
            print(f'Best Val Acc: {best_val_acc:.2f}%')
            print('-' * 50)
        
        # Early stopping
        if patience_counter >= patience:
            print(f'\n⚠️ Early stopping triggered after {epoch + 1} epochs')
            break
    
    # Final evaluation on test set - FIXED for PyTorch 2.6
    print("\n🧪 Evaluating on test set...")
    checkpoint = torch.load('./Output/best_model_weights.pth', weights_only=True)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in test_loader:
            try:
                images = batch['image'].to(CONFIG['device'])
                temporal_features = batch['temporal_features'].to(CONFIG['device'])
                frequency_features = batch['frequency_features'].to(CONFIG['device'])
                labels = batch['label'].to(CONFIG['device'])
                
                outputs, _ = model(images, temporal_features, frequency_features)
                _, predicted = torch.max(outputs, 1)
                
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                
            except Exception as e:
                print(f"⚠️ Error in test batch: {e}")
                continue
    
    test_acc = accuracy_score(all_labels, all_preds)
    print(f"\n🎯 Final Test Accuracy: {test_acc * 100:.2f}%")
    
    class_names = label_encoder.classes_
    print("\n📊 Classification Report:")
    print(classification_report(all_labels, all_preds, target_names=class_names, zero_division=0))
    
    # Plot training curves
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 3, 2)
    plt.plot(train_accuracies, label='Train Accuracy')
    plt.plot(val_accuracies, label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 3, 3)
    cm = confusion_matrix(all_labels, all_preds)
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=class_names, yticklabels=class_names, cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    plt.tight_layout()
    plt.savefig('./Output/training_results.png', dpi=300, bbox_inches='tight')
    print("\n📈 Training curves saved to: ./Output/training_results.png")
    
    print(f"\n✅ Training completed! Best validation accuracy: {best_val_acc:.2f}%")
    print("💾 Model weights saved to: ./Output/best_model_weights.pth")
    print("💾 Preprocessing objects saved to: ./Output/preprocessing_objects.pkl")
    
    return model, label_encoder, scaler_temporal, scaler_frequency


# ==================== MAIN EXECUTION ====================
if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("MULTI-DOMAIN DEEPFAKE DETECTION PIPELINE")
    print("=" * 80)
    
    try:
        # Step 5: Train model
        train_model()
        
        print("\n" + "=" * 80)
        print("✅ PIPELINE COMPLETED SUCCESSFULLY!")
        print("=" * 80)
        
    except Exception as e:
        print(f"\n❌ CRITICAL ERROR: {e}")
        import traceback
        traceback.print_exc()