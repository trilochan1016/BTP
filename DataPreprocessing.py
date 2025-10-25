import os
import shutil
import cv2
import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis, mode
from skimage.measure import shannon_entropy
import warnings
warnings.filterwarnings('ignore')

# ==== CONFIGURATION ====
input_root = './Data/FaceForensics++_C23'
output_root = './Output/Images/'
interval_ms = 1000

subfolders = ['original', 'Face2Face', 'FaceShifter', 'FaceSwap', 'NeuralTextures']
video_extensions = ('.mp4', '.avi', '.mkv', '.mov')

# ==== UTILITY FUNCTIONS ====
def clear_output_folders(root_folder):
    """Remove old output folders safely"""
    if os.path.exists(root_folder):
        for entry in os.listdir(root_folder):
            full_path = os.path.join(root_folder, entry)
            if os.path.isdir(full_path):
                print(f"Removing folder: {full_path}")
                try:
                    shutil.rmtree(full_path)
                except Exception as e:
                    print(f"Warning: Could not remove {full_path}: {e}")


def extract_frames_from_video(video_path, output_folder, video_index, interval_ms):
    """Extract frames using timestamp-based logic with error handling"""
    try:
        vidcap = cv2.VideoCapture(video_path)
        if not vidcap.isOpened():
            print(f"✗ Failed to open video {video_path}")
            return 0

        fps = vidcap.get(cv2.CAP_PROP_FPS)
        frame_count = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if fps == 0 or frame_count == 0:
            print(f"⚠️ Skipping invalid video: {video_path}")
            vidcap.release()
            return 0

        duration_ms = (frame_count / fps) * 1000
        timestamp_ms = 0
        saved_count = 0

        while timestamp_ms <= duration_ms:
            vidcap.set(cv2.CAP_PROP_POS_MSEC, timestamp_ms)
            ret, frame = vidcap.read()
            
            if not ret:
                break

            if frame is not None and frame.size > 0:
                saved_count += 1
                image_name = f"Video_{video_index:05d}_Image_{saved_count:06d}.jpg"
                image_path = os.path.join(output_folder, image_name)
                
                success = cv2.imwrite(image_path, frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
                if not success:
                    print(f"⚠️ Failed to write frame: {image_path}")

            timestamp_ms += interval_ms

        vidcap.release()
        print(f"✅ Extracted {saved_count} images from {os.path.basename(video_path)}")
        return saved_count
        
    except Exception as e:
        print(f"✗ Error processing video {video_path}: {e}")
        return 0


def compute_frame_features(frame):
    """Compute temporal features for one frame with error handling"""
    try:
        if frame is None or frame.size == 0:
            return None
            
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        pixels = gray.flatten().astype(np.float64)

        return {
            "mean_intensity": float(np.mean(pixels)),
            "variance": float(np.var(pixels)),
            "std_dev": float(np.std(pixels)),
            "skewness": float(skew(pixels)),
            "kurtosis": float(kurtosis(pixels)),
            "median_intensity": float(np.median(pixels)),
            "mode_intensity": float(mode(pixels, keepdims=True).mode[0]),
            "entropy": float(shannon_entropy(gray))
        }
    except Exception as e:
        print(f"⚠️ Error computing frame features: {e}")
        return None


def compute_dft_features(frame):
    """Compute DFT features with error handling"""
    try:
        if frame is None or frame.size == 0:
            return None
            
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        dft = np.fft.fft2(gray)
        dft_shift = np.fft.fftshift(dft)
        magnitude_spectrum = 20 * np.log(np.abs(dft_shift) + 1)

        return {
            "dft_mean": float(np.mean(magnitude_spectrum)),
            "dft_variance": float(np.var(magnitude_spectrum)),
            "dft_std": float(np.std(magnitude_spectrum)),
            "dft_median": float(np.median(magnitude_spectrum)),
            "dft_max": float(np.max(magnitude_spectrum))
        }
    except Exception as e:
        print(f"⚠️ Error computing DFT features: {e}")
        return None


# ==== STEP 1: FRAME EXTRACTION ====
def extract_all_frames():
    """Extract frames from all videos"""
    print("=" * 80)
    print("STEP 1: EXTRACTING FRAMES")
    print("=" * 80)
    
    clear_output_folders(output_root)

    for subfolder in subfolders:
        input_folder = os.path.join(input_root, subfolder)
        output_folder = os.path.join(output_root, subfolder)
        
        if not os.path.exists(input_folder):
            print(f"⚠️ Warning: Input folder not found: {input_folder}")
            continue
            
        os.makedirs(output_folder, exist_ok=True)

        video_files = sorted([f for f in os.listdir(input_folder) 
                            if f.lower().endswith(video_extensions)])

        print(f"\n📂 Processing {len(video_files)} videos in '{subfolder}'...")

        for idx, video_file in enumerate(video_files, start=1):
            video_path = os.path.join(input_folder, video_file)
            extract_frames_from_video(video_path, output_folder, idx, interval_ms)

    print("\n✅ Frame extraction completed!")


# ==== STEP 2: TEMPORAL FEATURE EXTRACTION ====
def extract_temporal_features():
    """Extract temporal features from frames"""
    print("\n" + "=" * 80)
    print("STEP 2: EXTRACTING TEMPORAL FEATURES")
    print("=" * 80)
    
    features_output_root = './Output/Features/'
    os.makedirs(features_output_root, exist_ok=True)

    for subfolder in subfolders:
        image_folder = os.path.join(output_root, subfolder)
        output_csv = os.path.join(features_output_root, f"{subfolder}_features.csv")

        if not os.path.exists(image_folder):
            print(f"⚠️ Skipping {subfolder}, no frames found.")
            continue

        feature_list = []
        print(f"\n📊 Extracting features from '{subfolder}'...")
        
        image_files = sorted([f for f in os.listdir(image_folder) 
                            if f.lower().endswith(('.jpg', '.jpeg', '.png'))])

        for idx, img_file in enumerate(image_files):
            img_path = os.path.join(image_folder, img_file)
            
            try:
                frame = cv2.imread(img_path)
                if frame is None:
                    print(f"⚠️ Skipping unreadable image: {img_file}")
                    continue

                features = compute_frame_features(frame)
                if features:
                    features["image_name"] = img_file
                    feature_list.append(features)
                    
                if (idx + 1) % 100 == 0:
                    print(f"   Processed {idx + 1}/{len(image_files)} images")
                    
            except Exception as e:
                print(f"⚠️ Error processing {img_file}: {e}")
                continue

        if feature_list:
            df = pd.DataFrame(feature_list)
            df.to_csv(output_csv, index=False)
            print(f"✅ Saved {len(df)} features to {output_csv}")
        else:
            print(f"⚠️ No features extracted for {subfolder}")

    print("\n✅ Temporal feature extraction completed!")


# ==== STEP 3: DFT FEATURE EXTRACTION ====
def extract_dft_features():
    """Extract DFT features from frames"""
    print("\n" + "=" * 80)
    print("STEP 3: EXTRACTING DFT FEATURES")
    print("=" * 80)
    
    dft_features_output_root = './Output/DFT_Features/'
    os.makedirs(dft_features_output_root, exist_ok=True)

    for subfolder in subfolders:
        image_folder = os.path.join(output_root, subfolder)
        output_csv = os.path.join(dft_features_output_root, f"{subfolder}_dft_features.csv")

        if not os.path.exists(image_folder):
            print(f"⚠️ Skipping {subfolder}, no frames found.")
            continue

        feature_list = []
        print(f"\n🔬 Computing DFT features for '{subfolder}'...")
        
        image_files = sorted([f for f in os.listdir(image_folder) 
                            if f.lower().endswith(('.jpg', '.jpeg', '.png'))])

        for idx, img_file in enumerate(image_files):
            img_path = os.path.join(image_folder, img_file)
            
            try:
                frame = cv2.imread(img_path)
                if frame is None:
                    continue

                features = compute_dft_features(frame)
                if features:
                    features["image_name"] = img_file
                    feature_list.append(features)
                    
                if (idx + 1) % 100 == 0:
                    print(f"   Processed {idx + 1}/{len(image_files)} images")
                    
            except Exception as e:
                print(f"⚠️ Error processing {img_file}: {e}")
                continue

        if feature_list:
            df = pd.DataFrame(feature_list)
            df.to_csv(output_csv, index=False)
            print(f"✅ Saved {len(df)} DFT features to {output_csv}")

    print("\n✅ DFT feature extraction completed!")


# ==== STEP 4: MERGE FEATURES ====
def merge_features():
    """Merge temporal and DFT features"""
    print("\n" + "=" * 80)
    print("STEP 4: MERGING FEATURES")
    print("=" * 80)
    
    merged_features_output_root = './Output/Merged_Features/'
    os.makedirs(merged_features_output_root, exist_ok=True)
    
    features_output_root = './Output/Features/'
    dft_features_output_root = './Output/DFT_Features/'

    for subfolder in subfolders:
        temporal_csv = os.path.join(features_output_root, f"{subfolder}_features.csv")
        dft_csv = os.path.join(dft_features_output_root, f"{subfolder}_dft_features.csv")

        if not (os.path.exists(temporal_csv) and os.path.exists(dft_csv)):
            print(f"⚠️ Missing features for {subfolder}, skipping merge.")
            continue

        df_temporal = pd.read_csv(temporal_csv)
        df_dft = pd.read_csv(dft_csv)

        df_merged = pd.merge(df_temporal, df_dft, on="image_name", how="inner")
        df_merged["subfolder"] = subfolder

        merged_csv = os.path.join(merged_features_output_root, f"{subfolder}_merged_features.csv")
        df_merged.to_csv(merged_csv, index=False)
        print(f"✅ Saved {len(df_merged)} merged features to {merged_csv}")

    print("\n✅ Feature merging completed!")

# ==================== MAIN EXECUTION ====================
if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("MULTI-DOMAIN DEEPFAKE DETECTION PIPELINE")
    print("=" * 80)
    
    try:
        # Step 1: Extract frames
        extract_all_frames()
        
        # Step 2: Extract temporal features
        extract_temporal_features()
        
        # Step 3: Extract DFT features
        extract_dft_features()
        
        # Step 4: Merge features
        merge_features()
        
    except Exception as e:
        print(f"\n❌ CRITICAL ERROR: {e}")
        import traceback
        traceback.print_exc()