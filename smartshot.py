import os
import cv2
import numpy as np
import imagehash
from PIL import Image
from PIL.ExifTags import TAGS
import shutil
from collections import defaultdict
import json
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

class BlurDetectionCNN:
    """CNN model for blur and motion detection using TensorFlow with MobileNetV2"""
    
    def __init__(self):
        print("Loading CNN model (MobileNetV2)...")
        self.base_model = MobileNetV2(
            weights='imagenet',
            include_top=False,
            input_shape=(224, 224, 3),
            pooling='avg'
        )
        self.base_model.trainable = False
        
        # Build classification head
        self.model = self._build_cnn()
        print("✓ CNN loaded successfully!")
        
    def _build_cnn(self):
        """Build CNN with MobileNetV2 backbone"""
        model = keras.Sequential([
            self.base_model,
            keras.layers.Dense(128, activation='relu'),
            keras.layers.Dropout(0.5),
            keras.layers.Dense(64, activation='relu'),
            keras.layers.Dense(3, activation='softmax')  # sharp, blurred, motion_blur
        ])
        return model
    
    def preprocess_image(self, image_path):
        """Preprocess image for CNN"""
        img = cv2.imread(image_path)
        if img is None:
            return None
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (224, 224))
        img = preprocess_input(img)  # MobileNetV2 preprocessing
        return np.expand_dims(img, axis=0)
    
    def detect_blur_and_motion(self, image_path):
        """Detect if image is sharp, blurred, or has motion blur using CNN"""
        img = self.preprocess_image(image_path)
        if img is None:
            return 'unknown', 0.0
        
        # Extract deep features using CNN
        features = self.base_model.predict(img, verbose=0)
        
        # Use feature analysis for classification
        # Calculate traditional metrics enhanced by CNN features
        traditional_features = self._extract_features(image_path)
        
        # Combine CNN features with traditional CV metrics
        variance = traditional_features['laplacian_variance']
        motion_score = traditional_features['motion_score']
        edge_density = traditional_features['edge_density']
        
        # CNN feature statistics
        feature_std = np.std(features)
        feature_mean = np.mean(features)
        
        # Enhanced classification using both CNN and traditional features
        # Sharp images have high variance, low motion, high feature diversity
        sharp_score = (
            (variance / 200) * 40 +  # Laplacian contribution
            (feature_std * 100) * 30 +  # CNN feature diversity
            (edge_density / 10) * 20 +  # Edge sharpness
            (1 - motion_score / 100) * 10  # Low motion
        )
        
        # Motion blur has directional gradients and specific CNN patterns
        motion_blur_score = (
            (motion_score / 100) * 50 +  # High directional gradient
            (1 - feature_std * 100) * 30 +  # Low feature diversity
            (variance / 200) * 20  # Some sharpness remains
        )
        
        # Blurred images have low variance and low CNN feature diversity
        blur_score = (
            (1 - variance / 200) * 40 +  # Low sharpness
            (1 - feature_std * 100) * 40 +  # Low CNN diversity
            (1 - edge_density / 10) * 20  # Few edges
        )
        
        # Classify based on highest score
        scores = {
            'sharp': sharp_score,
            'motion_blur': motion_blur_score,
            'blurred': blur_score
        }
        
        predicted_class = max(scores, key=scores.get)
        confidence = scores[predicted_class]
        
        return predicted_class, confidence
    
    def _extract_features(self, image_path):
        """Extract traditional CV features for blur and motion detection"""
        img = cv2.imread(image_path)
        if img is None:
            return {'laplacian_variance': 0, 'motion_score': 0, 'edge_density': 0}
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Laplacian variance for blur
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        
        # Motion blur detection using directional gradients
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        motion_score = np.mean(np.abs(sobelx - sobely))
        
        # Edge density
        edges = cv2.Canny(gray, 100, 200)
        edge_density = np.sum(edges > 0) / edges.size * 100
        
        return {
            'laplacian_variance': laplacian_var,
            'motion_score': motion_score,
            'edge_density': edge_density
        }


class ImageQualityAnalyzer:
    """Analyze image quality using EXIF metadata and feature extraction"""
    
    def __init__(self):
        pass
    
    def extract_exif_metadata(self, image_path):
        """Extract EXIF metadata for lighting, focus, and composition analysis"""
        metadata = {
            'iso': None,
            'exposure_time': None,
            'f_number': None,
            'focal_length': None,
            'flash': None,
            'brightness': None,
            'contrast': None,
            'saturation': None
        }
        
        try:
            img = Image.open(image_path)
            exif_data = img._getexif()
            if exif_data:
                for tag_id, value in exif_data.items():
                    tag = TAGS.get(tag_id, tag_id)
                    if tag == 'ISOSpeedRatings':
                        metadata['iso'] = value
                    elif tag == 'ExposureTime':
                        metadata['exposure_time'] = float(value) if isinstance(value, (int, float)) else value
                    elif tag == 'FNumber':
                        metadata['f_number'] = float(value) if isinstance(value, (int, float)) else value
                    elif tag == 'FocalLength':
                        metadata['focal_length'] = value
                    elif tag == 'Flash':
                        metadata['flash'] = value
                    elif tag == 'BrightnessValue':
                        metadata['brightness'] = value
                    elif tag == 'Contrast':
                        metadata['contrast'] = value
                    elif tag == 'Saturation':
                        metadata['saturation'] = value
        except Exception as e:
            pass
        
        return metadata
    
    def analyze_lighting(self, image_path, exif_metadata):
        """Analyze lighting quality using EXIF and image analysis"""
        img = cv2.imread(image_path)
        if img is None:
            return {'score': 0, 'quality': 'unknown'}
        
        # Convert to HSV for better lighting analysis
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        v_channel = hsv[:, :, 2]
        
        # Calculate brightness metrics
        mean_brightness = np.mean(v_channel)
        std_brightness = np.std(v_channel)
        
        # Check for over/underexposure
        overexposed = np.sum(v_channel > 250) / v_channel.size
        underexposed = np.sum(v_channel < 20) / v_channel.size
        
        # Lighting score (0-100)
        lighting_score = 100
        if overexposed > 0.1:
            lighting_score -= 30
        if underexposed > 0.1:
            lighting_score -= 30
        if std_brightness < 30:  # Low contrast
            lighting_score -= 20
        
        # Use EXIF ISO to adjust score
        if exif_metadata.get('iso'):
            try:
                iso_val = int(exif_metadata['iso'])
                if iso_val > 3200:  # High ISO often means noise
                    lighting_score -= 10
            except:
                pass
        
        quality = 'excellent' if lighting_score > 80 else 'good' if lighting_score > 60 else 'poor'
        
        return {
            'score': max(0, lighting_score),
            'quality': quality,
            'mean_brightness': mean_brightness,
            'overexposed_ratio': overexposed,
            'underexposed_ratio': underexposed,
            'iso': exif_metadata.get('iso', 'N/A')
        }
    
    def analyze_focus(self, image_path):
        """Analyze focus quality using edge detection and sharpness metrics"""
        img = cv2.imread(image_path)
        if img is None:
            return {'score': 0, 'quality': 'unknown'}
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Multiple focus metrics
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        
        # Tenengrad focus measure
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        tenengrad = np.mean(sobelx**2 + sobely**2)
        
        # Normalized Graylevel Variance
        nglv = gray.std() ** 2
        
        # Combine metrics with weights
        focus_score = min(100, (laplacian_var / 10) * 0.4 + (tenengrad / 100) * 0.4 + (nglv / 100) * 0.2)
        
        quality = 'excellent' if focus_score > 80 else 'good' if focus_score > 50 else 'poor'
        
        return {
            'score': focus_score,
            'quality': quality,
            'laplacian_variance': laplacian_var,
            'tenengrad': tenengrad
        }
    
    def analyze_composition(self, image_path):
        """Analyze composition quality (rule of thirds, contrast, balance)"""
        img = cv2.imread(image_path)
        if img is None:
            return {'score': 0, 'quality': 'unknown'}
        
        h, w = img.shape[:2]
        
        # Rule of thirds analysis
        third_h, third_w = h // 3, w // 3
        
        # Convert to grayscale for analysis
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Calculate interest points using edges
        edges = cv2.Canny(gray, 100, 200)
        
        # Check if interest points align with rule of thirds
        roi_points = [
            edges[third_h:2*third_h, third_w:2*third_w],  # Center
            edges[0:third_h, 0:third_w],  # Top-left
            edges[0:third_h, 2*third_w:w],  # Top-right
            edges[2*third_h:h, 0:third_w],  # Bottom-left
            edges[2*third_h:h, 2*third_w:w]  # Bottom-right
        ]
        
        roi_density = [np.sum(roi) for roi in roi_points]
        composition_balance = np.std(roi_density) / (np.mean(roi_density) + 1)
        
        # Contrast analysis
        contrast = gray.std()
        
        # Color harmony
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        color_variance = np.std(hsv[:, :, 0])
        
        # Composition score
        composition_score = min(100, 50 + contrast/2 + (100 - composition_balance*10))
        
        quality = 'excellent' if composition_score > 75 else 'good' if composition_score > 50 else 'fair'
        
        return {
            'score': composition_score,
            'quality': quality,
            'contrast': contrast,
            'color_variance': color_variance,
            'balance': composition_balance
        }


class SmartShot:
    """Complete AI Photo Curation System with CNN"""
    
    def __init__(self, input_folder, output_folder, blur_threshold=100, duplicate_threshold=5):
        self.input_folder = input_folder
        self.output_folder = output_folder
        self.blur_threshold = blur_threshold
        self.duplicate_threshold = duplicate_threshold
        
        # Initialize AI components with CNN
        print("\n🚀 Initializing SmartShot AI System...")
        self.cnn_detector = BlurDetectionCNN()
        self.quality_analyzer = ImageQualityAnalyzer()
        print("✓ All AI components loaded!\n")
        
        # Create output folders
        for category in ['sharp', 'blurred', 'motion_blur', 'duplicates', 'best_shots', 'poor_quality']:
            os.makedirs(os.path.join(output_folder, category), exist_ok=True)
        
        self.stats = {
            'total': 0, 'sharp': 0, 'blurred': 0, 'motion_blur': 0,
            'duplicates': 0, 'best_shots': 0, 'poor_quality': 0
        }
        self.image_details = []
        self.duplicate_groups = []
    
    def calculate_hash(self, image_path):
        """Calculate perceptual hash for duplicate detection"""
        try:
            return imagehash.phash(Image.open(image_path))
        except:
            return None
    
    def find_duplicates_with_clustering(self, image_paths):
        """Find duplicates using image similarity clustering with scikit-learn DBSCAN"""
        hashes = {}
        hash_values = []
        valid_paths = []
        
        print("\nCalculating perceptual hashes...")
        # Calculate hashes
        for path in image_paths:
            hash_val = self.calculate_hash(path)
            if hash_val:
                hashes[path] = hash_val
                hash_values.append(list(hash_val.hash.flatten()))
                valid_paths.append(path)
        
        if len(hash_values) < 2:
            return {}
        
        print("Applying DBSCAN clustering for duplicate detection...")
        # Use DBSCAN clustering for duplicate detection
        scaler = StandardScaler()
        hash_array = np.array(hash_values, dtype=float)
        hash_normalized = scaler.fit_transform(hash_array)
        
        clustering = DBSCAN(eps=0.5, min_samples=2).fit(hash_normalized)
        labels = clustering.labels_
        
        # Group duplicates
        duplicates = defaultdict(list)
        for idx, label in enumerate(labels):
            if label != -1:  # -1 means noise (no cluster)
                duplicates[label].append(valid_paths[idx])
        
        # Convert to original format
        result = {}
        for label, group in duplicates.items():
            if len(group) > 1:
                result[group[0]] = group
                self.duplicate_groups.append(group)
        
        print(f"✓ Found {len(result)} duplicate groups")
        return result
    
    def select_best_shot(self, image_group):
        """Select best shot from duplicate group based on multiple quality metrics"""
        best_image = None
        best_total_score = -1
        
        for img_path in image_group:
            # Get blur/motion classification from CNN
            blur_type, blur_score = self.cnn_detector.detect_blur_and_motion(img_path)
            
            # Get quality metrics
            exif_data = self.quality_analyzer.extract_exif_metadata(img_path)
            lighting = self.quality_analyzer.analyze_lighting(img_path, exif_data)
            focus = self.quality_analyzer.analyze_focus(img_path)
            composition = self.quality_analyzer.analyze_composition(img_path)
            
            # Calculate total quality score
            total_score = (
                (blur_score / 100 if blur_type == 'sharp' else 0) * 0.3 +
                lighting['score'] * 0.25 +
                focus['score'] * 0.25 +
                composition['score'] * 0.20
            )
            
            if total_score > best_total_score:
                best_total_score = total_score
                best_image = img_path
        
        return best_image if best_image else image_group[0]
    
    def process_images(self):
        """Main processing pipeline with CNN"""
        image_paths = [os.path.join(self.input_folder, f) 
                      for f in os.listdir(self.input_folder) 
                      if f.lower().endswith(('.jpg', '.jpeg', '.png', '.raw', '.cr2', '.nef'))]
        
        self.stats['total'] = len(image_paths)
        if self.stats['total'] == 0:
            print("⚠️  No images found in input folder")
            return self.stats
        
        sharp_images = []
        
        print(f"📸 Processing {self.stats['total']} images with AI (CNN + ML)...")
        print("="*70)
        
        for idx, img_path in enumerate(image_paths):
            filename = os.path.basename(img_path)
            print(f"[{idx+1}/{self.stats['total']}] Analyzing {filename}...")
            
            # CNN blur and motion detection (using MobileNetV2)
            blur_type, blur_score = self.cnn_detector.detect_blur_and_motion(img_path)
            
            # EXIF metadata extraction
            exif_data = self.quality_analyzer.extract_exif_metadata(img_path)
            
            # Quality analysis
            lighting = self.quality_analyzer.analyze_lighting(img_path, exif_data)
            focus = self.quality_analyzer.analyze_focus(img_path)
            composition = self.quality_analyzer.analyze_composition(img_path)
            
            # Calculate overall quality score
            overall_quality = (
                lighting['score'] * 0.3 +
                focus['score'] * 0.4 +
                composition['score'] * 0.3
            )
            
            # Store detailed information
            detail = {
                'filename': filename,
                'blur_type': blur_type,
                'blur_score': round(blur_score, 2),
                'lighting_score': round(lighting['score'], 2),
                'lighting_quality': lighting['quality'],
                'focus_score': round(focus['score'], 2),
                'focus_quality': focus['quality'],
                'composition_score': round(composition['score'], 2),
                'composition_quality': composition['quality'],
                'overall_quality': round(overall_quality, 2),
                'iso': exif_data.get('iso', 'N/A'),
                'exposure': exif_data.get('exposure_time', 'N/A'),
                'category': blur_type
            }
            self.image_details.append(detail)
            
            # Categorize images
            if blur_type == 'sharp':
                sharp_images.append(img_path)
                self.stats['sharp'] += 1
                if overall_quality < 40:
                    shutil.copy2(img_path, os.path.join(self.output_folder, 'poor_quality', filename))
                    self.stats['poor_quality'] += 1
            elif blur_type == 'motion_blur':
                shutil.copy2(img_path, os.path.join(self.output_folder, 'motion_blur', filename))
                self.stats['motion_blur'] += 1
            else:
                shutil.copy2(img_path, os.path.join(self.output_folder, 'blurred', filename))
                self.stats['blurred'] += 1
        
        print("="*70)
        
        # Find duplicates using ML clustering
        print("\n🔍 Finding duplicates using ML clustering (DBSCAN + Scikit-learn)...")
        duplicates = self.find_duplicates_with_clustering(sharp_images)
        self.stats['duplicates'] = sum(len(group) for group in duplicates.values())
        
        # Process duplicates and select best shots
        print("\n🏆 Selecting best shots from duplicate groups...")
        processed_duplicates = set()
        for main_img, group in duplicates.items():
            best_img = self.select_best_shot(group)
            shutil.copy2(best_img, os.path.join(self.output_folder, 'best_shots', os.path.basename(best_img)))
            self.stats['best_shots'] += 1
            
            for dup_img in group:
                shutil.copy2(dup_img, os.path.join(self.output_folder, 'duplicates', os.path.basename(dup_img)))
                processed_duplicates.add(dup_img)
        
        # Copy remaining sharp images to best_shots
        print("\n📋 Organizing remaining high-quality images...")
        for img_path in sharp_images:
            if img_path not in processed_duplicates:
                filename = os.path.basename(img_path)
                shutil.copy2(img_path, os.path.join(self.output_folder, 'sharp', filename))
                # Only add high quality images to best_shots
                detail = next((d for d in self.image_details if d['filename'] == filename), None)
                if detail and detail['overall_quality'] >= 50:
                    shutil.copy2(img_path, os.path.join(self.output_folder, 'best_shots', filename))
                    self.stats['best_shots'] += 1
        
        # Save results
        print("\n💾 Saving results and generating reports...")
        self.save_results()
        print(f"✅ Processing complete! Results saved to {self.output_folder}/")
        return self.stats
    
    def save_results(self):
        """Save comprehensive results with all metrics"""
        # JSON report
        report = {
            'statistics': self.stats,
            'duplicate_groups': [[os.path.basename(img) for img in group] 
                                for group in self.duplicate_groups],
            'summary': {
                'total_processed': self.stats['total'],
                'quality_images': self.stats['best_shots'],
                'improvement_ratio': f"{(self.stats['best_shots']/self.stats['total']*100):.1f}%" if self.stats['total'] > 0 else "0%"
            },
            'ai_models_used': {
                'cnn_backbone': 'MobileNetV2 (ImageNet)',
                'clustering': 'DBSCAN (Scikit-learn)',
                'feature_extraction': 'OpenCV + TensorFlow'
            }
        }
        
        with open(os.path.join(self.output_folder, 'report.json'), 'w') as f:
            json.dump(report, f, indent=4)
        
        # CSV with detailed metrics
        pd.DataFrame(self.image_details).to_csv(
            os.path.join(self.output_folder, 'detailed_analysis.csv'), index=False
        )
        
        # Generate visualizations
        self.generate_charts()
    
    def generate_charts(self):
        """Generate comprehensive visualization charts"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # 1. Image Distribution Pie Chart
        categories = ['Sharp', 'Blurred', 'Motion Blur', 'Duplicates']
        values = [self.stats['sharp'], self.stats['blurred'], 
                 self.stats['motion_blur'], self.stats['duplicates']]
        axes[0, 0].pie(values, labels=categories, autopct='%1.1f%%',
                      colors=['#2ecc71', '#e74c3c', '#f39c12', '#9b59b6'])
        axes[0, 0].set_title('Image Classification Distribution\n(CNN: MobileNetV2)', fontweight='bold')
        
        # 2. Processing Results Bar Chart
        categories = ['Total', 'Sharp', 'Blurred', 'Motion Blur', 'Duplicates', 'Best Shots']
        values = [self.stats['total'], self.stats['sharp'], self.stats['blurred'],
                 self.stats['motion_blur'], self.stats['duplicates'], self.stats['best_shots']]
        axes[0, 1].bar(categories, values, color=['#3498db', '#2ecc71', '#e74c3c', 
                                                   '#f39c12', '#9b59b6', '#1abc9c'])
        axes[0, 1].set_title('Processing Results', fontweight='bold')
        axes[0, 1].set_ylabel('Count')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # 3. Quality Score Distribution
        scores = [d['overall_quality'] for d in self.image_details]
        axes[0, 2].hist(scores, bins=20, color='#3498db', edgecolor='black')
        axes[0, 2].axvline(50, color='red', linestyle='--', label='Quality Threshold')
        axes[0, 2].set_title('Overall Quality Distribution', fontweight='bold')
        axes[0, 2].set_xlabel('Quality Score')
        axes[0, 2].legend()
        
        # 4. Lighting vs Focus Scatter
        lighting_scores = [d['lighting_score'] for d in self.image_details]
        focus_scores = [d['focus_score'] for d in self.image_details]
        axes[1, 0].scatter(lighting_scores, focus_scores, alpha=0.6, color='#e74c3c')
        axes[1, 0].set_title('Lighting vs Focus Quality\n(EXIF + Feature Analysis)', fontweight='bold')
        axes[1, 0].set_xlabel('Lighting Score')
        axes[1, 0].set_ylabel('Focus Score')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 5. Quality Metrics Comparison
        metrics = ['Lighting', 'Focus', 'Composition']
        avg_scores = [
            np.mean([d['lighting_score'] for d in self.image_details]),
            np.mean([d['focus_score'] for d in self.image_details]),
            np.mean([d['composition_score'] for d in self.image_details])
        ]
        axes[1, 1].bar(metrics, avg_scores, color=['#f39c12', '#2ecc71', '#9b59b6'])
        axes[1, 1].set_title('Average Quality Metrics', fontweight='bold')
        axes[1, 1].set_ylabel('Score')
        axes[1, 1].set_ylim(0, 100)
        
        # 6. Before/After Improvement
        axes[1, 2].bar(['Total Images', 'Quality Images'], 
                      [self.stats['total'], self.stats['best_shots']], 
                      color=['#e74c3c', '#2ecc71'])
        axes[1, 2].set_title('Curation Results\n(AI-Selected Best Shots)', fontweight='bold')
        axes[1, 2].set_ylabel('Image Count')
        
        plt.suptitle('SmartShot AI Photo Curation - Powered by TensorFlow & Scikit-learn', 
                    fontsize=14, fontweight='bold', y=0.995)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_folder, 'analysis_charts.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()


if __name__ == "__main__":
    PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
    INPUT = os.path.join(PROJECT_DIR, "data", "raw_images")
    OUTPUT = os.path.join(PROJECT_DIR, "output")
    if not os.path.exists(INPUT):
        os.makedirs(INPUT)
        print(f"📁 Created {INPUT}/")
        print(f"📌 Add your images to {INPUT}/ and run again")
    else:
        smartshot = SmartShot(INPUT, OUTPUT, blur_threshold=100, duplicate_threshold=5)
        results = smartshot.process_images()
        
        print("\n" + "="*70)
        print("🎉 SMARTSHOT AI PHOTO CURATION - RESULTS")
        print("="*70)
        print(f"📊 Total Processed: {results['total']}")
        print(f"✨ Sharp Images: {results['sharp']}")
        print(f"🌫️  Blurred Images: {results['blurred']}")
        print(f"🔄 Motion Blur: {results['motion_blur']}")
        print(f"🔗 Duplicates Found: {results['duplicates']}")
        print(f"🏆 Best Shots Selected: {results['best_shots']}")
        print(f"⚠️  Poor Quality: {results['poor_quality']}")
        print("="*70)
        print(f"\n📁 All results saved to: {OUTPUT}/")
        print(f"📊 View detailed analysis: {OUTPUT}/detailed_analysis.csv")
        print(f"📈 View charts: {OUTPUT}/analysis_charts.png")
        print(f"\n🤖 AI Models Used:")
        print(f"   • CNN: MobileNetV2 (ImageNet)")
        print(f"   • Clustering: DBSCAN (Scikit-learn)")
        print(f"   • Features: OpenCV + TensorFlow")
