
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import warnings
warnings.filterwarnings('ignore')

import streamlit as st
import os
import cv2
import numpy as np
from PIL import Image
import tempfile
import shutil
import zipfile
from io import BytesIO
import pandas as pd
from smartshot import SmartShot

st.set_page_config(page_title="SmartShot AI", page_icon="📸", layout="wide")

# Custom CSS
st.markdown("""
<style>
.big-font {font-size:20px !important; font-weight: bold;}
.metric-card {background-color: #f0f2f6; padding: 20px; border-radius: 10px; margin: 10px;}
</style>
""", unsafe_allow_html=True)

st.title("📸 SmartShot: AI Photo Curation Assistant")
st.markdown("**By Dania Alazzawi & Melanie Vaknin**")
st.markdown("---")

# Initialize session state
if 'processing_complete' not in st.session_state:
    st.session_state['processing_complete'] = False
if 'stop_processing' not in st.session_state:
    st.session_state['stop_processing'] = False

# Sidebar configuration
with st.sidebar:
    st.header("⚙️ Configuration")
    blur_thresh = st.slider("Blur Detection Threshold", 50, 200, 100, 
                            help="Higher = stricter blur detection")
    dup_thresh = st.slider("Duplicate Detection Threshold", 1, 10, 5,
                          help="Lower = stricter duplicate detection")
    
    st.markdown("---")
    st.markdown("### 🤖 AI Features")
    st.info("""
    ✅ CNN Blur & Motion Detection  
    ✅ EXIF Metadata Analysis  
    ✅ Lighting Quality Assessment  
    ✅ Focus Quality Analysis  
    ✅ Composition Scoring  
    ✅ ML-based Clustering  
    """)


def create_zip_file(folder_path, zip_name):
    """Create ZIP file from folder"""
    memory_file = BytesIO()
    with zipfile.ZipFile(memory_file, 'w', zipfile.ZIP_DEFLATED) as zf:
        for root, dirs, files in os.walk(folder_path):
            for file in files:
                file_path = os.path.join(root, file)
                arcname = os.path.relpath(file_path, folder_path)
                zf.write(file_path, arcname)
    memory_file.seek(0)
    return memory_file


def create_full_archive(output_folder):
    """Create complete archive with all sorted images"""
    memory_file = BytesIO()
    with zipfile.ZipFile(memory_file, 'w', zipfile.ZIP_DEFLATED) as zf:
        for root, dirs, files in os.walk(output_folder):
            for file in files:
                file_path = os.path.join(root, file)
                arcname = os.path.relpath(file_path, output_folder)
                zf.write(file_path, arcname)
    memory_file.seek(0)
    return memory_file


def show_before_after_comparison(input_folder, output_folder):
    """Show before/after folder comparison"""
    st.markdown("### 📁 Before/After Folder Comparison")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📂 BEFORE - Unsorted Folder")
        input_files = [f for f in os.listdir(input_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        st.metric("Total Unsorted Images", len(input_files))
        
        # Show sample images from input
        if len(input_files) > 0:
            sample_count = min(6, len(input_files))
            st.write(f"Sample of {sample_count} images:")
            cols = st.columns(3)
            for idx in range(sample_count):
                with cols[idx % 3]:
                    img_path = os.path.join(input_folder, input_files[idx])
                    img = Image.open(img_path)
                    st.image(img, caption=input_files[idx], width="stretch")
    
    with col2:
        st.subheader("📂 AFTER - Organized Folders")
        categories = ['best_shots', 'sharp', 'blurred', 'motion_blur', 'duplicates', 'poor_quality']
        
        organized_count = 0
        for cat in categories:
            cat_path = os.path.join(output_folder, cat)
            if os.path.exists(cat_path):
                count = len([f for f in os.listdir(cat_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
                organized_count += count
                if count > 0:
                    st.write(f"✅ **{cat}**: {count} images")
        
        st.metric("Total Organized Images", organized_count)
        
        # Show best shots
        best_shots_path = os.path.join(output_folder, 'best_shots')
        if os.path.exists(best_shots_path):
            best_files = [f for f in os.listdir(best_shots_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            if len(best_files) > 0:
                sample_count = min(6, len(best_files))
                st.write(f"🏆 Best Shots ({sample_count}):")
                cols = st.columns(3)
                for idx in range(sample_count):
                    with cols[idx % 3]:
                        img_path = os.path.join(best_shots_path, best_files[idx])
                        img = Image.open(img_path)
                        st.image(img, caption=best_files[idx], width="stretch")


# Main tabs
tab1, tab2, tab3 = st.tabs(["📤 Upload & Process", "📊 Analytics Dashboard", "🔬 Single Image Test"])

# ============== TAB 1: Upload & Process ==============
with tab1:
    st.header("Upload Your Photo Collection")
    
    uploaded_files = st.file_uploader(
        "Choose images (JPG, PNG supported)", 
        type=['jpg', 'jpeg', 'png'], 
        accept_multiple_files=True,
        help="Upload multiple images for AI-powered curation",
        key="file_uploader"
    )
    
    if uploaded_files:
        st.success(f"✅ {len(uploaded_files)} images uploaded")
        
        # Show buttons based on processing state
        if not st.session_state.get('processing_complete', False):
            if st.button("🚀 Start AI Processing", type="primary", width="stretch", key="start_btn"):
                st.session_state['start_processing'] = True
                st.rerun()
        
        # Process images if start was clicked
        if st.session_state.get('start_processing', False) and not st.session_state.get('processing_complete', False):
            
            # Create containers for stop button and processing
            stop_container = st.container()
            processing_container = st.container()
            
            with stop_container:
                col1, col2, col3 = st.columns([1, 1, 1])
                with col2:
                    if st.button("⏹️ Stop Processing", width="stretch", type="secondary", key="stop_btn"):
                        st.session_state['stop_processing'] = True
                        st.session_state['start_processing'] = False
                        st.warning("⚠️ Processing will stop...")
                        st.rerun()
            
            # Check if stop was clicked
            if st.session_state.get('stop_processing', False):
                st.error("❌ Processing stopped by user")
                st.session_state['start_processing'] = False
                st.session_state['stop_processing'] = False
                st.stop()
            
            with processing_container:
                PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
                temp_input = os.path.join(PROJECT_DIR, "temp_uploads")
                temp_output = os.path.join(PROJECT_DIR, "temp_output")

                # Create directories if they don't exist
                os.makedirs(temp_input, exist_ok=True)
                os.makedirs(temp_output, exist_ok=True)

                # Clean previous files
                for folder in [temp_input, temp_output]:
                    for file in os.listdir(folder):
                        file_path = os.path.join(folder, file)
                        try:
                            if os.path.isfile(file_path):
                                os.unlink(file_path)
                            elif os.path.isdir(file_path):
                                shutil.rmtree(file_path)
                        except Exception as e:
                            pass
                
                # Save uploaded files
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                for idx, file in enumerate(uploaded_files):
                    with open(os.path.join(temp_input, file.name), 'wb') as f:
                        f.write(file.getbuffer())
                    progress_bar.progress((idx + 1) / len(uploaded_files))
                    status_text.text(f"Uploading {idx + 1}/{len(uploaded_files)} images...")
                
                status_text.text("Starting AI analysis...")
                
                # Process images
                with st.spinner("🤖 AI is analyzing your images..."):
                    smartshot = SmartShot(temp_input, temp_output, blur_thresh, dup_thresh)
                    results = smartshot.process_images()
                
                st.balloons()
                st.success("✅ Processing Complete!")
                
                # Store results in session state
                st.session_state['results'] = results
                st.session_state['output_folder'] = temp_output
                st.session_state['input_folder'] = temp_input
                st.session_state['image_details'] = smartshot.image_details
                st.session_state['processing_complete'] = True
                st.session_state['start_processing'] = False
                st.rerun()
    
    # Show results if processing is complete
    if st.session_state.get('processing_complete', False):
        results = st.session_state['results']
        output_folder = st.session_state['output_folder']
        input_folder = st.session_state['input_folder']
        
        # Display metrics
        st.markdown("---")
        st.markdown("### 📊 Processing Summary")
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.metric("📷 Total Images", results['total'])
        with col2:
            st.metric("✨ Sharp", results['sharp'], 
                     delta=f"{results['sharp']/results['total']*100:.1f}%")
        with col3:
            st.metric("🌫️ Blurred", results['blurred'])
        with col4:
            st.metric("🔄 Motion Blur", results.get('motion_blur', 0))
        with col5:
            st.metric("🏆 Best Shots", results['best_shots'])
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("🔗 Duplicates", results['duplicates'])
        with col2:
            st.metric("⚠️ Poor Quality", results.get('poor_quality', 0))
        with col3:
            improvement = (results['best_shots'] / results['total'] * 100) if results['total'] > 0 else 0
            st.metric("📈 Quality Ratio", f"{improvement:.1f}%")
        
        st.markdown("---")
        
        # Before/After Comparison
        show_before_after_comparison(input_folder, output_folder)
        
        st.markdown("---")
        
        # Analysis charts
        st.markdown("### 📈 Visual Analysis Charts")
        chart_path = os.path.join(output_folder, 'analysis_charts.png')
        if os.path.exists(chart_path):
            st.image(chart_path, width="stretch")
            
            with open(chart_path, 'rb') as f:
                chart_data = f.read()
            st.download_button(
                label="📊 Download Analysis Charts",
                data=chart_data,
                file_name="smartshot_analysis.png",
                mime="image/png",
                key="download_charts"
            )
        
        st.markdown("---")
        
        # Download options
        st.markdown("### 📥 Download Results")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 📦 Complete Archive")
            zip_file = create_full_archive(output_folder)
            st.download_button(
                label="💾 Download All Sorted Images (ZIP)",
                data=zip_file,
                file_name="smartshot_complete_archive.zip",
                mime="application/zip",
                width="stretch",
                key="download_complete"
            )
        
        with col2:
            st.markdown("#### 📁 Individual Folders")
            
            categories = {
                '🏆 Best Shots': 'best_shots',
                '✨ Sharp Images': 'sharp',
                '🌫️ Blurred Images': 'blurred',
                '🔄 Motion Blur': 'motion_blur',
                '🔗 Duplicates': 'duplicates',
                '⚠️ Poor Quality': 'poor_quality'
            }
            
            for label, folder in categories.items():
                folder_path = os.path.join(output_folder, folder)
                if os.path.exists(folder_path) and os.listdir(folder_path):
                    count = len(os.listdir(folder_path))
                    zip_data = create_zip_file(folder_path, f"{folder}.zip")
                    st.download_button(
                        label=f"{label} ({count} images)",
                        data=zip_data,
                        file_name=f"smartshot_{folder}.zip",
                        mime="application/zip",
                        key=f"download_{folder}",
                        width="stretch"
                    )
        
        st.markdown("---")
        
        # Detailed CSV
        st.markdown("### 📄 Detailed Analysis Report")
        csv_path = os.path.join(output_folder, 'detailed_analysis.csv')
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            st.dataframe(df, width="stretch", height=400)
            
            csv_data = df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download Detailed CSV Report",
                data=csv_data,
                file_name="smartshot_detailed_analysis.csv",
                mime="text/csv",
                key="download_csv"
            )
        
        # Reset button
        st.markdown("---")
        if st.button("🔄 Process New Images", type="secondary", width="stretch"):
            st.session_state['processing_complete'] = False
            st.session_state['start_processing'] = False
            st.session_state['stop_processing'] = False
            st.rerun()

# ============== TAB 2: Analytics Dashboard ==============
with tab2:
    st.header("📊 Analytics Dashboard")
    
    if 'image_details' in st.session_state:
        df = pd.DataFrame(st.session_state['image_details'])
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 🎯 Quality Distribution")
            quality_ranges = pd.cut(df['overall_quality'], bins=[0, 40, 70, 100], 
                                   labels=['Poor', 'Good', 'Excellent'])
            quality_counts = quality_ranges.value_counts()
            st.bar_chart(quality_counts)
        
        with col2:
            st.markdown("#### 📸 Image Classification")
            category_counts = df['blur_type'].value_counts()
            st.bar_chart(category_counts)
        
        st.markdown("---")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("#### 💡 Lighting Quality")
            avg_lighting = df['lighting_score'].mean()
            st.metric("Average Score", f"{avg_lighting:.1f}/100")
            st.bar_chart(df['lighting_score'])
        
        with col2:
            st.markdown("#### 🎯 Focus Quality")
            avg_focus = df['focus_score'].mean()
            st.metric("Average Score", f"{avg_focus:.1f}/100")
            st.bar_chart(df['focus_score'])
        
        with col3:
            st.markdown("#### 🎨 Composition Quality")
            avg_composition = df['composition_score'].mean()
            st.metric("Average Score", f"{avg_composition:.1f}/100")
            st.bar_chart(df['composition_score'])
        
        st.markdown("---")
        
        # Top performers
        st.markdown("#### 🏆 Top Quality Images")
        top_images = df.nlargest(10, 'overall_quality')[['filename', 'overall_quality', 
                                                         'lighting_score', 'focus_score', 
                                                         'composition_score']]
        st.dataframe(top_images, width="stretch")
        
    else:
        st.info("👆 Upload and process images in the 'Upload & Process' tab to view analytics")

# ============== TAB 3: Single Image Test ==============
with tab3:
    st.header("🔬 Single Image Quality Test")
    st.markdown("Analyze individual image quality with AI")
    
    uploaded = st.file_uploader("Upload image for analysis", type=['jpg', 'jpeg', 'png'], key="single_test")
    
    if uploaded:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📷 Original Image")
            img = Image.open(uploaded)
            st.image(img, width="stretch")
        
        with col2:
            st.subheader("🤖 AI Analysis Results")
            
            # Save temp file
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.jpg')
            img.save(temp_file.name)
            
            # Analyze
            image_cv = cv2.imread(temp_file.name)
            gray = cv2.cvtColor(image_cv, cv2.COLOR_BGR2GRAY)
            
            # Sharpness
            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            st.metric("🎯 Sharpness Score", f"{laplacian_var:.2f}")
            
            # Classification
            if laplacian_var > blur_thresh:
                st.success("✅ Sharp Image")
            else:
                st.error("❌ Blurred Image")
            
            # Additional metrics
            hsv = cv2.cvtColor(image_cv, cv2.COLOR_BGR2HSV)
            brightness = np.mean(hsv[:, :, 2])
            st.metric("💡 Brightness", f"{brightness:.1f}/255")
            
            # Contrast
            contrast = gray.std()
            st.metric("🎨 Contrast", f"{contrast:.1f}")
            
            # Edge density
            edges = cv2.Canny(gray, 100, 200)
            edge_density = np.sum(edges > 0) / edges.size * 100
            st.metric("📐 Edge Density", f"{edge_density:.2f}%")
                        
            st.markdown("---")
            st.info("""
            **Analysis includes:**
            - Sharpness detection using Laplacian variance
            - Brightness and contrast measurement
            - Edge detection for composition analysis
            """)
            
            os.unlink(temp_file.name)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p><strong>SmartShot AI Photo Curation Assistant</strong></p>
    <p>Powered by TensorFlow CNN • OpenCV • Scikit-learn • EXIF Analysis</p>
    <p>© 2025 Dania Alazzawi & Melanie Vaknin</p>
</div>
""", unsafe_allow_html=True)
