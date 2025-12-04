# SmartShot: AI Photo Curation Assistant
## Overview

SmartShot is an artificial intelligence system that automatically organizes photo collections. The system detects blurred images, identifies duplicates, assesses image quality across multiple dimensions, and selects the best shots from large collections. This addresses the time-consuming manual workflow photographers face when sorting hundreds of images.

---

## Installation

Install all required dependencies using pip:

```bash
pip install -r requirements.txt
```

The system requires Python version 3.8 or higher. Key dependencies include TensorFlow for deep learning, OpenCV for image processing, Scikit-learn for machine learning, and Streamlit for the web interface.

---

## Usage

### Web Interface

Launch the interactive web interface:

```bash
streamlit run streamlit_app.py
```

This opens a browser application at `http://localhost:8501` where you can upload images, adjust blur and duplicate thresholds, view real-time processing progress with a stop button, and download organized results as ZIP files.

### Command Line

For batch processing:

```bash
python smartshot.py
```

Place images in the `data/raw_images` folder. The system creates this folder automatically if it does not exist. Results are saved to the `output` folder in the project directory.

---

## Methodology

The system employs three complementary artificial intelligence techniques to analyze and organize photographs.

### Convolutional Neural Network for Blur Detection

The blur detection module uses MobileNetV2, a pretrained convolutional neural network originally trained on the ImageNet dataset. Transfer learning allows the system to leverage deep features learned from millions of images. The network combines these learned features with traditional computer vision metrics including Laplacian variance for edge detection, directional gradients for motion analysis, and edge density calculations. This hybrid approach classifies images as sharp, blurred, or motion blurred.

### DBSCAN Clustering for Duplicate Detection

Duplicate detection begins by calculating perceptual hashes for each image. These compact fingerprints represent visual content regardless of minor variations in size or compression. The system then applies DBSCAN clustering from scikit-learn to group similar images based on hash distances. Unlike methods requiring predefined cluster counts, DBSCAN automatically discovers duplicate groups based on density.

### Multi-Dimensional Quality Assessment

Quality assessment analyzes three dimensions. Lighting quality evaluates brightness distribution, exposure levels, and ISO settings from EXIF metadata. Focus quality combines Laplacian variance, Tenengrad focus measure, and normalized graylevel variance to assess sharpness. Composition quality examines adherence to the rule of thirds, contrast levels, and color harmony. Each dimension receives a score from 0 to 100, weighted and combined into an overall quality metric.

The final quality score weights focus at 40%, lighting at 30%, and composition at 30%. This weighting reflects the primary importance of sharpness in photograph quality.

---

## Output Structure

The system organizes images into six folders and generates three types of reports.

**Image Folders:**
- `best_shots` contains curated high-quality images with overall quality score above 50
- `sharp` includes all sharp images classified by the CNN
- `blurred` contains defocused images classified by the CNN
- `motion_blur` holds images with directional blur from movement
- `duplicates` stores similar images found by DBSCAN clustering
- `poor_quality` contains sharp images with overall quality below 40

**Report Files:**
- `detailed_analysis.csv` provides per-image metrics and scores
- `report.json` contains processing statistics and summary data
- `analysis_charts.png` displays six visualization charts

---

## Interpreting Results

### Detailed Analysis CSV

The CSV file contains one row per image with the following key columns:

The `overall_quality` column provides a composite score from 0 to 100. Images scoring above 70 represent excellent quality. Images below 40 indicate poor quality suitable for deletion. The score combines lighting, focus, and composition metrics.

The `blur_type` column classifies each image as sharp, blurred, or motion_blur based on CNN analysis.

Individual quality dimensions appear in `lighting_score`, `focus_score`, and `composition_score` columns. Each ranges from 0 to 100 and can identify specific quality issues. For example, an image might have excellent composition but poor lighting.

Camera metadata appears in `iso` and `exposure` columns when available from EXIF data.

To find the best images, sort the CSV by `overall_quality` in descending order. To identify problematic images, filter for `blur_type` not equal to sharp or `overall_quality` below 40.

### Processing Statistics JSON

The JSON report provides aggregate statistics. The `statistics` section shows total images processed and counts for each category. The `duplicate_groups` section lists which images belong together. The `summary` section calculates improvement metrics including the percentage reduction from total images to curated best shots.

### Visualization Charts

The PNG file contains six charts. The pie chart shows the distribution across sharp, blurred, and motion blur categories. The bar chart compares counts across all categories. The histogram displays the distribution of overall quality scores. The scatter plot reveals correlation between lighting and focus quality. The metrics comparison shows average scores for each quality dimension. The before-after chart illustrates the curation impact by comparing total input images to final best shots.

---

## Performance Metrics

Testing with 60 diverse photographs from personal collections demonstrated the following results:

From the dataset of 60 images, all 60 were classified as sharp with 100% detection. No images were classified as blurred or motion blur. No duplicate groups were detected in this particular test set. The system processed images at approximately 5 seconds per image on standard hardware. 

Quality assessment scores showed the following averages: lighting quality scored 55.0 out of 100, focus quality scored 23.1 out of 100, and composition quality scored 100.0 out of 100. The system successfully organized all 60 images into the sharp category and identified 53 images as best shots based on quality thresholds above 50.

Processing time scales linearly with image count. The system completed processing of 60 images in approximately 5 minutes.

---

## Troubleshooting

**All images classified as blurred:** The default blur threshold may be too strict. In the Streamlit interface, adjust the slider in the sidebar. For command line, edit line 353 in smartshot.py to `blur_threshold=50` instead of 100.

**Excessive duplicate detection:** The duplicate threshold may be too permissive. Decrease the value to `duplicate_threshold=3` for stricter matching. Lower values require closer similarity for duplicate classification.

**Module import errors:** Reinstall all dependencies using `pip install -r requirements.txt` in the project directory. Verify Python version meets the 3.8 minimum requirement.

**No images found error:** Confirm images exist in the `data/raw_images` folder. For Streamlit, upload images through the web interface. For command line, the system creates the folder automatically if missing. Supported formats are JPG, JPEG, and PNG.

**Slow processing speed:** Processing time of 5 seconds per image is normal. Reduce the test image count during development. GPU acceleration can improve speed but is not required.

---

## Technical Architecture

The system architecture consists of four primary components implemented in Python.

The CNN blur detector loads MobileNetV2 with pretrained ImageNet weights. The base model layers remain frozen during inference. A custom classification head maps extracted features to three output classes. Traditional computer vision metrics supplement deep learning features for robust classification.

The quality analyzer extracts EXIF metadata using Pillow and calculates multiple focus metrics including Laplacian variance and Tenengrad measure. Lighting analysis converts images to HSV color space for brightness evaluation. Composition scoring applies rule of thirds principles and measures visual balance.

The duplicate detector uses the imagehash library to compute perceptual hashes. StandardScaler normalizes hash vectors before clustering. DBSCAN groups similar images with epsilon parameter 0.5 and minimum samples parameter 2. The algorithm handles variable cluster sizes without requiring predefined counts.

The main processing pipeline coordinates these components sequentially. Images first undergo blur classification. Sharp images proceed to duplicate detection and quality scoring. The system selects best shots from duplicate groups based on combined quality metrics. Finally, all images are copied to appropriate output folders and reports are generated.

---

## System Requirements

**Operating System:** Windows 10 or later, macOS 10.14 or later, Ubuntu 18.04 or later  
**Python Version:** 3.8, 3.9, 3.10, or 3.11  
**Memory:** 4GB minimum, 8GB recommended for large collections  
**Storage:** 2-5GB for images, models, and outputs  
**Processor:** Any modern CPU; GPU optional for faster TensorFlow operations

---

## Project Structure

```
smartshot/
├── smartshot.py              # Core processing engine
├── streamlit_app.py          # Web application interface  
├── requirements.txt          # Python dependencies
├── README.md                 # Documentation
├── data/
│   └── raw_images/          # Input image folder
└── output/                   # Results folder
    ├── sharp/
    ├── blurred/
    ├── motion_blur/
    ├── duplicates/
    ├── best_shots/
    ├── poor_quality/
    ├── detailed_analysis.csv
    ├── report.json
    └── analysis_charts.png
```

---

## Algorithm Parameters

Two parameters control processing behavior. In Streamlit, adjust using sidebar sliders. For command line, modify smartshot.py line 353:

The `blur_threshold` parameter sets minimum Laplacian variance for sharp classification. Default is 100, range 50 to 200. Higher values increase strictness.

The `duplicate_threshold` parameter sets maximum hash distance for duplicate grouping. Default is 5, range 1 to 10. Lower values require closer similarity.

---
