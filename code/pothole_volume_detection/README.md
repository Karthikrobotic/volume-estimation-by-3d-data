# 3D Pothole Boundary-Based Alignment and Volume Estimation

## Description
This project processes RGB-D images of potholes to:
- Segment the pothole region (using YOLOv8),  
- Back-project pixels into 3D,  
- Align the extracted point cloud based solely on pothole boundaries,  
- Estimate the pothole volume using convex-hull slicing.  


## Key Algorithms and Techniques

1. **Semantic Segmentation (YOLOv8)**  
   - Ultralytics YOLOv8 fine-tuned on a pothole dataset.  
   - Clean masks via morphological opening and extract 2D boundaries.

2. **Depth Back-Projection**  
   - Use camera intrinsics (fx, fy, cx, cy) to convert depth to (X, Y, Z).  
   - Build full-scene point cloud.

3. **Instance Point Cloud Extraction**  
   - Resize mask → threshold → extract valid 3D points.  
   - Separate boundary vs. interior points.

4. **Boundary-Based Plane Fitting & Alignment**  
   - Fit plane to boundary points via SVD → get normal.  
   - Rotate so normal aligns with Z-axis, then translate to origin.

5. **Convex-Hull Slicing for Volume Estimation**  
   - Slice cloud into horizontal slabs (dz).  
   - Compute 2D convex hull area × dz → sum volumes.

6. **Cross-Sectional Analysis (Optional)**  
   - Build BPA mesh with Trimesh & Open3D.  
   - Extract contour cross-sections and compute section areas.

## Step-by-Step Workflow

1. **Data Prep**  
   - Put `IMG16.png`, `D16.png` and your YOLO weights (`best_(4).pt`) into `assets/`.

2. **Configure**  
   - Edit `ai.py` → set `IMG_PATH`, `DEPTH_PATH`, `MODEL_WEIGHTS`, and `(fx, fy, cx, cy)`.

3. **Segmentation**  
   - Run YOLOv8 → get raw masks → clean + extract boundaries.

4. **Back-Project**  
   - Convert each mask pixel → 3D point using depth & intrinsics.

5. **Extract & Align**  
   - Filter by mask → get boundary pts → fit plane → rotate + translate.

6. **Volume Calculation**  
   - Slice into slabs → convex hull per slab → sum slab volumes.

7. **Visualize & Validate**  
   - Show 2D overlays, 3D cloud + mesh, print plane equations, bounds, and volume.

8. **Optional Sections**  
   - Use Trimesh to extract cross-section contours and compute their areas.

## Installation

```bash
git clone https://github.com/your-username/your-repo.git
cd your-repo
python3 -m venv yolov8_env
source yolov8_env/bin/activate
pip install -r requirements.txt
