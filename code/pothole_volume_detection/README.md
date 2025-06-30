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
     <img width="779" alt="Image" src="https://github.com/user-attachments/assets/feecf994-43b1-4c32-873b-c3a1d02faab9" />

2. **Depth Back-Projection**  
   - Use camera intrinsics (fx, fy, cx, cy) to convert depth to (X, Y, Z).  
   - Build pothole point cloud.
   - <img width="686" alt="Image" src="https://github.com/user-attachments/assets/022ad4b4-b566-4b0b-aa41-b3de2de3de3f" />

3. **Instance Point Cloud Extraction**  
   - Resize mask → threshold → extract valid 3D points.  
   - Clean masks via morphological opening and extract 2D boundaries.
     <img width="1440" alt="Image" src="https://github.com/user-attachments/assets/2f0e736c-dc90-48a0-962b-eda2b6e3434d" />

4. **Boundary-Based Plane Fitting & Alignment**  
   - Fit plane to boundary points via SVD → get normal.  
   - Rotate so normal aligns with Z-axis, then translate to origin.
     <img width="730" alt="Image" src="https://github.com/user-attachments/assets/67aee6df-c07b-4b3e-b0a1-de4d8722c07e" />
     <img width="973" alt="Image" src="https://github.com/user-attachments/assets/36c504a8-fea0-40e3-b601-21039972ce59" />

5. **Generate 3D mesh**  
   - With help of Ball-Pivoting Algorithm (BPA) generate mesh.
     <img width="778" alt="Image" src="https://github.com/user-attachments/assets/1d668f1e-f3cb-4dc0-a8fd-4b701e25919a" />
   

6. **Convex-Hull Slicing for Volume Estimation**
   - Slice cloud into horizontal slabs (dz).
   - applying shoelace algorithm to find the polygon areas
   - Compute 2D convex hull area × dz → sum volumes.
     <img width="942" alt="Image" src="https://github.com/user-attachments/assets/6e7549d8-c625-4802-b57f-dc72df086b14" />



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
