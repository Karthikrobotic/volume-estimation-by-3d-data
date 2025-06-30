import cv2
import trimesh
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
from ultralytics import YOLO

#-----path_to_folders------#
IMG_PATH      = "/Users/nkarthik/Downloads/poth/IMG16.png"
DEPTH_PATH    = "/Users/nkarthik/Downloads/poth/D16.png"
MODEL_WEIGHTS = "/Users/nkarthik/Downloads/best_(4).pt"


#-------ZED_2i_intrinsic_parameters----------#
fx, fy, cx, cy = 1350.22, 1349.80, 1104.00, 621.00


#--------------generating _3d_data_from_2d_and_depth_image--------------#
color = cv2.cvtColor(cv2.imread(IMG_PATH), cv2.COLOR_BGR2RGB)
H, W = color.shape[:2]
depth = (cv2.imread(DEPTH_PATH, cv2.IMREAD_UNCHANGED).astype(np.float32) / 1000.0)
u = np.arange(W)[None, :].repeat(H, 0)
v = np.arange(H)[:, None].repeat(W, 1)
X = (u - cx) * depth / fx
Y = (v - cy) * depth / fy
pts3d_all = np.dstack((X, Y, depth)).reshape(-1,3)
rgb_all   = (color.reshape(-1,3) / 255.0)


# ---- Run YOLO to get masks ----#
model = YOLO(MODEL_WEIGHTS)
res   = model.predict(IMG_PATH, save=False, augment=False)[0]
masks = res.masks.data.cpu().numpy()

pcs = []
image_with_masks = np.copy(color)
for inst_idx, mask in enumerate(masks):
    # A) Resize & clean mask
    m_rs    = cv2.resize(mask.astype(np.uint8), (W,H), interpolation=cv2.INTER_NEAREST)
    kernel  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    m_clean = cv2.morphologyEx(m_rs, cv2.MORPH_OPEN, kernel)


    #----For 2D segmentation image--------
    mask_color = np.random.randint(0, 255, size=3, dtype=np.uint8)
    colored_mask = np.zeros_like(color, dtype=np.uint8)
    colored_mask[m_clean == 1] = mask_color

    # Blend the colored mask with the original image
    alpha = 0.4
    image_with_masks = cv2.addWeighted(image_with_masks, 1 - alpha, colored_mask, alpha, 0)


    # B) 2D boundary
    boundary2d = cv2.morphologyEx(m_clean, cv2.MORPH_GRADIENT, kernel).astype(bool)

    # C) Back-project all mask pixels
    region_flat  = m_clean.ravel().astype(bool)
    valid_flat   = region_flat & (pts3d_all[:,2] > 0)
    pts3d_region = pts3d_all[valid_flat]
    rgb_region   = rgb_all[valid_flat]
    valid_idx    = np.nonzero(valid_flat)[0]
    boundary_idx = valid_idx[ boundary2d.ravel()[valid_idx] ]

    # Map to per-region index
    is_boundary = np.zeros(len(pts3d_region), dtype=bool)
    idx_map = {orig:i for i, orig in enumerate(valid_idx)}
    for b in boundary_idx:
        is_boundary[idx_map[b]] = True

    # D) Fit plane to boundary points
    boundary_pts = pts3d_region[is_boundary]
    # 1) Compute centroid
    centroid = boundary_pts.mean(axis=0)
    # 2) Subtract centroid
    pts_centered = boundary_pts - centroid
    # 3) SVD → normal = singular vector corresponding to smallest singular value
    _, _, vt = np.linalg.svd(pts_centered, full_matrices=False)
    normal = vt[-1, :]
    # 4) Compute d so that normal·x + d = 0 passes through centroid
    d = -normal.dot(centroid)
    a, b, c = normal
    print(f"Instance {inst_idx} boundary-plane: {a:.4f}x + {b:.4f}y + {c:.4f}z + {d:.4f} = 0")

    # E) Build and color the per-instance cloud
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts3d_region))
    colors = np.copy(rgb_region)
    colors[is_boundary] = [1.0, 0.0, 0.0]
    pcd.colors = o3d.utility.Vector3dVector(colors)
    pcs.append(pcd)


# --------- Display the 2D image with mask overlays ---------
plt.figure(figsize=(10, 8))
plt.imshow(image_with_masks)
plt.title("2D Image with Mask Segmentation")
plt.axis('off')
plt.show()

# ---- 3) Visualize each instance’s full cloud boundaries ----
o3d.visualization.draw_geometries(pcs)

plane_normal = np.array([a, b, c])
plane_normal /= np.linalg.norm(plane_normal)
target_normal = np.array([0.0, 0.0, 1.0])  

# Compute rotation axis and angle
rotation_axis = np.cross(plane_normal, target_normal)
rotation_axis /= np.linalg.norm(rotation_axis)
angle = np.arccos(np.dot(plane_normal, target_normal))

# Create rotation matrix
R = o3d.geometry.get_rotation_matrix_from_axis_angle(rotation_axis * angle)

#  Rotate the entire cloud
pcd.rotate(R)

# Translate to origin
points = np.asarray(pcd.points)
min_bounds = points.min(axis=0)
pcd.translate(-min_bounds)
downpcd=pcd

# Plot
points = np.asarray(downpcd.points)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=1, c='blue') 
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
plt.title('3D Point Cloud ofter alining')
plt.show()

pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=30))
pcd.orient_normals_consistent_tangent_plane(k=30)
distances = pcd.compute_nearest_neighbor_distance()
avg_dist = np.mean(distances)
radius = 3 * avg_dist

# Create BPA (Ball Pivoting Algorithm) mesh
bpa_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
    pcd, o3d.utility.DoubleVector([radius, radius * 2])
)
o3d.visualization.draw_geometries([bpa_mesh], mesh_show_back_face=True, window_name='BPA_mesh')
bpa_mesh = bpa_mesh.filter_smooth_laplacian(number_of_iterations=6)


# ----- pothole geometry-------
aabb = downpcd.get_axis_aligned_bounding_box()
min_bound = aabb.get_min_bound()
max_bound = aabb.get_max_bound()
width, height, depth = max_bound - min_bound

# Print box dimensions
print(f"Width: {width:.2f}")
print(f"Height: {height:.2f}")
print(f"Depth: {depth:.2f}")
aabb.color = (1, 0, 0) 

# Visualize point cloud with bounding box and bpamesh
o3d.visualization.draw_geometries([downpcd, aabb], window_name="Point Cloud with Bounding Box")
o3d.visualization.draw_geometries([bpa_mesh])




def calculate_area_from_random_points(points):
    """
    Calculates the area of a polygon from a set of random points
    by ordering them based on the angle around their centroid.

    Args:q
        points: A list of tuples or a NumPy array of shape (n, 2),
                where n is the number of vertices.

    Returns:
        The area of the polygon.
    """
    n = len(points)
    if n < 3:
        return 0.0  # Area is zero for less than 3 points

    # Separate x and y coordinates
    x_coords = points[:, 0]
    y_coords = points[:, 1]

    # Create the scatter plot
    plt.figure(figsize=(8, 6))  # Optional: Adjust the size of the plot
    plt.scatter(x_coords, y_coords)
    plt.xlabel("X-axis")
    plt.ylabel("Y-axis")
    plt.title("Visualization of 2D Points")
    plt.grid(True)
    plt.show()
  
    # 1. Calculate the centroid
    centroid_x = np.mean(points[:, 0])
    centroid_y = np.mean(points[:, 1])
    
    # 2. Calculate the angle of each point relative to the centroid
    angles = np.arctan2(points[:, 1] - centroid_y, points[:, 0] - centroid_x)

    # 3. Sort the points based on their angles
    sorted_indices = np.argsort(angles)
    ordered_points = points[sorted_indices]
    area = 0.0
    for i in range(n):
        x1, y1 = ordered_points[i]
        x2, y2 = ordered_points[(i + 1) % n]  
        area += (x1 * y2 - x2 * y1)

    return (abs(area) / 2.0)*0.005

def trimesh_to_open3d(trimesh_mesh):
    """
    Convert a Trimesh object to an Open3D TriangleMesh
    """
    vertices = np.asarray(trimesh_mesh.vertices)
    faces = np.asarray(trimesh_mesh.faces)
    o3d_mesh = o3d.geometry.TriangleMesh()
    o3d_mesh.vertices = o3d.utility.Vector3dVector(vertices)
    o3d_mesh.triangles = o3d.utility.Vector3iVector(faces)
    o3d_mesh.compute_vertex_normals()
    return o3d_mesh

def path3D_to_lineset(path3D):
    """
    Convert a Trimesh Path3D (cross-section) to Open3D LineSet
    """
    if path3D is None:
        return None
    
    points = np.asarray(path3D.vertices)
    lines = []

    for entity in path3D.entities:
        if hasattr(entity, 'points'):
            indices = entity.points
            for i in range(len(indices) - 1):
                lines.append([indices[i], indices[i+1]])
    
    if len(points) == 0 or len(lines) == 0:
        return None

    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(points)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    return line_set


def get_section_bounds(mesh, z_min, z_max):
    z_center = (z_min + z_max) / 2.0
    plane_origin = [0, 0, z_center]
    plane_normal = [0, 0, 1]

    # Get the cross-section contour from the mesh
    bpa_vertices = np.asarray(bpa_mesh.vertices)
    bpa_faces = np.asarray(bpa_mesh.triangles)
    bpa_trimesh = trimesh.Trimesh(vertices=bpa_vertices, faces=bpa_faces, process=False)
    mesh = bpa_trimesh  
    section = mesh.section(plane_origin=plane_origin, plane_normal=plane_normal)

    ''' ply to mesh'''
    o3d_mesh = trimesh_to_open3d(mesh)    
    o3d_section = path3D_to_lineset(section) 
    ''' ply to mesh'''

    if o3d_section:
       o3d.visualization.draw_geometries([o3d_mesh, o3d_section])

    if section is None:
        return None

    section_2D = np.asarray(section.vertices)

    if section_2D is None  or section == 0:
        return None

    vertices = section_2D
    area = calculate_area_from_random_points(vertices[:,:2])
    x_coords = vertices[:, 0]
    y_coords = vertices[:, 1]
    min_x, max_x = np.min(x_coords), np.max(x_coords)
    min_y, max_y = np.min(y_coords), np.max(y_coords)

    return (min_x, max_x, min_y, max_y), area

def auto_section_bounds(mesh, thickness):
    """
    Automatically divides the mesh's z-range into slices of given thickness,
    computes the bounds and area for each slice, and returns a list of the results.
    """
    ''' ply to mesh'''
    bbox = mesh.get_axis_aligned_bounding_box()
    overall_bounds = np.array([bbox.get_min_bound(), bbox.get_max_bound()])
    ''' ply to mesh'''
    overall_z_min = overall_bounds[0][2]
    overall_z_max = overall_bounds[1][2]

    results = []
    z_current = overall_z_min
    while z_current + thickness <= overall_z_max:
        print(z_current, z_current + thickness)
        bounds_area = get_section_bounds(mesh, z_current, z_current + thickness)
        if bounds_area is not None:
            bounds, area = bounds_area
            results.append({
                'z_range': (z_current, z_current + thickness),
                'bounds': bounds,
                'area': area
            })
        else:
            results.append({
                'z_range': (z_current, z_current + thickness),
                'bounds': None,
                'area': None
            })
        z_current += thickness
    return results

# Example: Automatically process the mesh in slices with a z-thickness of 0.2 units
sections = auto_section_bounds(bpa_mesh, thickness=0.005)
sections=sections[2:]
t=0
for sec in sections:
    print(f"Z-range: {sec['z_range']} - Bounds (x, y): {sec['bounds']} - Area: {sec['area']}")
    t+=sec['area']
print(t,t*1000)
