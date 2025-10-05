# --------------------------
# Utility functions
# --------------------------
def encode_image(path: Path) -> str:
    """Convert image file to base64 string"""
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")
    

def point_cloud_to_depthmap(pcd_path: Path, save_path: Path, resolution=2048, use_intensity=False):
    """
    Convert raw point cloud (.pcd/.ply) into a bird-eye depth map image.
    
    Args:
        pcd_path (Path): Input point cloud file (.pcd or .ply).
        save_path (Path): Output image path (.png).
        resolution (int): Histogram bin resolution (default=2048).
        use_intensity (bool): Whether to weight by intensity instead of z.
    
    Returns:
        Path: Path to saved depthmap image.
    """
    # --------- Load with pypcd4 ---------
    pcd_raw = PointCloud.from_path(str(pcd_path))
    arr = pcd_raw.numpy()   # shape (N, F), columns: x,y,z,intensity,...
    
    if arr.shape[1] < 3:
        raise ValueError(f"{pcd_path} does not contain x,y,z columns.")

    x, y, z = arr[:, 0], arr[:, 1], arr[:, 2]
    if use_intensity and "intensity" in pcd_raw.fields:
        weights = arr[:, pcd_raw.fields.index("intensity")]
    else:
        weights = z

    # --------- Visualization with Open3D (optional) ---------
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(arr[:, :3])
    pcd.paint_uniform_color([0, 0, 1])

    # --------- Plot bird-eye depth map ---------
    plt.figure(figsize=(6, 6))
    plt.hist2d(x, y, bins=resolution, weights=weights, cmap="viridis")
    plt.colorbar(label="Height (z)" if not use_intensity else "Intensity")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("Bird-eye Depth Map")
    plt.axis("equal")
    plt.savefig(save_path, dpi=200)
    plt.close()

    return save_path