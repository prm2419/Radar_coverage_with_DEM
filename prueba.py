import numpy as np
import rasterio
from rasterio.enums import Resampling
import matplotlib.pyplot as plt

# ============================================================
# === 1. LECTOR DE DEM (antes dem_io.py)
# ============================================================
def load_dem(path_dem, target_resolution=None, resampling=Resampling.bilinear):
    """
    Carga un archivo DEM GeoTIFF y devuelve la matriz de elevación, 
    coordenadas X/Y y metadatos corregidos.
    """
    with rasterio.open(path_dem) as src:
        if target_resolution:
            scale_x = src.res[0] / target_resolution
            scale_y = src.res[1] / target_resolution
            new_width = int(src.width / scale_x)
            new_height = int(src.height / scale_y)

            dem_data = src.read(
                1,
                out_shape=(new_height, new_width),
                resampling=resampling
            )
            transform = src.transform * src.transform.scale(
                src.width / dem_data.shape[-1],
                src.height / dem_data.shape[-2]
            )
        else:
            dem_data = src.read(1)
            transform = src.transform

        dem_data = dem_data.astype(np.float32)
        dem_data[dem_data == src.nodata] = np.nan

        H, W = dem_data.shape
        x_coords = np.arange(W) * transform.a + transform.c + transform.a / 2
        y_coords = np.arange(H) * transform.e + transform.f + transform.e / 2
        if y_coords[1] < y_coords[0]:
            y_coords = y_coords[::-1]
            dem_data = np.flipud(dem_data)

        meta = src.meta.copy()
        meta.update({
            "transform": transform,
            "width": dem_data.shape[1],
            "height": dem_data.shape[0],
        })
    return dem_data, x_coords, y_coords, meta


# ============================================================
# === 2. CREACIÓN DE LA GRILLA TENSORIAL (vectorizada)
# ============================================================
def build_tensor_grid(dem, x_coords, y_coords, dz=50, z_padding=500):
    """
    Construye una grilla 3D booleana del terreno a partir del DEM.
    Cada voxel True representa terreno sólido.
    """
    H, W = dem.shape
    dx = np.abs(x_coords[1] - x_coords[0])
    dy = np.abs(y_coords[1] - y_coords[0])

    z_min = np.nanmin(dem)
    z_max = np.nanmax(dem) + z_padding
    z_levels = np.arange(z_min, z_max + dz, dz)
    Nz = len(z_levels)

    dem_filled = np.nan_to_num(dem, nan=z_min)
    grid = (z_levels[:, None, None] <= dem_filled[None, :, :])

    meta = {
        "dx": dx,
        "dy": dy,
        "dz": dz,
        "Nx": W,
        "Ny": H,
        "Nz": Nz,
        "x_extent": (x_coords[0], x_coords[-1]),
        "y_extent": (y_coords[0], y_coords[-1]),
        "z_extent": (z_min, z_max),
    }

    return grid, z_levels, meta


# ============================================================
# === 3. VISUALIZACIÓN 3D DEL TERRENO VOXELIZADO
# ============================================================
def show_voxel_terrain(grid, step_xy=40, step_z=10):
    """
    Visualiza el terreno voxelizado en 3D.
    """
    print("Generando vista 3D (simplificada)...")
    grid_small = grid[::step_z, ::step_xy, ::step_xy]

    print(f"Tamaño reducido: {grid_small.shape} voxeles -> {np.count_nonzero(grid_small)} activos")
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    ax.voxels(grid_small, facecolors='sienna', edgecolor='none', alpha=0.9)

    ax.set_title("Terreno voxelizado en 3D (DEM → Grid)")
    ax.set_xlabel("X (índices)")
    ax.set_ylabel("Y (índices)")
    ax.set_zlabel("Altura (niveles Z)")
    ax.set_box_aspect([1, 1, 0.3])
    plt.tight_layout()
    plt.show()



# ============================================================
# === 4. EJECUCIÓN PRINCIPAL
# ============================================================
if __name__ == "__main__":
    # Ruta a tu archivo DEM Copernicus (GeoTIFF)
    path = r"C:\Users\Hp\Desktop\GitHub\Radar_coverage_with_EDM\dem_prueba.tif"

    print("Cargando DEM...")
    dem, xs, ys, meta = load_dem(path)
    print("DEM cargado:", dem.shape, f"Elevaciones: {np.nanmin(dem):.1f}–{np.nanmax(dem):.1f} m")

    print("Construyendo grilla tensorial...")
    grid, z_levels, grid_meta = build_tensor_grid(dem, xs, ys, dz=50)
    print("Grilla generada:", grid.shape)

    show_voxel_terrain(grid)
