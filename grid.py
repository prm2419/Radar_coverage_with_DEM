# grid.py — Generación de la grilla tensorial 3D (voxel grid) a partir de un DEM.


import numpy as np

def build_tensor_grid(dem, x_coords, y_coords, dz=30, z_padding=500):
    """
    Versión vectorizada — crea una grilla 3D a partir del DEM de forma eficiente.
    """
    H, W = dem.shape
    dx = np.abs(x_coords[1] - x_coords[0])
    dy = np.abs(y_coords[1] - y_coords[0])

    z_min = np.nanmin(dem)
    z_max = np.nanmax(dem) + z_padding
    z_levels = np.arange(z_min, z_max + dz, dz)
    Nz = len(z_levels)

    # Reemplazar NaN por mínimo (para evitar problemas de comparación)
    dem_filled = np.nan_to_num(dem, nan=z_min)

    # Expandir el DEM en un eje vertical para comparar todo a la vez
    # z_levels[:, None, None] tiene forma (Nz,1,1)
    # dem_filled[None, :, :] tiene forma (1,Ny,Nx)
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


if __name__ == "__main__":
    from dem_io import load_dem
    import matplotlib.pyplot as plt

    path = r"C:\Users\Hp\Desktop\GitHub\Radar_coverage_with_EDM\dem_prueba.tif"
    dem, xs, ys, meta = load_dem(path)

    grid, z_levels, grid_meta = build_tensor_grid(dem, xs, ys, dz=50)

    print("Tamaño de la grilla:", grid.shape)
    print("Altura máxima (m):", grid_meta["z_extent"][1])

    # Visualizar un corte vertical (por ejemplo, en la mitad del valle)

    mid_y = grid.shape[1] // 2
    x_km = (xs - xs.min()) * 111.32 * 1000 * np.cos(np.deg2rad(np.mean(ys))) / 1000  # aprox. en km

    plt.figure(figsize=(10,5))
    plt.imshow(grid[:, mid_y, :], cmap="gray_r", origin="lower",
            extent=[x_km.min(), x_km.max(), z_levels.min(), z_levels.max()],
            aspect='auto')
    plt.xlabel("Distancia aproximada Este-Oeste (km)")
    plt.ylabel("Altura (m)")
    plt.title("Corte vertical del terreno en la grilla tensorial")
    plt.show()

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # activa el modo 3D

# Suponiendo que ya tienes:
# dem, xs, ys = load_dem(...)
X, Y = np.meshgrid(xs, ys)

fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

# Graficar superficie
surf = ax.plot_surface(X, Y, dem, cmap='terrain', linewidth=0, antialiased=False)

ax.set_title("Superficie 3D del terreno (DEM)")
ax.set_xlabel("Longitud (°)")
ax.set_ylabel("Latitud (°)")
ax.set_zlabel("Altura (m)")
fig.colorbar(surf, shrink=0.5, aspect=10, label="Altura (m)")

plt.show()
