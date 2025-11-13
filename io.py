#io.py — Lectura y manejo de datos de elevación (DEM)

import numpy as np
import rasterio
from rasterio.enums import Resampling


def load_dem(path_dem, target_resolution=None, resampling=Resampling.bilinear):
    """
    Lee un archivo DEM y devuelve la matriz de elevación, coordenadas y metadatos.

    Parámetros
    ----------
    path_dem : str
        Ruta del archivo DEM (GeoTIFF, .tif, .asc, etc.)
    target_resolution : float or None
        Si se especifica (en unidades del CRS, usualmente metros),
        el DEM se reescala a esa resolución.
    resampling : rasterio.enums.Resampling
        Método de remuestreo (default: bilinear).

    Retorna
    -------
    dem_data : np.ndarray [H, W]
        Matriz 2D con elevaciones (NaN en nodata).
    x_coords : np.ndarray [W]
        Coordenadas X (Este / Longitud).
    y_coords : np.ndarray [H]
        Coordenadas Y (Norte / Latitud), ordenadas de norte a sur.
    meta : dict
        Metadatos raster completos (crs, transform, resolución, etc.)
    """

    with rasterio.open(path_dem) as src:
        # ¿Es necesario reescalar?
        """
        if target_resolution is not None:
            scale_factor_x = src.res[0] / target_resolution                     
            scale_factor_y = src.res[1] / target_resolution
            new_width = int(src.width * scale_factor_x)
            new_height = int(src.height * scale_factor_y)
            dem_data = src.read(1,                  
            out_shape=(new_height, new_width),
            resampling=resampling)  
            transform = src.transform * src.transform.scale(    
            (src.width / new_width),
            (src.height / new_height)
        )                           
        else:       
        """
        dem_data = src.read(1)
        transform = src.transform

        dem_data = dem_data.astype(np.float32)
        dem_data[dem_data == src.nodata] = np.nan

        # Coordenadas X, Y a partir de transform affine
        H, W = dem_data.shape
        x_coords = np.arange(W) * transform.a + transform.c + transform.a / 2
        y_coords = np.arange(H) * transform.e + transform.f + transform.e / 2

        # Ajustar si Y está invertida (típico en imágenes raster)
        if y_coords[1] < y_coords[0]:
            y_coords = y_coords[::-1]
            dem_data = np.flipud(dem_data)

        meta = src.meta.copy()
        meta.update({
            'transform': transform,
            'width': dem_data.shape[1],
            'height': dem_data.shape[0],
        })

    return dem_data, x_coords, y_coords, meta

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # Ruta a tu DEM
    path = "C:\\Users\\Hp\\Desktop\\GitHub\\Radar_coverage_with_EDM\\dem_prueba.tif"

    dem, xs, ys, meta = load_dem(path, target_resolution=30)

    print("Resolución (x,y):", meta['transform'].a, meta['transform'].e)
    print("Dimensiones:", dem.shape)
    print("Alturas (min, max):", np.nanmin(dem), np.nanmax(dem))

    plt.imshow(dem, cmap='terrain', extent=[xs.min(), xs.max(), ys.min(), ys.max()])
    plt.colorbar(label="Elevación (m)")
    plt.title("DEM cargado")
    plt.xlabel("X (m)")
    plt.ylabel("Y (m)")
    plt.show()

