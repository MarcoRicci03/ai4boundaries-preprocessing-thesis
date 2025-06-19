import rasterio
import geopandas as gpd
from rasterio import features
import numpy as np
import os
from scipy import ndimage
from pathlib import Path
from tqdm import tqdm

def create_mask_from_vector(gdf, template_raster_path, output_path, resolution):
    with rasterio.open(template_raster_path) as template:
        # Converte i confini del raster nel CRS del vettore e ritaglia
        template_bounds = rasterio.warp.transform_bounds(template.crs, gdf.crs, *template.bounds)
        gdf = gdf.to_crs(template.crs).cx[template_bounds[0]:template_bounds[2],
                                           template_bounds[1]:template_bounds[3]]
        
        if len(gdf) == 0:
            width = int((template.bounds.right - template.bounds.left) / resolution)
            height = int((template.bounds.top - template.bounds.bottom) / resolution)
            new_transform = rasterio.transform.from_origin(
                template.bounds.left,
                template.bounds.top,
                (template.bounds.right - template.bounds.left) / width,
                (template.bounds.top - template.bounds.bottom) / height,
            )
            extension = np.zeros((height, width), dtype=np.uint8)
            borders = np.zeros((height, width), dtype=np.uint8)
            distance = np.zeros((height, width), dtype=np.float32)
            vector_mask = np.zeros((height, width), dtype=np.float32)
            
            with rasterio.open(
                output_path,
                "w",
                driver="GTiff",
                compress="lzw",
                height=height,
                width=width,
                count=4,
                dtype=np.float32,
                crs=template.crs,
                transform=new_transform,
                nodata=-10000,
            ) as dst:
                dst.write(extension.astype(np.float32), 1)
                dst.write(borders.astype(np.float32), 2)
                dst.write(distance, 3)
                dst.write(vector_mask, 4)

            return None
        rgb_values = template.read()
        # Nuova trasformazione affine
        new_transform = rasterio.Affine(resolution, 0.0, template.transform.c, 0.0, -resolution, template.transform.f)
        width = int((template.bounds.right - template.bounds.left) / resolution)
        height = int((template.bounds.top - template.bounds.bottom) / resolution)
        
        # Rasterizza le geometrie e crea la maschera booleana
        shapes = [(geom, idx + 2) for idx, geom in enumerate(gdf.geometry)]
        rasterized_uint8 = features.rasterize(
            shapes=shapes,
            out_shape=(height, width),
            transform=new_transform,
            all_touched=True,
            merge_alg=rasterio.enums.MergeAlg.replace,
            default_value=0,
            dtype=np.uint8,
        )
        rasterized = rasterized_uint8.astype(bool)
        
        # Normalizzazione dei canali: applica la scalatura 0-255 per ciascun canale RGB
        rgb_normalized = rgb_values.copy()
        for i in range(3):
            cmin, cmax = rgb_values[i].min(), rgb_values[i].max()
            if cmax > cmin:
                rgb_normalized[i] = ((rgb_values[i] - cmin) * 255 / (cmax - cmin)).astype(np.uint8)
        
        extension = np.where(rasterized, 1, 0).astype(np.uint8)
        eroded = ndimage.binary_erosion(rasterized)
        borders = np.where(rasterized & ~eroded, 1, 0).astype(np.uint8)
        
        # Calcolo del canale distance
        distance_reprojected = rasterio.warp.reproject(
            rgb_normalized[2],
            np.zeros((height, width), dtype=np.float32),
            src_transform=template.transform,
            src_crs=template.crs,
            dst_transform=new_transform,
            dst_crs=template.crs,
            resampling=rasterio.warp.Resampling.bilinear,
        )[0]
        blue_norm = np.zeros((height, width), dtype=np.float32)
        if np.any(rasterized):
            valid = distance_reprojected[rasterized]
            if valid.max() > valid.min():
                blue_norm[rasterized] = (distance_reprojected[rasterized] - valid.min()) / (valid.max() - valid.min())
        distance = np.where(rasterized, blue_norm, 0).astype(np.float32)
        
        # Preparazione della vector_mask: di default NoData (-10000) e solo dentro le geometrie viene assegnato il valore
        vector_mask = np.full((height, width), -10000, dtype=np.float32)
        vector_mask[rasterized] = rasterized_uint8[rasterized].astype(np.float32)
        max_value = shapes[-1][1]
        vector_mask[vector_mask > max_value] = max_value
        
        with rasterio.open(
            output_path,
            "w",
            driver="GTiff",
            compress="lzw",
            height=height,
            width=width,
            count=4,
            dtype=np.float32,
            crs=template.crs,
            transform=new_transform,
            nodata=-10000,
        ) as dst:
            dst.write(extension.astype(np.float32), 1)
            dst.write(borders.astype(np.float32), 2)
            dst.write(distance, 3)
            dst.write(vector_mask, 4)

if __name__ == "__main__":
    template_base_dir = "../../datasets/AI4B/sentinel2/masks"
    vector_path = "../../datasets/AI4B/sampling/ai4boundaries_parcels_vector_sampled.gpkg"
    output_base_dir = "../../datasets/AI4B_SR/sentinel2/masks"
    country_folders = ['AT', 'ES', 'FR', 'LU', 'NL', 'SE', 'SI']

    gdf = gpd.read_file(vector_path)
    
    os.makedirs(output_base_dir, exist_ok=True)
    
    # Calcola il numero totale di file
    total_images = 0
    for country in country_folders:
        country_output_dir = os.path.join(output_base_dir, country)
        os.makedirs(country_output_dir, exist_ok=True)
        template_dir = os.path.join(template_base_dir, country)
        template_files = list(Path(template_dir).glob('*.tif'))
        total_images += len(template_files)
    
    with tqdm(total=total_images, desc="Processing Images", 
              bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]') as pbar:
        for country in country_folders:
            country_output_dir = os.path.join(output_base_dir, country)
            template_dir = os.path.join(template_base_dir, country)
            template_files = list(Path(template_dir).glob('*.tif'))
            for template_path in template_files:
                new_stem = template_path.stem.replace("10m", "2_5m")
                output_path = os.path.join(country_output_dir, f"{new_stem}.tif")
                if os.path.exists(output_path):
                    print(f"File already exists: {output_path}")
                    pbar.update(1)
                    continue
                # Esegui la funzione di processing in modo sequenziale
                result = create_mask_from_vector(gdf, str(template_path), output_path, 2.5)
                pbar.update(1)
                if result:
                    country_name, filename = result
                    pbar.set_postfix({'Country': country_name, 'File': filename}, refresh=False)

    print("Processing complete")