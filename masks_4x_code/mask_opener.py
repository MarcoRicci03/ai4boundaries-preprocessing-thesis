mask_path = '../../datasets/AI4B_SR/sentinel2/masks_prova/NL/NL_4921_S2label_2_5m_256.tif'
import rasterio
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

# Percorso al file .tif
# Caricamento delle bande
with rasterio.open(mask_path) as src:
    field_extent = src.read(1)
    field_boundary = src.read(2)
    distance_transform = src.read(3)
    field_instance = src.read(4).astype(np.float32)

# Maschera per valori > 0 nella field_instance
field_instance_masked = field_instance.copy()
field_instance_masked[field_instance_masked <= 0] = np.nan

# Colormap e normalizzazione per pseudocolor continuo
norm_instance = Normalize(vmin=2, vmax=92)
cmap_instance = plt.get_cmap("Reds")

# Crea 4 subplot orizzontali
fig, axs = plt.subplots(1, 4, figsize=(20, 5), constrained_layout=True)

# Dati e titoli
titles = [
    "Field Extent",
    "Field Boundary",
    "Distance from Boundary",
    "Field Instance (Pseudocolor)"
]
images = [
    axs[0].imshow(field_extent, cmap='gray'),
    axs[1].imshow(field_boundary, cmap='gray'),
    axs[2].imshow(distance_transform, cmap='magma'),
    axs[3].imshow(field_instance_masked, cmap=cmap_instance, norm=norm_instance)
]

# Imposta titoli e rimuove gli assi
for ax, title in zip(axs, titles):
    ax.set_title(title)
    ax.axis('off')

# Aggiunge la colorbar fuori dai subplot
cbar = fig.colorbar(
    ScalarMappable(norm=norm_instance, cmap=cmap_instance),
    ax=axs, location='right', shrink=0.8, label='Instance Value'
)

output_path = 'mask_visualization.png'  # oppure .jpg, .tif, etc.
fig.savefig(output_path, dpi=300, bbox_inches='tight')
plt.close(fig)

print(f"Figura salvata in: {output_path}")
