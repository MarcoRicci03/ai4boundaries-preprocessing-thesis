# ai4boundaries-preprocessing-thesis

# Preprocessing and Super-Resolution for the AI4Boundaries Dataset

This repository contains the code and methodologies developed for the Bachelor's thesis "Preprocessing, super-risoluzione e ottimizzazione delle maschere del dataset AI4Boundaries" by Marco Ricci, Università degli Studi dell'Insubria (A.A. 2024-2025).

The primary goal of this project is to enhance the AI4Boundaries dataset by applying super-resolution techniques to Sentinel-2 satellite imagery and developing a robust pipeline for upscaling the corresponding segmentation masks. The hypothesis is that improving the spatial quality of the input data can lead to more accurate and detailed delineation of agricultural field boundaries.

## Project Structure

This repository is organized into the following main directories:

-   **/sr4rs/**: Contains the script (`infer_nc_all.py`) developed to perform super-resolution on the original NetCDF image files. This script utilizes the pre-trained model from the SR4RS framework to upscale images from 10m to 2.5m resolution.
-   **/masks_4x_code/**: Includes the Python scripts responsible for the 4x upscaling of the segmentation masks. This process generates high-resolution ground truth masks from the original vector data to match the super-resolved images, avoiding the pitfalls of traditional raster interpolation.
-   **/ai4b/**: This directory likely contains the code related to the `UNet3DMultitask` model, including scripts for data loading, training, and validation on both the original and the preprocessed datasets. 

## Experimental Pipeline

The project follows a two-stage experimental pipeline:

1.  **Data Enhancement**:
    -   **Image Super-Resolution**: The original 10m Sentinel-2 images are upscaled to 2.5m resolution using the SR4RS model.
    -   **Mask Upscaling**: The ground truth masks are re-rasterized from vector polygons at a target resolution of 2.5m to ensure geometric consistency.
2.  **Model Validation**:
    -   A `UNet3DMultitask` model is trained and evaluated on two separate datasets: the original low-resolution data (baseline) and the enhanced high-resolution data, allowing for a direct comparison of performance.

## Installation and Usage

To set up the environment and run the scripts, follow these steps:

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/MarcoRicci03/ai4boundaries-preprocessing-thesis](https://github.com/MarcoRicci03/ai4boundaries-preprocessing-thesis)
    cd ai4boundaries-preprocessing-thesis
    ```

2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Run the pipeline:**
    *(Fornisci qui un esempio di come avviare i tuoi script principali)*
    ```bash
    # Esempio per lanciare la super-risoluzione
    python sr4rs/infer_nc_all.py --gpu 0 --model_dir path/to/sr4rs_model --input_dir path/to/lr_images --output_path path/to/sr_images

    # Esempio per generare le maschere
    python masks_4x_code/generate_masks.py --input path/to/vector_data --output path/to/hr_masks
    ```

## Acknowledgments

This work builds upon foundational research and utilizes powerful open-source tools from the scientific community. Full credit is given to the original authors.

-   **SR4RS Framework (for Super-Resolution Model):**
    -   The pre-trained model used for image upscaling is from the SR4RS framework.
    -   **Software:** [SR4RS on GitHub](https://github.com/remicres/sr4rs) by Rémi Cresson.
    -   **Paper:** Cresson, R. (2022). "SR4RS: A Tool for Super Resolution of Remote Sensing Images". *Journal of Open Research Software*, 10(1). [DOI: 10.5334/jors.369](http://doi.org/10.5334/jors.369).

-   **UNet3DMultitask Architecture (for Segmentation Model):**
    -   The segmentation model architecture is based on the concepts presented in the "Tackling fluffy clouds" paper.
    -   **Paper:** Foivos I. Diakogiannis, et al. (2024). "Tackling fluffy clouds: field boundaries detection using time series of S2 and/or S1 image".
    -   **Source:** [arXiv:2409.13568](https://arxiv.org/abs/2409.13568).
    -   **Software** [tfcl on Github] (https://github.com/feevos/tfcl/tree/master)

## How to Cite

If you find this work useful in your research, please consider citing the thesis:

> Ricci, M. (2025). *Preprocessing, super-risoluzione e ottimizzazione delle maschere del dataset AI4Boundaries*. Bachelor's Thesis, Università degli Studi dell'Insubria, Como, Italy.
