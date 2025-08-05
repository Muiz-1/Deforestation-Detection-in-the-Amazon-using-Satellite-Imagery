# CNN-Based Change Detection for Deforestation Monitoring

A deep learning project demonstrating land-use change detection by identifying deforestation in the Amazon rainforest using Sentinel-2 satellite imagery. This repository showcases a complete workflow from data acquisition to model training and visualization, directly aligning with key challenges in geospatial AI and environmental monitoring.

## Project Overview

This project addresses the critical need for scalable environmental monitoring by building a system to automatically detect deforestation. By leveraging a Convolutional Neural Network (CNN), the model analyzes pairs of satellite images taken years apart to classify areas where forest has been cleared. This serves as a practical demonstration of using deep learning to convert multi-sensor satellite data into actionable insights for climate and land-use applications.

### Key Features:
* **Change Detection:** Core functionality to identify changes in land use between two time periods.
* **Deep Learning Model:** A custom CNN built in TensorFlow/Keras, optimized for binary image classification on geospatial data.
* **Satellite Data Processing:** A full pipeline using Rasterio and NumPy to process Sentinel-2 multispectral imagery.
* **Data Visualization:** Clear visualizations of model performance and qualitative results on test data.

## Dataset

The model was trained on data that was manually sourced and selected from the **Copernicus Data Space Ecosystem**. This process involved a careful search to identify and download high-quality, cloud-free Sentinel-2 (Level-2A) product tiles for the specific region and timeframes.

* **Satellite:** Sentinel-2 (Level-2A products)
* **Data Source:** Copernicus Data Space Ecosystem (Formerly Copernicus Open Access Hub)
* **Location:** Rondônia, Brazil (A region known for significant deforestation)
* **Data Points:**
    * **"Before" Image:** A cloud-free observation from **August 2018**
    * **"After" Image:** A cloud-free observation from **mid-2025**

## Methodology

The project follows a structured machine learning workflow:

1.  **Data Acquisition & Preprocessing:**
    * Two multispectral images were acquired from the Copernicus Open Access Hub.
    * To prepare the data for the CNN, corresponding `64x64` pixel patches were extracted from both the "before" and "after" images.
    * These two `64x64x3` (RGB) patches were then stacked along the channel axis to create a single `64x64x6` input tensor. This technique allows the CNN to analyze the temporal changes within a single input.

2.  **Model Architecture:**
    * A custom Convolutional Neural Network was designed in TensorFlow. The architecture consists of several convolutional blocks for feature extraction followed by a dense classifier head. Dropout was used to prevent overfitting.

    | Layer Type          | Parameters                 | Output Shape      |
    | ------------------- | -------------------------- | ----------------- |
    | `InputLayer`        | (None, 64, 64, 6)          | (None, 64, 64, 6) |
    | `Conv2D`            | 32 filters, (3,3), ReLU    | (None, 62, 62, 32)|
    | `MaxPooling2D`      | (2,2)                      | (None, 31, 31, 32)|
    | `Conv2D`            | 64 filters, (3,3), ReLU    | (None, 29, 29, 64)|
    | `MaxPooling2D`      | (2,2)                      | (None, 14, 14, 64)|
    | `Flatten`           |                            | (None, 12544)     |
    | `Dense`             | 128 units, ReLU            | (None, 128)       |
    | `Dropout`           | 0.5                        | (None, 128)       |
    | `Dense` (Output)    | 1 unit, Sigmoid            | (None, 1)         |


3.  **Training:**
    * The model was trained using the `Adam` optimizer.
    * `BinaryCrossentropy` was used as the loss function, as this is a binary (Deforested/No Change) classification task.
    * The model was trained for 25 epochs, monitoring validation accuracy to assess performance.


## Future Work & Potential Improvements

This project provides a strong foundation that can be extended in several ways, aligning with complex real-world challenges:

* **Semantic Segmentation:** Transition from classification to a U-Net or similar architecture to produce pixel-level deforestation maps instead of classifying patches.
* **Automated Labeling:** Replace manual coordinate selection with an automated approach using vegetation indices (like NDVI) to create a much larger training dataset.
* **Multi-Sensor Fusion:** Incorporate SAR (Synthetic Aperture Radar) data from Sentinel-1. SAR can penetrate clouds, providing a robust, all-weather monitoring capability that complements optical imagery—a key technique for applications in the Global South.
