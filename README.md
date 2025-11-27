# Impact of Sampling Frequency on HAR Classification  
*A Reproducible Python Pipeline for Human Activity Recognition Using HHAR*

This project implements a **scalable and reproducible pipeline** for evaluating how the **reduction of sampling frequency** affects the performance of several machine learning models in **Human Activity Recognition (HAR)** using inertial sensor data (accelerometers) from the **HHAR dataset**.

The work is motivated by real-world constraints in **IoT and embedded systems**, where memory, compute, and battery life are critical and full-resolution signals might be unnecessary or too costly to process.

---

## Dataset  
The analysis is based on the **Heterogeneity Human Activity Recognition (HHAR)** dataset, containing:

- **43,930,257 rows** of accelerometer/gyroscope signals  
- **9 users**, **12 devices** (phones + smartwatches)  
- **6 activities**: Biking, Sitting, Standing, Walking, Stairs Up, Stairs Down  
- Signals recorded at **maximum device-permitted frequency**, causing heterogeneity  

To avoid timestamp desynchronization between accelerometer and gyroscope, the study uses **only accelerometer data**, which still provides enough discriminative power.

Link to [UC Irvine HHAR dataset here](https://archive.ics.uci.edu/dataset/344/heterogeneity+activity+recognition)

---

## Pipeline Architecture  
The complete pipeline is implemented in Python and structured as follows:

main.py: Orchestrates the full pipeline

config.yaml: Controls model type, sampling stride, windowing, etc.

preprocessing.py: Cleans dataset, adds row_id, performs sub-sampling

processing.py: Windowing + feature extraction / raw segmentation

classification.py: Classical ML models (RF, XGB, HGB)

timeSeries.py: CNN models for raw time-series

customLogger.py: Logs every run to console + txt file

### Key Features of the Pipeline  
- **Distributed data processing with Dask** (for handling 44M rows efficiently)  
- **Windowing (512 samples, 50% overlap)**  
- Supports two modelling strategies:
  1. **Feature-based (statistical summary)** → good for tree models  
  2. **Raw time-series (512×3 windows)** → for CNN models  
- **Programmatic downsampling** via integer stride on `row_id`, enabling reproducible sampling reduction  
- **Full run reproducibility** through `config.yaml`  

---

## Models Implemented  
### Classical ML (feature-based)
- **Random Forest**  
- **XGBoost**  
- **Histogram-based Gradient Boosting (HGB)**  

These models use manually engineered features:  
- mean, std, min, max per axis (X,Y,Z)

### Deep Learning (raw signal)
- **1D Convolutional Neural Network (CNN)**  
  - Conv1D (64 filters, kernel=5) × 2  
  - MaxPooling  
  - Dense(128) + Dropout  
  - Softmax output  

---

## Sampling Frequency Reduction  
Because the dataset lacks a stable sampling frequency, classical time-based resampling is impossible.  
Instead, the pipeline introduces a **row-based virtual sampling frequency**:

downsample_stride = k   # keep 1 of every k samples

Values used:  
`0 (full data), 2, 5, 10, 20, 50` → equivalent to **100%, 50%, 20%, 10%, 5%, 2%** of the original dataset  

---

## Summary of Results  
(All results computed using stratified user-based train/test split.)

### Global Accuracy Trends  
Across all models, accuracy **remains stable down to ~20% of original data**, then drops rapidly.  
Tree-based models are consistently more robust than CNNs.

| Model | 100% | 50% | 20% | 10% | 5% | 2% |
|-------|------|------|------|------|------|------|
| **Random Forest** | 0.92 | **0.93** | 0.88 | 0.87 | 0.81 | 0.73 |
| **XGBoost** | 0.91 | **0.93** | 0.87 | 0.87 | 0.82 | 0.69 |
| **Gradient Boosting** | 0.89 | **0.92** | 0.87 | 0.86 | 0.79 | 0.75 |
| **CNN (raw)** | 0.87 | 0.88 | 0.80 | 0.81 | 0.67 | 0.63 |

**Key observation:**  
> Some models actually improve at **50% of the data**, likely due to noise reduction.

Results for each of the iterations on each model can be found inside the folders as .txt files 

---

## Class-Specific Behavior  
Activities depending on **high-frequency signal changes** degrade fastest:  
- **Bike**  
- **Stairs Up**  
- **Stairs Down** (worst degradation)  

Examples (F1-score):

- Bike: from **0.90 → 0.70** (XGBoost)  
- StairsDown: **0.85 → 0.21** (XGBoost)  
- Stand: stays above **0.90** until <10%  

These findings are consistent with biomechanics literature showing that descending stairs generates highly irregular, high-impact motion patterns sensitive to downsampling.

---

## Main Conclusions  
1. **Tree-based models with statistical features are the most robust** under reduced sampling.  
2. **CNNs require high-resolution data**; performance collapses below ~20%.  
3. Activities with **high temporal variability** degrade fastest.  
4. **20% of the data** (~5× reduction) still preserves high accuracy (>0.85).  
5. For IoT / embedded devices:  
   - **Statistical-feature + tree models** → better energy efficiency + robustness  
   - **CNN on raw data** → simpler inference pipeline, but needs higher sampling  

---

## How to Run the Pipeline

### 1. Download the dataset HHAR on a known location

### 2. Configure experiment  
Modify `config.yaml`:

```yaml
model: xgboost
downsample_stride: 5
window_size: 512
overlap: 256
dataset_path: "./path/to/csv/sensor/file"
epochs: 15
batch_size: 128
