# 🧠 Anomaly Detection in ECG Signals Using CNN Autoencoders.

![yato](https://github.com/user-attachments/assets/748060e0-c37a-48ad-89a3-60dbc727a28c)

This project implements a Convolutional Neural Network (CNN) based autoencoder for anomaly detection in electrocardiogram (ECG) signals. The model is trained to reconstruct normal heartbeats and flags deviations as potential anomalies, such as those caused by cardiac disorders.

## 📊 Dataset Overview

We use the **PTB Diagnostic ECG Database** sourced from Physionet:

- **Samples:** 14,552 ECG recordings
- **Classes:** Normal vs. Abnormal heartbeats
- **Sampling Frequency:** 125 Hz

This high-resolution dataset enables in-depth analysis of heartbeat anomalies using deep learning.

## 🧠 Methodology

- **Preprocessing**: Signals are segmented, normalized, and reshaped for CNN compatibility.
- **Model Architecture**:
  - CNN Autoencoder with Conv1D layers
  - Experimented with both **upsampling** and **transposed convolution** in decoder
- **Loss Function**: Mean Absolute Error (MAE)
- **Evaluation**:
  - Threshold-based classification on reconstruction error
  - Metrics used: Accuracy, Precision, Recall, F1 Score

## 🚀 Results

| Method                | Accuracy | Precision | Recall | F1 Score |
|----------------------|----------|-----------|--------|----------|
| With Upsampling      | 71.99%   | 49.78%    | 87.91% | 63.57%   |
| With Transposed Conv | 76.93%   | 55.23%    | 89.81% | 68.40%   |

> Using **transposed convolution** significantly improved performance across all metrics.

![yato 2](https://github.com/user-attachments/assets/e607e5c6-b6ef-4474-ba07-08c6f28eb2c7)

## 🛠 Libraries & Tools

- TensorFlow / Keras
- NumPy, Pandas
- Matplotlib, Seaborn
- Scikit-learn

## 📁 Structure

