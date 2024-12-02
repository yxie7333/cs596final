# cs596final
In this project, our goal is to study how parallel processing techniques can improve the efficiency and training speed of LSTM models, a type of Recurrent Neural Network (RNN). We will benchmark the efficiency of training the same model using GPU-based computing compared to CPU-based computing, focusing on how GPU acceleration reduces training time. Additionally, we will evaluate the performance of different GPUs, such as Nvidia T40 and A100, to analyze their impact on model training efficiency.

# Team
* Wen Lin
* Yanjun Liu
* Yuchen Xie
  
# Reference

- **Parallel LSTM for Sequence Prediction from Sequential Data:** [GitHub](https://github.com/baobuiquang/ParallelLSTM/tree/main)
  
# Dataset
## Information
Dataset Name: ERA5 Weather Dataset

Description:
* ERA5 provides hourly weather data on a global scale. The Google Cloud Public Dataset Program hosts ERA5 data that spans from 1940 to May 2023.
* The dataset includes multiple features (e.g., temperature, humidity, wind speed) at high spatial and temporal resolutions, which makes it computationally intensive to process sequentially.

Google Cloud Link: https://cloud.google.com/storage/docs/public-datasets/era5
## Task to solve
Predict temperature for a certain Location.

Since weather data is continuous and has strong time dependencies, LSTMs are designed to handle this. 

# Parallel Strategy
There are several parallel strategies for improving the efficiency of LSTM models, such as data parallelism, model parallelism, and temporal parallelism, among others. In this project, we are using data parallelism, which splits the dataset into smaller chunks and processes them in parallel on GPUs, as illustrated in the figure below.
![strategy image](strategy.png)
