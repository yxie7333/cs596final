# cs596final
In this project, we aim to enhance the efficiency and training speed of the LSTM model, a type of Recurrent Neural Network (RNN). By leveraging parallel processing techniques and GPU computing, we will enable the model to handle larger datasets and significantly reduce training time.

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
* ERA5 provides hourly weather data on a global scale, covering decades of observations.
* The dataset includes multiple features (e.g., temperature, humidity, wind speed) at high spatial and temporal resolutions, which makes it computationally intensive to process sequentially.

Google Cloud Link: https://cloud.google.com/storage/docs/public-datasets/era5
## Task to solve
Predict temperature for a certain place (????)

Since weather data is continuous in nature and has strong time dependencies, which LSTMs are designed to handle. 

# Parallel Strategy
There are seveal parallel strategies for improving the efficiency of LSTM models, such as data parallelism, model parallelism, temporal parallelism, among others. In this project, we are using data parallelism which Split the dataset into smaller chunks and process them in parallel on multiple devices.
