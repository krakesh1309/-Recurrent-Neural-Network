## Predicting Google Stock Prices with Recurrent Neural Networks (RNN)

## Overview

This repository contains code and documentation for building a Recurrent Neural Network (RNN) model to predict Google stock prices. RNNs are particularly suited for sequential data like time series, making them a suitable choice for stock price prediction tasks.

## Dataset
The dataset used for this project is the historical daily stock price data of Google (Alphabet Inc.) obtained from a reliable financial data source.
The dataset includes features such as opening price, closing price, highest price, lowest price, and trading volume.

## Model Architecture
The RNN model architecture consists of one or more LSTM (Long Short-Term Memory) layers followed by one or more dense layers for regression.
LSTM layers are chosen for their ability to retain information over long sequences, making them suitable for capturing temporal dependencies in time series data.
The model is trained to predict the future stock price based on a sequence of past stock prices and other relevant features.

## Requirements
- Python 3.x
- TensorFlow or PyTorch (choose based on preference)
- NumPy
- Pandas
- Matplotlib
- Spyder

##  Notebooks

**Data Preparation**: A notebook demonstrating data preprocessing steps including loading, cleaning, and transforming the Google stock price data.

**RNN Model Training**: A notebook showcasing the implementation and training of the RNN model for stock price prediction.

**Model Evaluation and Prediction**: A notebook for evaluating the trained model and making predictions on unseen data.

## Documentation
For detailed documentation on the project, including data preprocessing, model architecture, training process, and evaluation metrics, refer to the Documentation directory.

## Acknowledgments
[TensorFlow](https://www.tensorflow.org/) or [PyTorch](https://pytorch.org/): For providing powerful libraries for building and training neural networks.

[NumPy](https://numpy.org/) : For providing support for large, multi-dimensional arrays and matrices in Python.

[Pandas](https://pandas.pydata.org/): For providing powerful data manipulation tools in Python.

[Matplotlib](https://matplotlib.org/): For providing data visualization capabilities in Python.
