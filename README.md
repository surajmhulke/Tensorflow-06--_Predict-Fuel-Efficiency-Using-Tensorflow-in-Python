# Tensorflow-_Predict-Fuel-Efficiency-Using-Tensorflow-in-Python
 
  

In this project, we will build a machine learning model to predict fuel efficiency using TensorFlow. The dataset contains features such as the distance an engine has traveled, the number of cylinders in the car, and other relevant attributes. This project will guide you through the process of loading the data, exploring it, and building a predictive model.

## Table of Contents

- [Introduction](#introduction)
- [Importing Libraries](#importing-libraries)
- [Importing Dataset](#importing-dataset)
- [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)
- [Data Preprocessing](#data-preprocessing)
- [Data Input Pipeline](#data-input-pipeline)
- [Model Architecture](#model-architecture)
- [Model Training](#model-training)
- [Conclusion](#conclusion)

## Introduction

Predicting fuel efficiency is a critical task that can help us better understand and optimize vehicle performance. In this project, we aim to build a machine learning model using TensorFlow to predict fuel efficiency. We will analyze the dataset, preprocess the data, build a predictive model, and evaluate its performance.

## Importing Libraries

We start by importing the necessary libraries for data handling, visualization, machine learning, and deep learning.

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
import tensorflow as tf
from tensorflow import keras
from keras import layers
import warnings
warnings.filterwarnings('ignore')
```

## Importing Dataset

The dataset used in this project can be downloaded from [dataset link]. Load the dataset and examine its contents.

```python
df = pd.read_csv('auto-mpg.csv')
df.head()
```

## Exploratory Data Analysis (EDA)

Perform an exploratory data analysis to understand the dataset, its shape, data types, and unique values. EDA will help in identifying any data discrepancies.

## Data Preprocessing

Preprocess the data, handling any missing or incorrect values and making necessary data type conversions.

## Data Input Pipeline

Prepare the data input pipeline using TensorFlow's data processing capabilities to efficiently train the model.

## Model Architecture

Define the architecture of the machine learning model using TensorFlow's Keras API. Configure the model with layers, dropout, and normalization to achieve the desired predictive capability.

```python
model = keras.Sequential([
    layers.Dense(256, activation='relu', input_shape=[6]),
    layers.BatchNormalization(),
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.3),
    layers.BatchNormalization(),
    layers.Dense(1, activation='relu')
])
```

## Model Training

Train the model using the preprocessed data. Monitor training and validation losses and metrics. Adjust hyperparameters as needed to improve performance.

```python
history = model.fit(train_ds,
                    epochs=50,
                    validation_data=val_ds)
```

## Conclusion

 ![image](https://github.com/surajmhulke/Tensorflow-_Predict-Fuel-Efficiency-Using-Tensorflow-in-Python/assets/136318267/8d6cc5d1-3bed-4e58-b191-ed7f76c03e24)
The training error has gone down smoothly but the case with the validation is somewhat different.
 
In this project, we successfully built a fuel efficiency prediction model using TensorFlow. The model's performance is promising, but there is room for further improvement. This project serves as a starting point for predicting fuel efficiency and can be extended for more advanced analysis.
 

  

 
