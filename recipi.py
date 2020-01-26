
from sklearn import datasets
from sklearn import model_selection
from sklearn import preprocessing
from keras.models import Sequential
from keras.layers.core import Dense, Activation
import pandas as pd
import random
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris


ing_df = pd.read_excel("鶴山さんポートフォリオ完成.xlsx",index_col=0)
recepi_df = pd.read_excel("鶴山さんポートフォリオ原料.xlsx",index_col=0)
rec_df = recepi_df.iloc[:,:14]
rec_df_ing = recepi_df.iloc[:,14:]

ing_df

def xx(x):
    if x >= 5:
        return 1
    else :
        return 0
for i in range(len(rec_df)):
    rec_df.iloc[i,13] = xx(rec_df.iloc[i,13])

rec_df

rec_df_ing 

train_rec_df_nan_x = rec_df.drop("総合評価",axis = 1)
train_rec_df_nan_y = rec_df[['総合評価']]

train_rec_df_nan_x["甘味"] = train_rec_df_nan_x["甘味"].astype("float64")

train_rec_df_nan_x.dtypes

x_train, x_test, y_train, y_test = train_test_split(train_rec_df_nan_x.values, train_rec_df_nan_y.values, test_size = 0.2)

model = tf.keras.models.Sequential([tf.keras.layers.Dense(13 ,activation = tf.nn.relu),
                                    tf.keras.layers.Dense(11 ,activation = tf.nn.softmax),
                                    tf.keras.layers.Dense(8 ,activation = tf.nn.softmax),
                                     tf.keras.layers.Dense(8 ,activation = tf.nn.softmax),
                                    tf.keras.layers.Dense(4 ,activation = tf.nn.relu),
                                    tf.keras.layers.Dense(3 ,activation = tf.nn.softmax),
                                    tf.keras.layers.Dense(2 ,activation = tf.nn.relu)])

model.compile(optimizer="adam",loss="sparse_categorical_crossentropy",metrics=["accuracy"])

model.fit(x_train, y_train,
          batch_size = 100,
          epochs = 1000,
          verbose = 1,
          validation_split = 0.8)

