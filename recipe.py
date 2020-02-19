#osをimportしたのはなんでや？
import os

#機械学習を行うために「sklearnライブラリ」をインポートします。機械学習全般のアルゴリズムが実装されています。
from sklearn import datasets
from sklearn import model_selection

#preprocessingをインポートしたのはなんでや？
from sklearn import preprocessing

from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris

#深層学習(ニューラルネットワーク)のためにtensorflowをインポート[tf]とおく
import tensorflow as tf

#ニューラルネットワークの構築のためにkerasをインポート
import keras
from keras.models import Sequential
from keras.layers.core import Dense, Activation
import pandas as pd

#数値計算のために「numpy」をインストール *[np]とおく（多次元配列)
import numpy as np

#表計算のために「pandas」をインストール *[pd]とおく
import pandas as pd

#乱数を生み出すためにrandomをインポート
import random

ingredient_df = pd.read_excel("C:/Users/keita/Desktop/portfolio/鶴山さんポートフォリオ原料.xlsx", index_col = 0)
recipe_df_import_data = pd.read_excel("C:/Users/keita/Desktop/portfolio/鶴山さんポートフォリオ完成.xlsx", index_col = 0)
recipe_df = recipe_df_import_data.iloc[:,:14]
recipe_df_ingredient = recipe_df_import_data.iloc[:,14:]

print(recipe_df)
print(recipe_df_import_data)

ingredient_mean_df_list = []

for number in range(len(recipe_df_ingredient)):
    ingredient_list = [ing for ing in recipe_df_ingredient.iloc[number,:]]
    part_dict = {}
    for part in ingredient_list:
        if type(part) == str:
            data = {part:ingredient_df.loc[part,:].values}
            
            part_dict.update(data)
            
    ingredient_mean_df_list.append(pd.DataFrame(part_dict).T.mean())

# 説明変数
train_2_x = pd.DataFrame(ingredient_mean_df_list)

# 目的変数
train_2_recipe_df_nan_Y = recipe_df.drop("総合評価",axis = 1)

Flag = True

input_ingredient_list = []

input_ingredient_mean_df_list = []

input_part_dict = {}

while Flag:
    user_input = input("具材を入力してください >>")
    if user_input in list(ingredient_df.index):
        input_ingredient_list.append(user_input)
    else:
        print("もう一度入力してください。")
    another_user_input = input("他の具材を入力する場合は【1】を、終了する場合は【0】を入力してください。 >>")
    if another_user_input == "0":
        Flag = False

for input_part in input_ingredient_list:        
    ing_data = {input_part:ingredient_df.loc[input_part,:].values}
    input_part_dict.update(ing_data)
otanoshimi_DataFrame = pd.DataFrame(input_part_dict).T.mean()

hako = []
hako_model_compile = []
hako_train_test_split = []
l = []

for i in range(len(train_2_recipe_df_nan_Y.columns)):
    x_2_train, x_2_test, Y_2_train, Y_2_test = train_test_split(train_2_x.values, train_2_recipe_df_nan_Y.iloc[:,i].values, test_size = 0.2)
    print("debug1", x_2_test)

    hako_train_test_split.append([x_2_train, x_2_test, Y_2_train, Y_2_test])
    print("debug2", x_2_test)
   
    model_2 = tf.keras.models.Sequential([tf.keras.layers.Dense(13 ,activation = tf.nn.softmax),
                                          tf.keras.layers.Dense(12 ,activation = tf.nn.softmax),
                                          tf.keras.layers.Dense(11 ,activation = tf.nn.softmax),
                                          tf.keras.layers.Dense(11 ,activation = tf.nn.softmax)])
    
    hako.append([model_2])
        
    hako_model_compile.append(model_2.compile(optimizer="adam",loss="sparse_categorical_crossentropy",metrics=["accuracy"]))

    
for j in range(len(hako_model_compile)):
    hako[j].append(hako_model_compile[j])
    
for i in range(len(ingredient_df.columns)):
    hako[i][0]
    hako[i][1]
    hako[i][0].fit(hako_train_test_split[i][0],hako_train_test_split[i][2] ,
                         batch_size = 100,
                         epochs = 50,
                         verbose = 1,
                         validation_split = 0.8)

                        #  https://www.sejuku.net/blog/72890

    print("debug----------------------")
    print(otanoshimi_DataFrame.values)                         
    print(x_2_test[0])
    b = hako[i][0].predict(np.vstack((otanoshimi_DataFrame.values,x_2_test)))[0]
    l.append(list(b).index(max(b)))

train_recipe_df_nan_x = recipe_df.drop("総合評価",axis = 1)

train_recipe_df_nan_y = recipe_df[['総合評価']]
# float型にしないと「train_test_split」出来なかった。
train_recipe_df_nan_x["甘味"] = train_recipe_df_nan_x["甘味"].astype("float64")

x_train, x_test, y_train, y_test = train_test_split(train_recipe_df_nan_x.values, train_recipe_df_nan_y.values, test_size = 0.2)
model_rasuto = tf.keras.models.Sequential([tf.keras.layers.Dense(13 ,activation = tf.nn.relu),
                                           tf.keras.layers.Dense(13 ,activation = tf.nn.softmax)])


model_rasuto.compile(optimizer="adam",loss="sparse_categorical_crossentropy",metrics=["accuracy"])

model_rasuto.fit(x_train, y_train,
                 batch_size = 100,
                 epochs = 1000,
                 verbose = 1,
                 validation_split = 0.8)

a = model_rasuto.predict(np.vstack((l,x_test[0])))[0]

print(list(a).index(max(a)))

