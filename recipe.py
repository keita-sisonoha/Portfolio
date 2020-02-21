import os

#機械学習を行うために「sklearnライブラリ」をインポートします。機械学習全般のアルゴリズムが実装されています。
from sklearn import datasets
from sklearn import model_selection
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

#原材料のデータをエクセルから読み込む
ingredient_df = pd.read_excel("C:/Users/keita/Desktop/portfolio/鶴山さんポートフォリオ原料.xlsx", index_col = 0)

#料理完成形データをエクセルから読み込む
recipe_df_import_data = pd.read_excel("C:/Users/keita/Desktop/portfolio/鶴山さんポートフォリオ完成.xlsx", index_col = 0)

#料理完成形データの数値部分だけを抜き取る
recipe_df = recipe_df_import_data.iloc[:,:14]

#料理完成形データの材料名の部分だけを抜き取る
recipe_df_ingredient = recipe_df_import_data.iloc[:,14:]


#空のリストを作っておく。
ingredient_mean_df_list = []

#レシピの数をとってくる
for number in range(len(recipe_df_ingredient)):
    #それぞれのレシピの材料をリストに格納する
    ingredient_list = [ing for ing in recipe_df_ingredient.iloc[number,:]]
    #空のディクトを作っておく
    part_dict = {}
    #先程作ったレシピ材料のリストから、１つ１つ食材をとってくる
    for part in ingredient_list:
        #str型にしていする。(欠損値を除くため)
        if type(part) == str:
　　　　　　　#それぞれの食材のデータを、原材料のデータからとってくる
            data = {part:ingredient_df.loc[part,:].values}
　　　　　　　#空のディクトに入れる
            part_dict.update(data)
    #個々の料理に使われている食材データの平均を空のリストに格納する            
    ingredient_mean_df_list.append(pd.DataFrame(part_dict).T.mean())

#説明変数を「train_2_x」とする。「train_2_x」は、個々の料理に使われている食材データの平均が格納されたリスト
train_2_x = pd.DataFrame(ingredient_mean_df_list)

#目的変数を「train_2_recipe_df_nan_Y」とする。「train_2_recipe_df_nan_Y」は、個々の料理のデータ(*総合評価のデータのみ除く)
train_2_recipe_df_nan_Y = recipe_df.drop("総合評価",axis = 1)

#FlagをTrueとおく。ブーリアン型で表現できるようにするため。
Flag = True

#空のリストを作っておく。
input_ingredient_list = []

#空のリストを作っておく。
input_ingredient_mean_df_list = []

#空のディクトを作っておく。
input_part_dict = {}

#もし、Flag(true)なら、使用者に具材を入力してもらう。
while Flag:
    user_input = input("具材を入力してください >>")

　　#使用者が入力した具材が原材料のデータに含まれていたら、その食材を空のリストに格納する
    if user_input in list(ingredient_df.index):
        input_ingredient_list.append(user_input)

　　#使用者が入力した具材が原材料のデータに含まれていなかったら、再度入力をお願いする
    else:
        print("もう一度入力してください。")

　　#他の食材を入力するか、判断してもらう。もし、入力しないを選択した場合、「Flag」を「False」にする。
    another_user_input = input("他の具材を入力する場合は【1】を、終了する場合は【0】を入力してください。 >>")
    if another_user_input == "0":
        Flag = False

#入力してもらった食材材を１つ１つ取り出す。
for input_part in input_ingredient_list:
　　#入力してもらった食材のデータを、原材料のデータから取り出す。     
    ing_data = {input_part:ingredient_df.loc[input_part,:].values}
   #空のディクトに格納する
    input_part_dict.update(ing_data)

#入力してもらった食材データの平均をだす。
otanoshimi_DataFrame = pd.DataFrame(input_part_dict).T.mean()

#空のリストを用意する
hako = []
#空のリストを用意する
hako_model_compile = []
#空のリストを用意する
hako_train_test_split = []
#空のリストを用意する
l = []

#目的変数のコラムの数を取ってくる
for i in range(len(train_2_recipe_df_nan_Y.columns)):
　　#それぞれのtestとtrainに分割する
    x_2_train, x_2_test, Y_2_train, Y_2_test = train_test_split(train_2_x.values, train_2_recipe_df_nan_Y.iloc[:,i].values, test_size = 0.2)
　　#空のリストに格納する
    hako_train_test_split.append([x_2_train, x_2_test, Y_2_train, Y_2_test])

   #それぞれモデルを作る
    model_2 = tf.keras.models.Sequential([tf.keras.layers.Dense(13 ,activation = tf.nn.softmax),
                                          tf.keras.layers.Dense(12 ,activation = tf.nn.softmax),
                                          tf.keras.layers.Dense(11 ,activation = tf.nn.softmax),
                                          tf.keras.layers.Dense(11 ,activation = tf.nn.softmax)])
   #モデルを空のリストに格納する
    hako.append([model_2])
    
　　#モデルを定義する
    hako_model_compile.append(model_2.compile(optimizer="adam",loss="sparse_categorical_crossentropy",metrics=["accuracy"]))

#それぞれのモデルを順に適応させる
for j in range(len(hako_model_compile)):
    hako[j].append(hako_model_compile[j])

#それぞれのモデルを適応させていく     
for i in range(len(ingredient_df.columns)):
    hako[i][0]
    hako[i][1]
    hako[i][0].fit(hako_train_test_split[i][0],hako_train_test_split[i][2] ,
                         batch_size = 100,
                         epochs = 50,
                         verbose = 1,
                         validation_split = 0.8)

    #学習したモデルで、予測していく。                            
    b = hako[i][0].predict(np.vstack((otanoshimi_DataFrame.values,x_2_test)))[0]
　　#最も高い確率で予想される数値をそれぞれ空のリストに追加する
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

