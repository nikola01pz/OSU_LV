import pandas as pd
import numpy as np
from sklearn import datasets
from tensorflow import keras
from sklearn.preprocessing import OneHotEncoder

# ucitavanje dataseta
array = np.loadtxt("filename", delimiter=",", skiprows=9)

data = pd.read_csv("filename")

# brisanje odredenog stupca
array = array[array[:,5]!=0.0] # brise sve vrijednosti == 0.0

# array to df
data_df = pd.DataFrame(array)

# iris dataset
iris = datasets.load_iris()

df = pd.DataFrame(data= np.c_[iris["data"], iris["target"]],
                  columns=iris["feature_names"]+["target"])

df["species"] = pd.Categorical.from_codes(iris.target, iris.target_names)
print(df)

# iris dataset v2
iris = datasets.load_iris()
data_df = pd.DataFrame(data=iris.data)
data_df["species"] = pd.Categorical.from_codes(iris.target, iris.target_names)
data_df_versicolor = data_df[data_df["species"]=="versicolor"]
data_df_versicolor = data_df_versicolor.reset_index(drop = True)

# lv8 loadanje iz kerasa
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# brisanje vrijednosti koje nedostaju
data = data.dropna()

# category -> numerical ili OneHotEncoder
data_df.loc[data_df.Sex=="male", "Sex"]=0
data_df.loc[data_df.Embarked=="S", "Embarked"]=0

ohe = OneHotEncoder()
X_encoded_sex = ohe.fit_transform(data[["Sex"]]).toarray()
X_encoded_embarked = ohe.fit_transform(data[["Embarked"]]).toarray()

