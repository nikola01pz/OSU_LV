import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets

iris = datasets.load_iris()

df = pd.DataFrame(iris.data, columns=iris.feature_names)
df["target"] = iris.target
df["species"] = iris.target_names[iris.target]
print(df)

# a)
versicolor = df[df["species"]=="versicolor"]
virginica = df[df["species"]=="virginica"]

plt.figure()
plt.scatter(versicolor["petal width (cm)"], versicolor["sepal length (cm)"], color="blue")
plt.scatter(virginica["petal width (cm)"], virginica["sepal length (cm)"], color="red")
plt.title("Odnos duljina casica i latica od versicolora i virginice")
plt.xlabel("Duljina latice")
plt.ylabel("Duljina casice")
plt.legend(["versicolor plavo", "virginica crveno"])
plt.show()
# duljine casice i latice kod virginice su vece naspram versicolora

# b)
versicolorSepalWidthMean = versicolor["sepal width (cm)"].mean()
virginicaSepalWidthMean = virginica["sepal width (cm)"].mean()
setosa = df[df["species"]=="setosa"]
setosaSepalWidthMean = setosa["sepal width (cm)"].mean()
plt.figure()
dictionary = {"versicolor":versicolorSepalWidthMean, "virginica": virginicaSepalWidthMean, "setosa": setosaSepalWidthMean}
values = list(dictionary.values())
keys = list(dictionary.keys())

plt.bar(keys, values)
plt.title("Prosjek sirine casice")
plt.xlabel("Nazivi")
plt.ylabel("Sirina latice")
plt.show()
# setosa ima najvecu sirinu casice

# c)
vecaSirinaCasice = virginica[virginica["sepal width (cm)"]>virginicaSepalWidthMean]
print(f"{len(vecaSirinaCasice)} jedinki ima vecu sirinu casice od prosjecne")

