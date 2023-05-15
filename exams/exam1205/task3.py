#import bibilioteka
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from tensorflow import keras
from keras.models import load_model
from sklearn.model_selection import train_test_split

##################################################
#3. zadatak
##################################################

array = np.loadtxt("exams/exam1205/winequality-red.csv", delimiter=";", skiprows=1)
df = pd.DataFrame(array, columns=["fixed_acidity","volatile_acidity","citric_acid","residual_sugar","chlorides","free_sulfur_dioxide","total_sulfur_dioxide","density","pH","sulphates","alcohol","quality"])

df.loc[df.quality < 5, "quality"] = 0
df.loc[df.quality >= 5, "quality"] = 1

X = df.drop(columns=["quality"]).to_numpy()
y = df["quality"].copy().to_numpy()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 5)

#a)
model = keras.Sequential()
model.add(keras.layers.Input(shape=(11,)))
model.add(keras.layers.Dense(units=20, activation="relu"))
model.add(keras.layers.Dense(units=12, activation="relu"))
model.add(keras.layers.Dense(units=4, activation="relu"))
model.add(keras.layers.Dense(units=1, activation="sigmoid"))
model.summary()

#b)
model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

#c)
history = model.fit(X_train, y_train, batch_size=50, epochs=2, validation_split=0.1)

#d)
model.save("exams/exam1205/Model/")
del model

model = load_model("exams/exam1205/Model/")

#e)
score = model.evaluate(X_test, y_test, verbose=0)
print("Test loss: ", score[0])
print("Test accuracy: ", score[1])

#f)
predictions = model.predict(X_test)
predictions = np.around(predictions).astype(np.int32)

cm = confusion_matrix(y_test, predictions)
display = ConfusionMatrixDisplay(confusion_matrix=cm)
display.plot()
plt.show()

# Model prikazuje vrlo dobre rezultate, prikazuje točnost od 97,5% što je u granicama dobro izmodeliranih sustava. Jako mali broj rezultata je netočan.
