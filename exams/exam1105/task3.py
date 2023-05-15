import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from tensorflow import keras
from keras import layers
from keras.models import load_model
from keras.utils import to_categorical
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import  datasets

iris = datasets.load_iris()
X = iris.data
y = iris.target

X_train, X_test, y_train, y_test= train_test_split(X, y, test_size=0.25)
y_train = to_categorical(y_train, 3)
y_test = to_categorical(y_test, 3)

sc = StandardScaler()
X_train_n=sc.fit_transform(X_train)
X_test_n=sc.transform(X_test)

# a) 
model = keras.Sequential()
model.add(keras.layers.Input(shape=(4,)))
model.add(keras.layers.Dense(units=10, activation="relu"))
model.add ( keras.layers.Dropout (0.3) )
model.add(keras.layers.Dense(units=7, activation="relu"))
model.add (keras.layers.Dropout (0.3) )
model.add(keras.layers.Dense(units=5, activation="relu"))
model.add(keras.layers.Dense(units=3, activation="softmax"))
model.summary()

# b)
model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
# c)
history = model.fit(X_train, y_train, batch_size=7, epochs=500, validation_split=0.1)
# d)
model.save("exams/exam1105/Model/")

del model

model = load_model("exams/exam1105/Model/")

# e)
score = model.evaluate(X_test, y_test, verbose=0)
for i in range(len(model.metrics_names)):
    print(f'{model.metrics_names[i]} = {score[i]}')
# f)
predictions = model.predict(X_test)
predictions = np.around(predictions).astype(np.int32)

cm = confusion_matrix(y_test.argmax(axis=1), predictions.argmax(axis=1))
display = ConfusionMatrixDisplay(confusion_matrix=cm)
display.plot()
plt.show()
