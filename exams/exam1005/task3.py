import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from tensorflow import keras
from keras import layers
from keras.models import load_model
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

data = pd.read_csv("exams/exam1005/titanic.csv")
data.dropna(axis=0, inplace=True)
data.drop_duplicates(inplace=True)

print("types:\n", data.dtypes)
input_variables = [
    "Pclass",
    "Sex",
    "Fare",
    "Embarked",
]

ohe = OneHotEncoder()

X_encoded_sex = ohe.fit_transform(data[["Sex"]]).toarray()
X_encoded_embarked = ohe.fit_transform(data[["Embarked"]]).toarray()

data["Sex"] = X_encoded_sex
data["Embarked"] = X_encoded_embarked

output_variable = ["Survived"]

X = data[input_variables].to_numpy()
y = data[output_variable].to_numpy()

X_train , X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=1)

sc = StandardScaler()
X_train_n = sc.fit_transform(X_train)
X_test_n = sc.transform((X_test))

# a)
model = keras.Sequential()
model.add(layers.Input(shape=(4,)))
model.add(layers.Dense(units=16, activation="relu"))
model.add(layers.Dense(units=8, activation="relu"))
model.add(layers.Dense(units=4, activation="relu"))
model.add(layers.Dense(units=1, activation="sigmoid"))
model.summary()

# b)
model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy", ])

# c)
history = model.fit(X_train, y_train, batch_size=5, epochs=100, validation_split=0.1)

# d)
model.save("exams/exam1005/Model")

del model

model = load_model("exams/exam1005/Model/")

# e)
score = model.evaluate(X_test, y_test, verbose=0)
print("Test loss: ", score[0])
print("Test accuracy: ", score[1])

# f)
predictions = model.predict(X_test)
predictions = np.around(predictions).astype(np.int32)

cm = confusion_matrix(y_test, predictions)
display = ConfusionMatrixDisplay(confusion_matrix=cm)
display.plot()
plt.show()

