import numpy as np
import pandas as pd
from tensorflow import keras
from keras import layers
from keras.models import load_model
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split

array = np.loadtxt("exams/exam0805/pima-indians-diabetes.csv", skiprows=9, delimiter=",")

data_df = pd.DataFrame(array, columns=['num_pregnant', 'plasma', 'blood_pressure', 'triceps', 'insulin', 'BMI', 'diabetes_function', 'age', 'diabetes'])
X = data_df.drop(columns=["diabetes"]).to_numpy()
y = data_df["diabetes"].copy().to_numpy()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 5)

# a)
model = keras.Sequential()
model.add(layers.Input(shape = (8,) ))
model.add(keras.layers.Dense(12, activation = "relu" ))
model.add(keras.layers.Dense(8, activation = "relu" ))
model.add(keras.layers.Dense(1, activation = "sigmoid" ))
model.summary()

# b) loss moze biti i "categorical_crossentropy"
model.compile(loss = "binary_crossentropy", optimizer = "adam", metrics =["accuracy",])

# c)
history = model.fit(X_train, y_train, batch_size = 10, epochs = 1, validation_split = 0.1)

# d)
model.save("exams/exam0805/Model/")
del model

model = load_model("exams/exam0805/Model/")

# e)
score = model.evaluate(X_test, y_test, verbose = 0)
print("Test loss: ", score[0])
print("Test accuracy: ", score[1])

# drugi način
#for i in range(len(model.metrics_names)):
#    print(f'{model.metrics_names[i]} = {score[i]}')

# f)
predictions = model.predict(X_test)
predictions = np.around(predictions).astype(np.int32)

cm = confusion_matrix(y_test, predictions)
display = ConfusionMatrixDisplay(confusion_matrix=cm)
display.plot()
plt.show()