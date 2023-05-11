import numpy as np
import random
from tensorflow import keras
from keras import layers
from matplotlib import pyplot as plt
from keras.models import load_model

(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

x_train_s = x_train.astype("float32") / 255
x_test_s = x_test.astype("float32") / 255

x_train_s = x_train_s.reshape(x_train_s.shape[0], 28*28)
x_test_s = x_test_s.reshape(x_test_s.shape[0], -1)

y_train_s = keras.utils.to_categorical(y_train, 10)
y_test_s = keras.utils.to_categorical(y_test, 10)

model = keras.models.load_model("LV8/FCN/model.keras")

predictions = model.predict(x_test_s)
predictions = (predictions >= 0.5).astype(int)
score = model.evaluate(x_test_s, y_test_s, verbose = 0)

y_pred = np.argmax(predictions, axis=1)
y_real = np.argmax(y_test_s, axis=1)

x_wrong_predicted = x_test_s[y_pred != y_real]
y_wrong_predicted = y_pred[y_pred != y_real]
y_true = y_real[y_pred != y_real]

miss_index_list=[]
for i in range(x_test_s.shape[0]):
    if y_pred[i]!=y_true[i]:
        miss_index_list.append(i)

for _ in range(3):
    plt.figure()
    i=random.choice(miss_index_list)
    plt.imshow(x_test[i,:,:])
    plt.title(f'Predicted number: {y_pred[i]}       Real number: {y_true[i]}')
    plt.show()
