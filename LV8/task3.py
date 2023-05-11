import numpy as np
from tensorflow import keras
from PIL import Image
from matplotlib import pyplot as plt

model = keras.models.load_model('LV8/FCN/model.keras')
img = Image.open('LV8/test.png').convert('L')

img_array = np.array(img, dtype='float32') / 255
print(img_array.shape)

image_s = img_array.reshape(1, 28*28)
print(image_s.shape)

predictions = model.predict(image_s)
print(np.argmax(predictions))
