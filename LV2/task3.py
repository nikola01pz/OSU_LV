import numpy as np
import matplotlib.pyplot as plt

img = plt.imread("lv2/road.jpg")
img = img[:,:,0].copy()

plt.figure()
plt.title("Normalna slika")
plt.imshow(img, cmap="gray")

brighter_img = img+100
brighter_img[brighter_img<100] = 255
plt.figure()
plt.title("Svjetlija slika")
plt.imshow(brighter_img, cmap="gray", alpha=1)

plt.figure()
plt.title("Zrcaljena slika")
plt.imshow(np.fliplr(img), cmap="gray")

quarter_img = img[:, int(img.shape[1]/4):int(img.shape[1]/2)]
plt.figure()
plt.title("Četvrtina slike")
plt.imshow(quarter_img, cmap="gray")

rotated_img = np.rot90(img, -1)
plt.figure()
plt.title("Rotirana slika udesno")
plt.imshow(rotated_img, cmap="gray")

plt.show()

