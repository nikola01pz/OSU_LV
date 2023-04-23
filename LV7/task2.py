import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as Image
from sklearn.cluster import KMeans

# ucitaj sliku
img = Image.imread("lv7/imgs/test_1.jpg")

# prikazi originalnu sliku
plt.figure()
plt.title("Originalna slika")
plt.imshow(img)
plt.tight_layout()
plt.show()

# pretvori vrijednosti elemenata slike u raspon 0 do 1
img = img.astype(np.float64) / 255

# transformiraj sliku u 2D numpy polje (jedan red su RGB komponente elementa slike)
w,h,d = img.shape
img_array = np.reshape(img, (w*h, d))

# rezultatna slika
img_array_aprox = img_array.copy()

print("Original image color count: ", len(np.unique(img_array, axis=0)))

km = KMeans(n_clusters=4, init="random", n_init=5, random_state=0)
km.fit(img_array_aprox)
labels = km.predict(img_array_aprox)

centroids = km.cluster_centers_ * 255.0
colors = centroids[labels.astype(int)]

image = colors.reshape((w, h, d)).astype(int)

plt.figure()
plt.imshow(image)
plt.show()

plt.figure()
for i in range(1, 10):
    km = KMeans(n_clusters=i, init="random", n_init=5, random_state=0)
    km.fit(img_array_aprox)
    plt.plot(i, km.inertia_, ".-r", linewidth=2)
    km.fit(img_array)
    plt.title("The Elbow Method")
    plt.xlabel("K")
    plt.ylabel("J")

plt.show()

clusters = 4
for i in range(clusters):
    bit_values = labels==[i]
    binary_img = np.reshape(bit_values, (img.shape[0:2]))
    binary_img = binary_img*1
    x=int(i/2)
    y=i%2
    plt.imshow(binary_img)
    plt.show()
