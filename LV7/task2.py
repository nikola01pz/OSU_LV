import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as Image
from sklearn.cluster import KMeans

for i in range(1,7):
    # ucitaj sliku
    img = Image.imread(f"lv7/imgs/test_{i}.jpg")

    # prikazi originalnu sliku
    plt.figure()
    plt.title("Originalna slika")
    plt.imshow(img)
    plt.tight_layout()
    plt.show()

    # pretvori vrijednosti elemenata slike u raspon 0 do 1
    img = img.astype(np.float64) / 255
    if i == 4:
        img = img[:,:,:3]

    # transformiraj sliku u 2D numpy polje (jedan red su RGB komponente elementa slike)
    w,h,d = img.shape
    img_array = np.reshape(img, (w*h, d))

    # rezultatna slika
    img_array_aprox = img_array.copy()

    print("Color count on original image:", len(np.unique(img_array_aprox, axis=0)))

    k = [2,3,4,5,7,10]
    sse = []

    for j in range(len(k)):

        img_array_aprox = img_array.copy()

        km = KMeans( n_clusters = k[j], init ='random', n_init =5 , random_state = 0 )

        km.fit(img_array_aprox)

        sse.append(km.inertia_)

        for z in np.unique(km.labels_):
            img_array_aprox[km.labels_==z,:] = km.cluster_centers_[z]        

        img2 = img_array_aprox.reshape(img.shape)

        plt.subplot(3,2,j+1)
        plt.title('k = ' + str(k[j]))
        plt.imshow(img2)

    plt.show()


    plt.plot(k, sse)
    plt.show()


    clusters = [3,3,3,3,4,5]

    img_array_aprox = img_array.copy()

    km = KMeans( n_clusters = clusters[i-1], init ='random', n_init =5 , random_state = 0 )

    km.fit(img_array_aprox)

    fig, axs = plt.subplots(nrows=1, ncols=clusters[i-1], figsize=(10,5))
    for l, ax in zip(np.unique(km.labels_), axs):
        layer_arr = np.zeros_like(km.labels_)
        layer_arr[km.labels_ == l] = 1
        layer_img = layer_arr.reshape(w, h)
        ax.imshow(layer_img, cmap='gray', vmin=0, vmax=1)
    plt.suptitle('the primary group is represented in white')
    plt.show()