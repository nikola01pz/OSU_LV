#import bibilioteka
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

##################################################
#1. zadatak
##################################################

array = np.loadtxt("exams/exam1205/winequality-red.csv", delimiter=";", skiprows=1)
df = pd.DataFrame(array)
#print(df.info())

#a)
print(f"Mjerenje je izvršeno na {len(df)} vina")

#b)
quality = array[:,11]
alchocol = array[:,10]
plt.figure()
plt.bar(quality, alchocol, color="red")
plt.title("Ovisnost kvalitete vina o alkoholnoj razini")
plt.xlabel("wine quality")
plt.ylabel("wine alchocol")
plt.show()
# veća razina alkohola ne znači kvalitetnije vino, po grafu vidimo da je optimalna jakost vina 14, te da kvaliteta vina onda ovisi o drugim parametrima

#c)
lowQuality = quality[quality < 5]
print("Broj vina sa kvalitetom manjom od 5:",len(lowQuality))

highQuality = quality[quality >= 5]
print("Broj vina sa kvalitetom većom ili jednakom 5:",len(highQuality))

#d)
print("\n Korelacija svih veličina numeričkih veličina u datasetu:")
print(df.corr())
# Korelacija prikazuje ovisnost među vrijednosti, logično je da kad veličina korelira sama sa sobom da je 1, što je veći broj, tj. bliži broju 1, to nam govori stupanj ovisnosti veličina, suprotno tome, ako stupanj ide u -, tj prema -1 znači da veličine ne ovise međusobno
