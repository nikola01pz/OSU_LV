import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


data = pd.read_csv("exams/exam1005/titanic.csv")
#array = np.loadtxt("exams/exam1005/titanic.csv", delimiter=",", skiprows=1)

# a)
print(f'Broj osoba: {len(data)}')

# b)
data_survived = data[data["Survived"] == 1]
print("Osobe koje su prezivjele potonuce:", len(data_survived))

# c)
male_survived = data_survived[data_survived["Sex"]=="male"]
female_survived = data_survived[data_survived["Sex"]=="female"]
male = data[data["Sex"]=="male"]
female = data[data["Sex"]=="female"]
print("Prezivjeli muskarci", len(male_survived))
print("Prezivjele zene", len(female_survived))
procentage_male_surv = len(male_survived)/len(male)
procentage_female_surv = len(female_survived)/len(female)
print("Postotak prezivjelih muskaraca", procentage_male_surv*100)
print("Postotak prezivjelih zena", procentage_female_surv*100)
prezivjeli = {procentage_male_surv, procentage_female_surv}

plt.figure()
dictionary = {"postotak muskaraca prezivjelih":procentage_male_surv, "postotak zena prezivjelih":procentage_female_surv}
values = list(dictionary.values())
keys = list(dictionary.keys())
plt.bar(keys, values)
plt.show()

# d)
print("Prosjecna dob prezivjelih muskaraca", male_survived["Age"].mean())
print("Prosjecna dob prezivjelih zena", female_survived["Age"].mean())

# e)
class1 = male_survived[male_survived["Pclass"]==1]
print("najmladji muskarac u mjesecima:", 12*class1["Age"].min())
class2 = male_survived[male_survived["Pclass"]==2]
print(class2["Age"].min())
class3 = male_survived[male_survived["Pclass"]==3]
print(class3["Age"].min())

