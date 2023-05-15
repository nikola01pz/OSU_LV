import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ucitavanje dataseta
array = np.loadtxt("exams/exam0805/pima-indians-diabetes.csv", delimiter=",", skiprows=9)

# a)
print(f"Broj mjerenja: {len(array)}")

# b)
array = array[array[:,5]!=0.0] # brise sve sa BMI === 0.0
data = pd.DataFrame(array)

print(data.isnull().sum())
print(data.duplicated().sum())
data = data.drop_duplicates()
data = data.dropna(axis = 0)
print(len(data))

# c)
#a,b,c,d,f,bmi,e,dob,g = array.T
#plt.scatter(dob, bmi)
dob = array[:,7]
bmi = array[:,5]
plt.scatter(dob, bmi)
plt.title("Ovisnost dobi o BMI")
plt.xlabel("Age(years)")
plt.ylabel("BMI(weight in kg/(height in m)^2)")
#plt.show()

# d)
print("Max BMI:", max(bmi))
print("Min BMI:", min(bmi))
print("Srednja vrijednost BMI:", data[5].mean())

# e)
withDiabetes = data[data[8] == 1]
print("Broj osoba sa dijabetesom:", len(withDiabetes))
print("Max BMI osobe sa dijabetesom:", withDiabetes[5].max())
print("Min BMI osobe sa dijabetesom:", withDiabetes[5].min())
print("Srednja vrijednost BMI osobe sa dijabetesom:", withDiabetes[5].mean())

withoutDiabetes = data[data[8] == 0]
print("Broj osoba bez dijabetesa:", len(withoutDiabetes))
print("Max BMI osobe bez dijabetesa:", withoutDiabetes[5].max())
print("Min BMI osobe bez dijabetesa:", withoutDiabetes[5].min())
print("Srednja vrijednost BMI osobe bez dijabetesa:", withoutDiabetes[5].mean())