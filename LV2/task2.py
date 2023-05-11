import numpy as np
import matplotlib.pyplot as plt

data = np.loadtxt("lv2/data.csv", delimiter=",", dtype=float, skiprows=1)

# transponira se zbog dodjeljivanja vrijednosti
gender, height, weight = data.T
rows, columns = np.shape(data)

plt.scatter(height, weight)
# kreni od prvog podatka, idi do zadnjeg sa korakom 50
plt.scatter(height[0:-1:50], weight[0:-1:50])
plt.xlabel("Height")
plt.ylabel("Weight")
plt.show()

print("Mjerenja su izvršena na",rows,"ljudi")
print("\nMinimalna visina", np.min(height),"cm")
print("Maksimalna visina", np.max(height),"cm")
print("Prosječna visina", np.mean(height),"cm")

# sa : uzimamo sve podatke, a 0 predstavlja index stupca
# nastaje novo 1D polje sa True i False vrijednostima
male = (data[:,0] == 1)

# svi retci u stupcu indexa 1 kojima odgovara True iz "male"
print("\nMinimalna visina muškarca:",data[male,1].min(),"cm")
print("Maksimalna visina muškarca:",data[male,1].max(),"cm")
print("Prosječna visina muškarca:",data[male,1].mean(),"cm")
