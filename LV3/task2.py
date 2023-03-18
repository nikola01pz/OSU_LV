import matplotlib.pyplot as plt
import pandas as pd

data = pd.read_csv("lv3/data_C02_emission.csv")

plt.figure()
data["CO2 Emissions (g/km)"].plot( kind = "hist", bins = 150)
plt.title("a)")
plt.show()

data["Fuel Color"] = data["Fuel Type"].map(
   {
    "X" : "Yellow",
    "Z" : "Orange",
    "D" : "Pink",
    "E" : "Red",
    "N" : "Purple"
   }
)

data.plot.scatter(x="Fuel Consumption City (L/100km)", y="CO2 Emissions (g/km)", c="Fuel Color")
plt.title("b)")
plt.show()

data.boxplot(column=["Fuel Consumption Hwy (L/100km)"], by="Fuel Type")
plt.title("c)")
plt.show()

data.groupby("Fuel Type").size().plot(kind='bar')
plt.title("d)")
plt.show()

data.groupby("Cylinders")["CO2 Emissions (g/km)"].mean().plot(kind="bar")
plt.title("e)")
plt.show()