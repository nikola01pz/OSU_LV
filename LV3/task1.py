import pandas as pd
import numpy as np

data = pd.read_csv("lv3/data_C02_emission.csv")
print("a) Datoteka sadrzi", len(data), "elemenata")

print("\na) Osnovne informacije:")
data.info()

print("\na) Izostale vrijednosti po kategoriji:")
print(data.isnull().sum())
print("\na) Duplicirane vrijednosti po kategoriji:")
print(data.duplicated().sum())

print("\na) Kategoričke veličine konvertirane u tip category:")
columns = ['Make', 'Model', 'Vehicle Class', 'Transmission', 'Fuel Type']
data[columns] = data[columns].astype('category')
data.info()

data.sort_values(by=["Fuel Consumption City (L/100km)"],
                 ascending=False, inplace=True)
print("\nb) Automobili sa najvećom gradskom potrošnjom:")
print(data[["Make", "Model", "Fuel Consumption City (L/100km)"]].head(3))
print("\nb) Automobili sa najmanjom gradskom potrošnjom:")
print(data[["Make", "Model", "Fuel Consumption City (L/100km)"]].tail(3))

new_data = data[(data["Cylinders"] > 2.5) & (data["Engine Size (L)"] < 3.5)]
print("\nc) Prosječna CO2 emisija za vozila veličine motora između 2.5 i 3.5L:",
      round(new_data["CO2 Emissions (g/km)"].mean(), 2))

audi_data = data[(data["Make"] == "Audi")]
print("\nd) Broj mjerenja koji se odnosi na vozila proizvođača Audi: ",
      len(audi_data))
audi_data_4cyl = audi_data[(audi_data["Cylinders"] == 4)]
print("\nd) Prosječna CO2 emisija za Audi vozila sa 4 cilindra:",
      round(audi_data_4cyl["CO2 Emissions (g/km)"].mean(), 2))

cars_4cyl = data[(data["Cylinders"] == 4)]
cars_5cyl = data[(data["Cylinders"] == 5)]
cars_6cyl = data[(data["Cylinders"] == 6)]
cars_8cyl = data[(data["Cylinders"] == 8)]
cars_10cyl = data[(data["Cylinders"] == 10)]
cars_12cyl = data[(data["Cylinders"] == 12)]
cars_16cyl = data[(data["Cylinders"] == 16)]

print("\ne)")
print("Broj vozila sa 4 cilindra:", len(cars_4cyl), "; prosječna emisija co2:",
      round(cars_4cyl["CO2 Emissions (g/km)"].mean(), 2))
print("Broj vozila sa 5 cilindra:", len(cars_5cyl), "; prosječna emisija co2:",
      round(cars_5cyl["CO2 Emissions (g/km)"].mean(), 2))
print("Broj vozila sa 6 cilindra:", len(cars_6cyl), "; prosječna emisija co2:",
      round(cars_6cyl["CO2 Emissions (g/km)"].mean(), 2))
print("Broj vozila sa 8 cilindra:", len(cars_8cyl), "; prosječna emisija co2:",
      round(cars_8cyl["CO2 Emissions (g/km)"].mean(), 2))
print("Broj vozila sa 10 cilindra:", len(cars_10cyl), "; prosječna emisija co2:",
      round(cars_10cyl["CO2 Emissions (g/km)"].mean(), 2))
print("Broj vozila sa 12 cilindra:", len(cars_12cyl), "; prosječna emisija co2:",
      round(cars_12cyl["CO2 Emissions (g/km)"].mean(), 2))
print("Broj vozila sa 16 cilindra:", len(cars_16cyl), "; prosječna emisija co2:",
      round(cars_16cyl["CO2 Emissions (g/km)"].mean(), 2))

diesel_cars = data[(data["Fuel Type"] == "D")]
print("\nf) Prosječna gradska potrošnja vozila koja koriste dizel:",
      round(diesel_cars["Fuel Consumption City (L/100km)"].mean(), 8))
print("Medijalna vrijednost prosječne gradske potrošnje vozila koja koriste dizel:",
      round(diesel_cars["Fuel Consumption City (L/100km)"].median(), 8))
regular_gas_cars = data[(data["Fuel Type"] == "X")]

print("f) Prosječna gradska potrošnja vozila koja koriste regularni benzin:",
      round(regular_gas_cars["Fuel Consumption City (L/100km)"].mean(), 8))
print("Medijalna vrijednost prosječne gradske potrošnje vozila koja koriste regularni benzin:",
      round(regular_gas_cars["Fuel Consumption City (L/100km)"].median(), 8))

cars = data[(data["Cylinders"] == 4) & (data["Fuel Type"] == "D")]
cars.sort_values(by=["Fuel Consumption City (L/100km)"], ascending=False, inplace=True)
print("\ng) Vozilo s 4 cilindra koje koristi dizelsko gorivo i ima najveću gradsku potrošnju:")
print(cars[["Make", "Model"]].head(1))

manual_cars = data[data["Transmission"].str.startswith("M")]
print("\nh) Broj automobila sa ručnim mjenjačem:", len(manual_cars))

print("\ni) Korelacija između numeričkih veličina:")
print(data.corr(numeric_only=True))