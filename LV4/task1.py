import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import sklearn.linear_model as lm
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score

data = pd.read_csv("lv4/data_C02_emission.csv")
data.info()

# axis 1 nam govori da se uklanjaju stupci (0 je za retke)
data = data.drop(["Make", "Model"], axis = 1)

data.info()

input_variables = ["Fuel Consumption City (L/100km)",
                   "Fuel Consumption Hwy (L/100km)",
                   "Fuel Consumption Comb (L/100km)",
                   "Fuel Consumption Comb (mpg)",
                   "Engine Size (L)",
                   "Cylinders"]

output_variable = ["CO2 Emissions (g/km)"]
X = data[input_variables].to_numpy()
y = data[output_variable].to_numpy()

# a) Podjela podataka na skupove za učenje i testiranje
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 1)

# b) Ovisnost emisije C02 plinova o jednoj numerickoj ulaznoj velicini
plot = plt.scatter(X_train[:,0], y_train, color = "blue")
print("Broj train podataka:",len(plot.get_offsets()))
plot = plt.scatter(X_test[:,0], y_test, color = "red")
print("Broj test podataka:",len(plot.get_offsets()))
plt.show()

# c) Skaliranje skupova podataka za ucenje i testiranje
sc = MinMaxScaler()
X_train_n = sc.fit_transform(X_train)
X_test_n = sc.transform(X_test)

plt.hist(X_train[:,0])
plt.show()
plt.hist(X_train_n[:,0])
plt.show()
 
# d)
linearModel = lm.LinearRegression()
linearModel.fit(X_train_n, y_train)
print(linearModel.intercept_)
print(linearModel.coef_)

# e)
y_test_p = linearModel.predict(X_test_n)
plt.figure()
plt.scatter(y_test, y_test_p)
plt.title("e)")
plt.show()

# f)
r2 = r2_score(y_test,y_test_p)
print(r2)

MAE = mean_absolute_error(y_test, y_test_p)
print(MAE)

# g) promjenom (smanjivanjem broja ulaznih veličina), mean se povećava a r2 se smanjuje

