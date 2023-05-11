import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import sklearn.linear_model as lm
from sklearn.metrics import max_error
from sklearn.metrics import r2_score
from sklearn.preprocessing import OneHotEncoder

data = pd.read_csv("lv4/data_C02_emission.csv")

full_data = data
data = data.drop(["Make", "Model"], axis = 1)

input_variables = ["Fuel Consumption City (L/100km)",
                   "Fuel Consumption Hwy (L/100km)",
                   "Fuel Consumption Comb (L/100km)",
                   "Fuel Consumption Comb (mpg)",
                   "Engine Size (L)",
                   "Cylinders",
                   "Fuel Type"]

ohe = OneHotEncoder()
# konvertiranje kategoričke veličine 
X_encoded = ohe.fit_transform(data[["Fuel Type"]]).toarray()
data["Fuel Type"] = X_encoded

output_variable = ["CO2 Emissions (g/km)"]
X = data[input_variables].to_numpy()
y = data[output_variable].to_numpy()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 1)

linearModel = lm.LinearRegression()
linearModel.fit(X_train, y_train)

y_test_p = linearModel.predict(X_test)

r2 = r2_score(y_test,y_test_p)
print(r2)

MA = max_error(y_test, y_test_p)
print("Maksimalna pogreška iznosi:", MA)

error = abs(y_test_p - y_test)
id = np.argmax(error)
car = full_data.iloc[id]
print("Model je:", car["Make"], car["Model"])

