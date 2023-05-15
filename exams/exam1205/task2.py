#import bibilioteka
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import sklearn.linear_model as lm
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error, mean_absolute_percentage_error

##################################################
#2. zadatak
##################################################

array = np.loadtxt("exams/exam1205/winequality-red.csv", delimiter=";", skiprows=1)
df = pd.DataFrame(array, columns=["fixed_acidity","volatile_acidity","citric_acid","residual_sugar","chlorides","free_sulfur_dioxide","total_sulfur_dioxide","density","pH","sulphates","alcohol","quality"])

df.loc[df.quality < 5, "quality"] = 0
df.loc[df.quality >= 5, "quality"] = 1

X = df.drop(columns=["quality"]).to_numpy()
y = df["quality"].copy().to_numpy()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 5)

sc = MinMaxScaler()
X_train_n = sc.fit_transform(X_train)
X_test_n = sc.transform(X_test)

#a)
linearModel  = lm.LinearRegression()
linearModel.fit(X_train_n, y_train)
print("intercept_: ", linearModel.intercept_)
print("coef_: ", linearModel.coef_)

#b)
y_test_p = linearModel.predict(X_test_n)
plt.figure()
plt.scatter(y_test, y_test_p)
plt.title("2. zadatak: b)")
plt.xlabel("y_test")
plt.ylabel("y_test_p")
plt.show()

#c)
RMSE = mean_squared_error(y_test, y_test_p, squared=False)
print("RMSE: ", RMSE)

MAE = mean_absolute_error(y_test, y_test_p)
print("MAE: ", MAE)

MAPE = mean_absolute_percentage_error(y_test, y_test_p)
print("MAPE: ", MAPE)

r2 = r2_score(y_test,y_test_p)
print("r2: ", r2)

# brojke odgovaraju podacima, nemaju nerealne rezultate nego kada razmislimo o datasetu i podacima, ove kalkulacije su u granicama normale






