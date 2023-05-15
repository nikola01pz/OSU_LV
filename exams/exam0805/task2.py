import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn . model_selection import train_test_split
from sklearn . linear_model import LogisticRegression
from sklearn . metrics import accuracy_score, classification_report, precision_score, recall_score
from sklearn . metrics import confusion_matrix , ConfusionMatrixDisplay

array = np.loadtxt("exams/exam0805/pima-indians-diabetes.csv", skiprows=9, delimiter=",")

data_df = pd.DataFrame(array, columns=['num_pregnant', 'plasma', 'blood_pressure', 'triceps', 'insulin', 'BMI', 'diabetes_function', 'age', 'diabetes'])
X = data_df.drop(columns=["diabetes"]).to_numpy()
y = data_df["diabetes"].copy().to_numpy()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 5)

# a)
logReg_model = LogisticRegression(max_iter=300)
logReg_model.fit(X_train, y_train)

# b)
y_test_p = logReg_model.predict(X_test)

# c)
cm = confusion_matrix(y_test, y_test_p)
print("Confusion matrix:\n", cm)
display = ConfusionMatrixDisplay(cm)
display.plot()
plt.title("c) Confusion matrix")
#plt.show()
# treba objasnit, pogledat u predlosku

# d)
print("classification_report", classification_report(y_test , y_test_p))
print("accuracy_score", accuracy_score(y_test , y_test_p))
print("precision_score", precision_score(y_test , y_test_p))
print("recall_score", recall_score(y_test , y_test_p))
# treba objasnit, pogledat u predlosku