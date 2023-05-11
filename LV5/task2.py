import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import classification_report, accuracy_score

labels= {0:'Adelie', 1:'Chinstrap', 2:'Gentoo'}

def plot_decision_regions(X, y, classifier, resolution=0.02):
    plt.figure()
    # setup marker generator and color map
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])
    
    # plot the decision surface
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
    np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.3, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())
    
    # plot class examples
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0],
                    y=X[y == cl, 1],
                    alpha=0.8,
                    c=colors[idx],
                    marker=markers[idx],
                    edgecolor = 'w',
                    label=labels[cl])

# load data
df = pd.read_csv("lv5/penguins.csv")

# missing column values
print(df.isnull().sum())

# column sex has missing values, drop it
df = df.drop(columns=['sex'])

# delete rows with missing values
df.dropna(axis=0, inplace=True)

# category value - coding
df['species'].replace({'Adelie' : 0,
                        'Chinstrap' : 1,
                        'Gentoo': 2}, inplace = True)

print(df.info())

# output value: species
output_variable = ['species']

# input values: bill length, flipper_length
input_variables = ['bill_length_mm',
                    'flipper_length_mm']

X = df[input_variables].to_numpy()
y = df[output_variable].to_numpy()

# train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 123)

#a)
unique_train, unique_train_count = np.unique(y_train, return_counts=True)
unique_test, unique_test_count = np.unique(y_test, return_counts=True)

plt.bar(unique_train,unique_train_count, color="blue", label="Train data")
plt.bar(unique_test, unique_test_count, color="red", label="Test data")
plt.xlabel("Species")
plt.ylabel("Number")
plt.title("a)")
plt.legend()
plt.show()

#b)
logisticRegression_model = LogisticRegression()
logisticRegression_model.fit(X_train, y_train)

#c)
print("Coefficients:", logisticRegression_model.coef_)
print("Intersection point: ", logisticRegression_model.intercept_)

#d)
plot_decision_regions(X_test, y_test, classifier=logisticRegression_model)

#e)
y_test_p = logisticRegression_model.predict(X_test)
print("Accuracy: ", accuracy_score(y_test, y_test_p))

confusionMatrix = confusion_matrix(y_test, y_test_p)
print("Confusion Matrix:\n", confusionMatrix)
display = ConfusionMatrixDisplay(confusion_matrix(y_test, y_test_p))
display.plot()
plt.show()

print(classification_report(y_test, y_test_p))

#f)
input_variables = [ "bill_length_mm",
                    "flipper_length_mm",
                    "bill_depth_mm",
                    "body_mass_g" ]

X = df[input_variables].to_numpy()
# zelimo osigurati sa [:,0] da izlazna varijabla bude 1D niz
y = df[output_variable].to_numpy()[:,0]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 1)

logisticRegression_model2 = LogisticRegression()
logisticRegression_model2.fit(X_train, y_train)

y_pred = logisticRegression_model2.predict(X_test)
print(classification_report(y_test, y_pred))

# more input values => better classification_reports results are
