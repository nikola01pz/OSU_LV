import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline, make_pipeline

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
                    label=cl)


# load data
data = pd.read_csv("lv6/Social_Network_Ads.csv")
print(data.info())

data.hist()
plt.show()

# dataframe to numpy
X = data[["Age","EstimatedSalary"]].to_numpy()
y = data["Purchased"].to_numpy()

# divide data 80%-20% (train, test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, stratify=y, random_state = 10)

# scale input values
sc = StandardScaler()
X_train_n = sc.fit_transform(X_train)
X_test_n = sc.transform((X_test))

# logistic regression model
logisticRegression_model = LogisticRegression(penalty=None) 
logisticRegression_model.fit(X_train_n, y_train)

# evaluation
y_train_p = logisticRegression_model.predict(X_train_n)
y_test_p = logisticRegression_model.predict(X_test_n)

print("Logistic Regression: ")
print("train accuracy: " + "{:0.3f}".format((accuracy_score(y_train, y_train_p))))
print("test accuracy: " + "{:0.3f}".format((accuracy_score(y_test, y_test_p))))

# decision border using logistic regresion
plot_decision_regions(X_train_n, y_train, classifier=logisticRegression_model)
plt.xlabel('x_1')
plt.ylabel('x_2')
plt.legend(loc='upper left')
plt.title("logReg accuracy: " + "{:0.3f}".format((accuracy_score(y_train, y_train_p))))
plt.tight_layout()
plt.show()

# KNN
# task1
KNN_model = KNeighborsClassifier ( n_neighbors = 7 )
KNN_model.fit( X_train_n, y_train )

y_train_p_KNN = KNN_model.predict(X_train_n)
y_test_p_KNN = KNN_model.predict( X_test_n )

plot_decision_regions(X_train_n, y_train, classifier=KNN_model)
plt.xlabel('x_1')
plt.ylabel('x_2')
plt.legend(loc='upper left')
plt.title("KNN accuracy: " + "{:0.3f}".format((accuracy_score(y_train, y_train_p_KNN))))
plt.tight_layout()
plt.show()

print("\nKNN:")
print("train accuracy: " + "{:0.3f}".format((accuracy_score(y_train, y_train_p_KNN))))
print("test accuracy: " + "{:0.3f}".format((accuracy_score(y_test, y_test_p_KNN))))
# border using KNN isn't linear compared to linear regression
# when K=1, it is overfitting
# when K=100, it is underfitting

# task2
KNN_model2 = KNeighborsClassifier()
param_grid = {"n_neighbors": np.arange(1, 100)}
# cv parameter specifies the number of folds to be used in cross-validation
knn_gs = GridSearchCV(KNN_model2, param_grid, cv = 5 )
knn_gs.fit(X_train_n, y_train )
print ("Best KNN parametars are:",knn_gs.best_params_)

# SVM
# task3
SVM_model = svm.SVC(kernel ="rbf", gamma = 1, C=10)
SVM_model.fit(X_train_n , y_train )
y_train_p_SVM = SVM_model.predict(X_train_n)
y_test_p_SVM = SVM_model.predict( X_test_n)

plot_decision_regions(X_train_n, y_train, classifier=SVM_model)
plt.xlabel('x_1')
plt.ylabel('x_2')
plt.legend(loc='upper left')
plt.title("SVM accuracy: " + "{:0.3f}".format((accuracy_score(y_train, y_train_p_SVM))))
plt.show()

print("\nSVM:")
print("train accuracy: " + "{:0.3f}".format((accuracy_score(y_train, y_train_p_SVM))))
print("test accuracy: " + "{:0.3f}".format((accuracy_score(y_test, y_test_p_SVM))))

# bigger C -> bigger penalization for wrong classification

# task4
SVM_model2 = svm.SVC(kernel ="rbf", gamma = 1, C=0.1 )
param_grid = {"C": [10 , 100 , 100 ], "gamma": [10 , 1 , 0.1 , 0.01 ]}
svm_gscv = GridSearchCV( SVM_model2 , param_grid , cv =5 , scoring ="accuracy", n_jobs = -1 )
svm_gscv.fit( X_train_n , y_train )
print ("Best SVM parametars are:",svm_gscv.best_params_)
print (svm_gscv.best_score_)