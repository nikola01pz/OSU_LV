import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

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

data_df = pd.read_csv("exams/exam1005/titanic.csv")
# print(data_df.info())
print(len(data_df))
data_df = data_df.dropna()
print(len(data_df))
data_df = data_df.drop_duplicates()
print(len(data_df))
data_df=data_df.reset_index(drop=True)

data_df.loc[data_df.Sex=="male", "Sex"]=0
data_df.loc[data_df.Sex=="female", "Sex"]=1

data_df.loc[data_df.Embarked=="S", "Embarked"]=0
data_df.loc[data_df.Embarked=="C", "Embarked"]=1
data_df.loc[data_df.Embarked=="Q", "Embarked"]=2

data_df["Sex"] = data_df["Sex"].astype(int)
data_df["Embarked"] = data_df["Embarked"].astype(int)

X = data_df[["Pclass","Sex","Fare","Embarked"]].copy().to_numpy()
y = data_df["Survived"].copy().to_numpy()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.4, random_state = 5)

sc = StandardScaler()
X_train_n = sc.fit_transform(X_train)
X_test_n = sc.transform(X_test)

# a)
KNN_model = KNeighborsClassifier ( n_neighbors = 5 )
KNN_model.fit( X_train_n, y_train )

y_train_p_KNN = KNN_model.predict(X_train_n)
y_test_p_KNN = KNN_model.predict( X_test_n )

plot_decision_regions(X_train_n, y_train, classifier=KNN_model)
plt.xlabel('x_1')
plt.ylabel('x_2')
plt.legend(loc='upper left')
plt.title("Tocnost, KNN : " +"{:0.3f}".format((accuracy_score(y_train, y_train_p_KNN))))
plt.tight_layout()
plt.show()

# b)
print("accuracy_score for k=5:", accuracy_score(y_train, y_train_p_KNN))

# c)
k_range = list(range(1, 31))
param_grid = dict(n_neighbors=k_range)
knn_gscv = GridSearchCV(KNN_model, param_grid, cv=5, scoring='accuracy', n_jobs =-1)

knn_gscv.fit(X_train_n, y_train)

print("\nbest params for train: ", knn_gscv.best_params_ )
print("best score for train: ", knn_gscv.best_score_ )
knn_gscv.fit(X_test_n, y_test)

print("best params for test: ", knn_gscv.best_params_ )
print("best score for test: ", knn_gscv.best_score_ )

# d) 
KNN_model2 = KNeighborsClassifier ( n_neighbors = 5 )
KNN_model2.fit( X_train_n, y_train )

y_train_p_KNN2 = KNN_model2.predict(X_train_n)
y_test_p_KNN2 = KNN_model2.predict( X_test_n )
print("\naccuracy_score for k=13:", accuracy_score(y_train, y_train_p_KNN2))