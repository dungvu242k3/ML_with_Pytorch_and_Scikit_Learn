import numpy as np
from sklearn import datasets
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

iris = datasets.load_iris()
X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1, stratify=y)

scaler = StandardScaler()
X_train_std = scaler.fit_transform(X_train)
X_test_std = scaler.transform(X_test)

knn_std = KNeighborsClassifier(n_neighbors=5, p=2, metric='minkowski')
knn_std.fit(X_train_std, y_train)

y_pred_std = knn_std.predict(X_test_std)
accuracy = accuracy_score(y_test,y_pred_std)
print(f'Acc std: {accuracy * 100:.2f}%')

knn_no_std = KNeighborsClassifier(n_neighbors=5, p=2, metric='minkowski')
knn_no_std.fit(X_train,y_train)

y_pred_no_std = knn_std.predict(X_test)
accuracy = accuracy_score(y_test,y_pred_no_std)
print(f'Acc no std: {accuracy * 100:.2f}%')
