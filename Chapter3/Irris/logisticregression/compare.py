import numpy as np
from sklearn import datasets
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

iris = datasets.load_iris()
X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=23)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

logreg_scaled = LogisticRegression(max_iter=1000, random_state=23)
logreg_scaled.fit(X_train_scaled, y_train)

y_test_pred_scaled = logreg_scaled.predict(X_test_scaled)
accuracy_scaled = accuracy_score(y_test, y_test_pred_scaled)
print(f'Accuracy with StandardScaler: {accuracy_scaled * 100:.2f}%')

logreg_no_scaler = LogisticRegression(max_iter=1000, random_state=23)
logreg_no_scaler.fit(X_train, y_train)

y_test_pred_no_scaler = logreg_no_scaler.predict(X_test)
accuracy_no_scaler = accuracy_score(y_test, y_test_pred_no_scaler)
print(f'Accuracy without StandardScaler: {accuracy_no_scaler * 100:.2f}%')
