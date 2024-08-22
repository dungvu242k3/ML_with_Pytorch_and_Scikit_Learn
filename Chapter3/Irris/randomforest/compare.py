import numpy as np
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

iris = load_iris()
X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

forest_no_std = RandomForestClassifier(n_estimators=25, random_state=1, n_jobs=2)
forest_no_std.fit(X_train, y_train)

y_pred = forest_no_std.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'acc no std: {accuracy * 100:.2f}%')

scaler = StandardScaler()
X_train_std = scaler.fit_transform(X_train)
X_test_std = scaler.transform(X_test)

forest_std = RandomForestClassifier(n_estimators=25, random_state=1, n_jobs=2)
forest_std.fit(X_train_std, y_train)

y_pred = forest_std.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'acc std: {accuracy * 100:.2f}%')