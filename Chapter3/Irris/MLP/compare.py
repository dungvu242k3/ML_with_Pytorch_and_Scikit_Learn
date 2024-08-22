import numpy as np
from sklearn import datasets
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

iris = datasets.load_iris()
X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

mlp_scaled = MLPClassifier(hidden_layer_sizes=(10,), activation='relu', solver='adam', max_iter=1000, random_state=42)
mlp_scaled.fit(X_train_scaled, y_train)

y_test_pred_scaled = mlp_scaled.predict(X_test_scaled)
accuracy_scaled = accuracy_score(y_test, y_test_pred_scaled)
print(f'Acc std: {accuracy_scaled * 100:.2f}%')

mlp_no_scaler = MLPClassifier(hidden_layer_sizes=(10,), activation='relu', solver='adam', max_iter=1000, random_state=42)
mlp_no_scaler.fit(X_train, y_train)

y_test_pred_no_scaler = mlp_no_scaler.predict(X_test)
accuracy_no_scaler = accuracy_score(y_test, y_test_pred_no_scaler)
print(f'Acc no std: {accuracy_no_scaler * 100:.2f}%')
