import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (accuracy_score, confusion_matrix,
                             mean_squared_error, r2_score)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

iris = datasets.load_iris()
X = iris.data[:, :2]  
y = iris.target

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.3, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)


y_pred_class = np.round(y_pred).astype(int)
y_pred_class = np.clip(y_pred_class, 0, len(np.unique(y)) - 1)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred_class)
conf_matrix = confusion_matrix(y_test, y_pred_class)
print(f'Accuracy: {accuracy * 100:.2f}%')
print('Confusion Matrix:')
print(conf_matrix)
print(f'Mean Squared Error: {mse:.2f}')
print(f'R^2 Score: {r2:.2f}')

plt.figure(figsize=(12, 6))

plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=plt.get_cmap('jet', 3), edgecolor='k', s=50, label='train')

plt.scatter(X_test[:, 0], X_test[:, 1], c = "white", marker='o', edgecolor='k', s=100, cmap=plt.get_cmap('jet', 3), label='test',alpha =0.3)

x_range = np.linspace(X[:, 0].min(), X[:, 0].max(), 100)

y_range = (-model.coef_[0] * x_range - model.intercept_) / model.coef_[1]

plt.plot(x_range, y_range, color='black', lw=2)

plt.xlabel('Sepal Length')
plt.ylabel('Sepal Width')
plt.title('Linear Regression Model on Iris Dataset')
plt.show()
