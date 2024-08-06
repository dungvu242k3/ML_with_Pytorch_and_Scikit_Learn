import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets
from sklearn.linear_model import LinearRegression
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

plt.figure(figsize=(12, 6))

scatter = plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=plt.get_cmap('jet', 3), edgecolor='k', s=50)
plt.colorbar(scatter, ticks=[0, 1, 2], label='Species')

x_range = np.linspace(X[:, 0].min(), X[:, 0].max(), 100)
y_range = (-model.coef_[0] * x_range - model.intercept_) / model.coef_[1]
plt.plot(x_range, y_range, color='black', lw=2)

plt.xlabel('Sepal Length')
plt.ylabel('Sepal Width')
plt.title('Linear Regression Model on Iris Dataset')
plt.show()
