import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from sklearn import datasets
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

iris = datasets.load_iris()
X = iris.data[:,:2]  
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=23)

mlp = MLPClassifier(hidden_layer_sizes=(50,), activation='relu', solver='adam', alpha=0.0001, max_iter=200)
mlp.fit(X_train, y_train)

y_train_pred = mlp.predict(X_train)
accuracy_score_train = mlp.score(X_train, y_train)
print(f'Accuracy_train: {accuracy_score_train * 100:.2f}%')

y_test_pred = mlp.predict(X_test)
accuracy_score_test = mlp.score(X_test, y_test)
print(f'Accuracy_test: {accuracy_score_test * 100:.2f}%')

conf_matrix = confusion_matrix(y_test,y_test_pred)
print('confushion matrix :')
print(conf_matrix)

def plot_decision_boundaries(X_train, y_train, X_test, y_test, model, title):
    x_min, x_max = X_train[:, 0].min() - 1, X_train[:, 0].max() + 1
    y_min, y_max = X_train[:, 1].min() - 1, X_train[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01),
                         np.arange(y_min, y_max, 0.01))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, alpha=0.8, cmap=ListedColormap(('red', 'lightgreen', 'blue')))
    plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, edgecolors='k', marker='o', s=20, cmap=ListedColormap(('red', 'lightgreen', 'blue')), label='Train set')
    plt.scatter(X_test[:, 0], X_test[:, 1], c='none', edgecolor='black', alpha=0.3, linewidth=2, marker='o', s=100, label='Test set')
    plt.title(title)
    plt.xlabel('sepal length')
    plt.ylabel('sepal width')
    plt.legend()
    plt.show()

plot_decision_boundaries(X_train, y_train, X_test, y_test, mlp,"MLP")
