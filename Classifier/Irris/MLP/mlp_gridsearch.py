import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from sklearn import datasets
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

iris = datasets.load_iris()
X = iris.data[:,:2]
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=23)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)



param_grid = {
    'hidden_layer_sizes': [(50,), (50, 30), (100,)],
    'activation': ['tanh', 'relu'],
    'solver': ['sgd', 'adam'],
    'alpha': [0.0001, 0.05],
    'learning_rate': ['constant', 'adaptive'],
    'max_iter': [400, 800, 1000],
}


mlp = MLPClassifier(random_state=42)
grid = GridSearchCV(estimator=mlp, param_grid=param_grid, cv=5, scoring='accuracy')
grid_result = grid.fit(X_train, y_train)


print("Best parameters found: ", grid.best_params_)
print("Best accuracy score: ", grid.best_score_)

best_model = grid.best_estimator_
accuracy = best_model.score(X_test, y_test)
print("Test set accuracy: ", accuracy)

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
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend()
    plt.show()

plot_decision_boundaries(X_train, y_train, X_test, y_test, best_model, "Optimized MLP with GridSearch")
