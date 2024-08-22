import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from sklearn import datasets
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

# Tải dữ liệu Iris và chuẩn bị
iris = datasets.load_iris()
X = iris.data  # Sử dụng tất cả các đặc trưng
y = iris.target

# Chia dữ liệu thành tập huấn luyện và tập kiểm tra
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=23)

# Tiền xử lý dữ liệu
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Sử dụng PCA để giảm số lượng đặc trưng và tăng hiệu suất mô hình
pca = PCA(n_components=2)  # Giảm xuống còn 2 đặc trưng chính
X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)

# Thiết lập GridSearch với tham số mở rộng
param_grid = {
    'hidden_layer_sizes': [(10,), (20,), (30,), (10, 10), (20, 10), (30, 15), (40, 20)],
    'alpha': [0.0001, 0.001, 0.01],
    'learning_rate_init': [0.001, 0.01, 0.1],
    'max_iter': [1000, 2000, 3000],
    'early_stopping': [True],  # Thêm early stopping
    'validation_fraction': [0.1]  # Thêm validation fraction
}

mlp = MLPClassifier(activation='relu', solver='adam', random_state=23)
grid = GridSearchCV(estimator=mlp, param_grid=param_grid, cv=5, verbose=2, n_jobs=-1)
grid_result = grid.fit(X_train, y_train)

# In kết quả tốt nhất
print(f"Best: {grid_result.best_score_} using {grid_result.best_params_}")

# Liệt kê tất cả các điểm số
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
for mean, std, params in zip(means, stds, grid_result.cv_results_['params']):
    print(f"{mean} (+/- {std}) for {params}")

# Đánh giá mô hình tốt nhất
best_model = grid_result.best_estimator_
y_test_pred = best_model.predict(X_test)
accuracy_score_test = accuracy_score(y_test, y_test_pred)
print(f'Accuracy_test: {accuracy_score_test * 100:.2f}%')

# Ma trận nhầm lẫn
conf_matrix = confusion_matrix(y_test, y_test_pred)
print('Confusion matrix:')
print(conf_matrix)

# Vẽ các đường biên quyết định
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

# Sử dụng mô hình tốt nhất từ GridSearch để vẽ đường biên quyết định
plot_decision_boundaries(X_train, y_train, X_test, y_test, best_model, "Optimized MLP with GridSearch")
