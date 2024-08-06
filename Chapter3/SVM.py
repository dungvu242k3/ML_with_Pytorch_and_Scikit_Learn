import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.colors import ListedColormap
from sklearn import datasets
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

# Tải tập dữ liệu Iris
iris = datasets.load_iris()
X = iris.data
y = iris.target

# Chọn hai thuộc tính để dễ dàng trực quan hóa
X = X[:, [2, 3]]  # Petal length và Petal width

# Chia dữ liệu thành tập huấn luyện và tập kiểm tra
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Tiêu chuẩn hóa dữ liệu
sc = StandardScaler()
X_train_std = sc.fit_transform(X_train)
X_test_std = sc.transform(X_test)

# Kết hợp dữ liệu huấn luyện và kiểm tra để vẽ vùng quyết định
X_combined_std = np.vstack((X_train_std, X_test_std))
y_combined = np.hstack((y_train, y_test))

# Khởi tạo và huấn luyện mô hình SVM
svm = SVC(kernel='linear', C=1.0, random_state=1)
svm.fit(X_train_std, y_train)

# Định nghĩa hàm để vẽ các vùng quyết định
def plot_decision_regions(X, y, classifier, test_idx=None, resolution=0.02):
    # Đặt giới hạn cho các trục
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1

    # Tạo lưới các điểm để dự đoán các vùng quyết định
    xx, yy = np.meshgrid(np.arange(x_min, x_max, resolution),
                         np.arange(y_min, y_max, resolution))
    Z = classifier.predict(np.array([xx.ravel(), yy.ravel()]).T)
    Z = Z.reshape(xx.shape)

    # Vẽ các vùng quyết định
    plt.contourf(xx, yy, Z, alpha=0.3, cmap=ListedColormap(('red', 'green', 'blue')))
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolor='k', marker='o', cmap=ListedColormap(('red', 'green', 'blue')))

    if test_idx:
        X_test_std, y_test = X[test_idx, :], y[test_idx]
        plt.scatter(X_test_std[:, 0], X_test_std[:, 1], c=y_test, edgecolor='k', marker='x', cmap=ListedColormap(('red', 'green', 'blue')), s=100)

# Vẽ các vùng quyết định
plt.figure(figsize=(10, 6))
plot_decision_regions(X_combined_std, y_combined, classifier=svm, test_idx=range(105, 150))
plt.xlabel('Petal length [standardized]')
plt.ylabel('Petal width [standardized]')
plt.legend(loc='upper left')
plt.tight_layout()
plt.show()

# Đánh giá mô hình
y_pred = svm.predict(X_test_std)
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred, target_names=iris.target_names)
conf_matrix = confusion_matrix(y_test, y_pred)

print(f"Accuracy: {accuracy * 100:.2f}%")
print("Classification Report:")
print(report)

# Vẽ ma trận nhầm lẫn
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=iris.target_names, yticklabels=iris.target_names)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()
