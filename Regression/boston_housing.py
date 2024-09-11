import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer

# Tải dữ liệu từ file CSV
boston = pd.read_csv("C:/Users/dungv/Projects/ML_with_Pytorch_and_Scikit_Learn/Clustering/HousingData.csv")

# Kiểm tra dữ liệu có giá trị NaN hay không
print(boston.isna().sum())

# Sử dụng SimpleImputer để thay thế giá trị NaN bằng giá trị trung bình
imputer = SimpleImputer(strategy='mean')
X = imputer.fit_transform(boston.iloc[:, :-1].values)  # Các đặc trưng (features), loại bỏ cột nhãn
y = boston.iloc[:, -1].values   # Nhãn (target), cột cuối cùng

# Số cụm cần phân loại
n_clusters = 3

# Áp dụng K-means
kmeans = KMeans(n_clusters=n_clusters, init='k-means++', max_iter=300, n_init=10, random_state=0)
y_kmeans = kmeans.fit_predict(X)

# Vẽ biểu đồ phân cụm
# Để đơn giản hóa, chúng ta chỉ chọn 2 đặc trưng đầu tiên vì dữ liệu có nhiều chiều
plt.figure(figsize=(20, 17))
for i in range(n_clusters):
    plt.scatter(X[y_kmeans == i, 0], X[y_kmeans == i, 1], s=100, label=f'Cluster {i + 1}')

# Vẽ centroid của các cụm
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=200, c='red', label='Centroids', marker='X')

plt.xlabel('Feature 1')  # Thay thế bằng tên đặc trưng thực tế nếu có
plt.ylabel('Feature 2')  # Thay thế bằng tên đặc trưng thực tế nếu có
plt.title('K-means Clustering on Boston Housing Data')
plt.legend()
plt.show()
