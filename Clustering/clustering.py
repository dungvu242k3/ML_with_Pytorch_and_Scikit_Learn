import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import fetch_openml
from sklearn.metrics import adjusted_rand_score

mnist = fetch_openml('mnist_784', version=1, as_frame=False)

X, y = mnist['data'], mnist['target']

y = y.astype(int)

X = X / 255.0

kmeans = KMeans(n_clusters=10, init='k-means++', max_iter=300, n_init=10, random_state=0)
y_kmeans = kmeans.fit_predict(X)

ari_score = adjusted_rand_score(y, y_kmeans)
print(f'Adjusted Rand Index (ARI): {ari_score:.4f}')

centroids = kmeans.cluster_centers_.reshape(10, 28, 28)

plt.figure(figsize=(8, 4))
for i in range(10):
    plt.subplot(2, 5, i + 1)
    plt.imshow(centroids[i], cmap='gray')
    plt.title(f'Centroid {i}')
    plt.axis('off')
plt.show()
