import matplotlib.pyplot as plt
import numpy as np
from mlxtend.plotting import plot_decision_regions
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

iris = load_iris()
X = iris.data[:, [2, 3]]
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

X_combined = np.vstack((X_train, X_test))
y_combined = np.hstack((y_train, y_test))

forest = RandomForestClassifier(n_estimators=25, random_state=1, n_jobs=2)
forest.fit(X_train, y_train)

plot_decision_regions(X_combined, y_combined, clf=forest)

plt.scatter(X_test[:, 0], X_test[:, 1], c='white',
            edgecolor='black', alpha=0.3,
            linewidth=3, marker='o',
            s=100, label='Test set')

plt.xlabel('Chiều dài cánh hoa')
plt.ylabel('Chiều rộng cánh hoa')
plt.legend(loc='upper left')
plt.title('Random Forest')
plt.show()

y_pred = forest.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Độ chính xác: {accuracy * 100:.2f}%')

conf_matrix = confusion_matrix(y_test, y_pred)
print('Ma trận nhầm lẫn:')
print(conf_matrix)

report = classification_report(y_test, y_pred)
print('Báo cáo phân loại:')
print(report)
