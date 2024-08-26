import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (accuracy_score, confusion_matrix,
                             mean_squared_error, r2_score)
from sklearn.model_selection import train_test_split

iris = sns.load_dataset('iris')

X = iris['petal_length']
y = iris['petal_width']



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=23)
X_train = np.array(X_train).reshape(-1, 1)
X_test = np.array(X_test).reshape(-1, 1)

lr = LinearRegression()
lr.fit(X_train, y_train)

y_pred_train = lr.predict(X_train)
y_pred_test = lr.predict(X_test)

test_set = iris.iloc[y_test.index]
print(test_set[['species', 'petal_length', 'petal_width']])

species_unique = iris['species'].unique()
markers = ['o', 's', 'D']  
colors = ['red', 'green', 'blue'] 

for species, marker, color in zip(species_unique, markers, colors):
    subset = iris[iris['species'] == species]
    plt.scatter(subset['petal_length'], subset['petal_width'], 
                color=color, marker=marker, label=species, edgecolor='black')
plt.plot(X_test, y_pred_test, color="black", label='Regression Line')
plt.xlabel('Petal Length')
plt.ylabel('Petal Width')
plt.title('Petal Length vs Petal Width with Linear Regression Line')
plt.legend()


plt.show()
mse_train = mean_squared_error(y_train,y_pred_train)
mse_test = mean_squared_error(y_test, y_pred_test)

r2_train = r2_score(y_train,y_pred_train)
r2_test = r2_score(y_test, y_pred_test)

"""y_pred_test_rounded = np.round(y_pred_test)

accuracy_test = accuracy_score(np.round(y_test), y_pred_test_rounded)

conf_matrix = confusion_matrix(np.round(y_test), y_pred_test_rounded)"""

print(f'Mean Squared Error (MSE) on train set: {mse_train}')
print(f'Mean Squared Error (MSE) on Test Set: {mse_test}')

print(f'R-squared (R²) on Train set: {r2_train}')
print(f'R-squared (R²) on Test Set: {r2_test}')

"""print(f'Accuracy: {accuracy_test * 100:.2f}%')
print('Confusion Matrix:')
print(conf_matrix)
"""