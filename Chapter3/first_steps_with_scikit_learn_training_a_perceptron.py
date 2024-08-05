import numpy as np
from sklearn import datasets
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

iris = datasets.load_iris()
X = iris.data[:,[2,3]]
y = iris.target
print("class label",np.unique(y))


from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.3,random_state = 1,stratify = y)
print('label counts in y ',np.bincount(y))
print("labels counts in y_train ", np.bincount(y_train))
print("label count in y_test", np.bincount(y_test))

sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)
print(X_train_std)
print(X_test_std)


