import matplotlib.pyplot as plt
import numpy as np


class LogisticRegressionGD:
    def __init__(self, eta = 0.01, n_iter = 50,random_state = 1) :
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state
    def fit(self,X,y) :
        rgen = np.random.RandomState(self.random_state)
        self.w_ = rgen.normal(loc = 0.0 , scale = 0.01, size = X.shape[1])
        self.b_ = np.float_(0.)
        self.losses_ = []
        
        for i in range(self.n_iter) :
            net_input = self.net_input(X)
            output = self.activation(net_input)
            errors = y - output
            self.w_ += self.eta * 2.0 * X.T.dot(errors) / X.shapre[0]
            self.b_ += self.eta * 2.0 * errors.mean()
            loss = (-y.dot(np.log(output))- ((1 - y).dot(np.log(1 - output)))/ X.shape[0])
            self.losses_.append(loss)
        return self
    
    def net_input(self,X) :
        return np.dot(X,self.w_) + self.b_
    
    def activation(self,z) :
        return 1./ (1 + np.exp(-np.clip(z,-250,250)))
    def predict(self,X) :
        return np.where(self.activation(self.net_input(X)) >= 0.5,1,0)
X_train_01_subset = X_train_std[(y_train == 0) | (y_train == 1)]
y_train_01_subset = y_train[(y_train == 0) | (y_train == 1)]
lrgd = LogisticRegressionGD(eta=0.3,n_iter=1000,random_state=1)
lrgd.fit(X_train_01_subset,y_train_01_subset)
plot_decision_regions(X=X_train_01_subset,y=y_train_01_subset,
classifier=lrgd)
plt.xlabel('Petal length [standardized]')
plt.ylabel('Petal width [standardized]')
plt.legend(loc='upper left')
plt.tight_layout()
plt.show()

from sklearn.linear_model import LogisticRegression

lr = LogisticRegression(C=100.0, solver='lbfgs',multi_class='ovr')
lr.fit(X_train_std, y_train)
plot_decision_regions(X_combined_std,y_combined,classifier=lr,test_idx=range(105, 150))
plt.xlabel('Petal length [standardized]')
plt.ylabel('Petal width [standardized]')
plt.legend(loc='upper left')
plt.tight_layout()
plt.show()

weights, params = [], []
for c in np.arange(-5, 5):
    lr = LogisticRegression(C=10.**c,multi_class='ovr')
    lr.fit(X_train_std, y_train)
    weights.append(lr.coef_[1])
    params.append(10.**c)
weights = np.array(weights)
plt.plot(params, weights[:, 0],label='Petal length')
plt.plot(params, weights[:, 1], linestyle='--',label='Petal width')
plt.ylabel('Weight coefficient')
plt.xlabel('C')
plt.legend(loc='upper left')
plt.xscale('log')
plt.show()


from sklearn.svm import SVC

svm = SVC(kernel='linear', C=1.0, random_state=1)
svm.fit(X_train_std, y_train)
plot_decision_regions(X_combined_std,y_combined,classifier=svm,test_idx=range(105, 150))
plt.xlabel('Petal length [standardized]')
plt.ylabel('Petal width [standardized]')
plt.legend(loc='upper left')
plt.tight_layout()
plt.show()


svm = SVC(kernel='rbf', random_state=1, gamma=0.10, C=10.0)
svm.fit(X_xor, y_xor)
plot_decision_regions(X_xor, y_xor, classifier=svm)
plt.legend(loc='upper left')
plt.tight_layout()
plt.show()