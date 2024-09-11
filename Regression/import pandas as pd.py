import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sn
import statsmodels.api as sm
from mlxtend.feature_selection import SequentialFeatureSelector as sfs
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, PolynomialFeatures
from statsmodels.compat import lzip
from statsmodels.stats.outliers_influence import variance_inflation_factor

Boston = pd.read_csv("C:/Users/dungv/Projects/ML_with_Pytorch_and_Scikit_Learn/Regression/Boston.csv")
print(Boston.head())
print(Boston.info())
Boston["CHAS"] = Boston["CHAS"].astype('category')
Boston.info()
def iqr_func(data):
    q3, q1 = np.percentile(data, [75 ,25])
    iqr = q3 - q1
    return iqr
def outlier_func(data):
    outlier = []
    q3, q1 = np.percentile(data, [75 ,25])
    iqr = q3 - q1
    for i in data :
        if (i > (q3 + 1.5 * iqr) or i < (q1 - 1.5 * iqr)):
            outlier.append(True)
        else:
            outlier.append(False)    
    return outlier
column_names = list(Boston.columns)
Boston_out   = pd.DataFrame()
for name in column_names:
    Boston_out[name] = outlier_func(Boston[name])    
Boston_out.drop('CHAS', inplace=True, axis=1)
print(Boston_out.head(3))

Boston_colsum = Boston_out.sum()
print(Boston_colsum)
Boston_rowsum = Boston_out.sum(axis = 1)
print(Boston_colsum) 
list(Boston_rowsum[Boston_rowsum  == 4].index)
n_otlier_feature = 3
for j in range(len(Boston)):
    if Boston_rowsum[j] >= n_otlier_feature:
        Boston = Boston.drop(j)
print(Boston.info())
print(Boston.head(3))
mms = MinMaxScaler() 
print(Boston.info())
Boston_scaling = mms.fit_transform(Boston.loc[:, Boston.columns != "CHAS"]) 
Boston_scaling = pd.DataFrame(Boston_scaling)
print(Boston_scaling.info())
Boston_scaling.columns = ['CRIM', 'ZN','INDUS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX',
                           'PTRATIO','BLACK','LSTAT', 'MEDV']
print(Boston_scaling.head(3))
fig = plt.figure(figsize =(10, 7))
boxplot = Boston_scaling.boxplot(column = list(Boston_scaling.columns)) 

fig = plt.figure(figsize=(8, 8))
gs = fig.add_gridspec(2, 2, hspace=.3, wspace=.3)
(ax1, ax2), (ax3, ax4) = gs.subplots(sharex='none', sharey='none')
fig.suptitle('ScatterPlot Matrix')
ax1.scatter(Boston_scaling["CRIM"], Boston_scaling["MEDV"], c ="blue")
ax1.set(xlabel='crim', ylabel='medv')
ax2.scatter(Boston_scaling["LSTAT"], Boston_scaling["MEDV"], c ="red")
ax2.set(xlabel='lstat', ylabel='medv')
ax3.scatter(Boston_scaling["NOX"], Boston_scaling["MEDV"], c ="green")
ax3.set(xlabel='nox', ylabel='medv')
ax4.scatter(Boston_scaling["INDUS"], Boston_scaling["MEDV"], c ="orange")
ax4.set(xlabel='indus', ylabel='medv')

corrMatrix = Boston_scaling.corr()
print (corrMatrix)
fig = plt.figure(figsize =(14, 10))
sn.heatmap(corrMatrix, annot=True)
plt.show()


Data_step2 = Boston.loc[:, ["MEDV", "LSTAT"]]
X_train ,X_test = train_test_split(Data_step2,test_size=0.2,random_state=42)
model = LinearRegression()
x_train = np.array(X_train["LSTAT"]).reshape((-1, 1))
y_train = np.array(X_train["MEDV"])
model.fit(x_train,y_train)

r_sq = model.score(x_train, y_train)
print(f"coefficient of determination: {r_sq}")

print(f"intercept: {model.intercept_}")

print(f"slope: {model.coef_}")

mean_squared_error(y_train, model.predict(x_train))

x_test = np.array(X_test["LSTAT"]).reshape((-1, 1))
y_test = np.array(X_test["MEDV"])

mean_squared_error(y_test, model.predict(x_test))

plt.figure(figsize=(5, 4))
plt.scatter(Boston_scaling["LSTAT"], Boston_scaling["MEDV"], color = 'blue')  
plt.plot(x_train, model.predict(x_train), color = 'red')
plt.title('Linear Regression')
plt.xlabel('lstat')
plt.ylabel('medv')
  
plt.show()