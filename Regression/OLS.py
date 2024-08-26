import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

dataset = pd.read_csv("C:/Users/dungv/Projects/ML_with_Pytorch_and_Scikit_Learn/Regression/HousingData.csv")
imputer = SimpleImputer(strategy='mean')
X = imputer.fit_transform(dataset.iloc[:, 0:13].values)
y = dataset.iloc[:, 13].values

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=25)
X_train = sm.add_constant(X_train)
X_test = sm.add_constant(X_test)

ols_model = sm.OLS(y_train, X_train).fit()

y_pred_train = ols_model.predict(X_train)
r2_score_train = r2_score(y_train, y_pred_train)

y_pred_test = ols_model.predict(X_test)
r2_score_test = r2_score(y_test,y_pred_test)

rmse_rf = (np.sqrt(mean_squared_error(y_test, y_pred_test)))

print('R2_score (train): ', r2_score_train)
print('R2_score (test): ', r2_score_test)
print("RMSE: ", rmse_rf)

plt.figure(figsize=(10,6))
plt.scatter(y_test, y_pred_test, label='randomforest regression')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red')
plt.xlabel('Actual Prices')
plt.ylabel('Predicted Prices')
plt.legend()
plt.show()
