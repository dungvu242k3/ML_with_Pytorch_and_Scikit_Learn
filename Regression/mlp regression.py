import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

dataset = pd.read_csv("C:/Users/dungv/Projects/ML_with_Pytorch_and_Scikit_Learn/Regression/HousingData.csv")
imputer = SimpleImputer(strategy='mean')
X = imputer.fit_transform(dataset.iloc[:, 0:13].values)
y = dataset.iloc[:, 13].values.reshape(-1,1)

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=25)

mlp = make_pipeline(StandardScaler(),MLPRegressor(hidden_layer_sizes=(100,100),max_iter=1000,random_state = 25))
mlp.fit(X_train,y_train)
y_pred_train = mlp.predict(X_train)
r2_score_train = r2_score(y_train, y_pred_train)

y_pred_test = mlp.predict(X_test)
r2_score_test = r2_score(y_test,y_pred_test)

rmse_rf = (np.sqrt(mean_squared_error(y_test, y_pred_test)))

print('R2_score (train): ', r2_score_train)
print('R2_score (test): ', r2_score_test)
print("RMSE: ", rmse_rf)
