import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

df = pd.read_csv("C:/Users/dungv/Projects/ML_with_Pytorch_and_Scikit_Learn/Chapter3/multi layer perceptron/factbook.csv")
df.head()

df = df.replace(' ', np.nan)
df.rename(columns={'Country':'country', ' Area':'area', ' Birth rate':'birth rate', '  Current account balance ':'current account balance',
       ' Death rate':'death rate', ' Electricity consumption':'electricity consumption', '  Electricity production ':'electricity production',
       '  Exports ':'exports', '  GDP ':'gdp', '  GDP per capita ':'gdp per capita', ' GDP real growth rate':'gdp real growth rate',
       '  Highways ':'highways', '  Imports ':'imports', ' Industrial production growth rate':'industrial production growth rate',
       ' Infant mortality rate':'infant mortality rate', ' Inflation rate ':'inflation rate', '  Internet users ':'internet users',
       ' Investment':'investment', '  Labor force ':'labor force', ' Life expectancy at birth':'life expectancy at birth',
       ' Military expenditures':'military expenditures', '  Natural gas consumption ':'natural gas consumption',
       '  Oil consumption ':'oil consumption', '  Population ':'population', ' Public debt':'public debt', ' Railways':'railways',
       '  Reserves of foreign exchange & gold ':'reserves of foreign exchange and gold', ' Total fertility rate':'total fertility rate',
       ' Unemployment rate':'unemployement rate'}, inplace=True)
df = df[['exports', 'imports', 'industrial production growth rate', 'investment', 'unemployement rate', 'gdp']]
df.head()
df.shape()

