#!/usr/bin/env python
# coding: utf-8

# TESTING FOR VIF VALUE USING VIF IN PYTHON

# In[1]:


import numpy as np
import pandas as pd
from pandas.plotting import table
import matplotlib.pyplot as plt
import matplotlib.colors as pltcol
import matplotlib.ticker as ticker
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
import math
import statsmodels.api as sm
from statsmodels.formula.api import ols
import statsmodels.tools.eval_measures as ev
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from patsy import dmatrices
from statsmodels.stats.outliers_influence import variance_inflation_factor
# importing r2_score module
from sklearn.metrics import mean_absolute_error, make_scorer
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
# predicting the accuracy score
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler,MinMaxScaler


# In[5]:


df = pd.read_csv('StudentData.csv')
print(df.head())


# In[6]:


# temp = pd.get_dummies(df['Education'], drop_first=True) 
# df = pd.concat([df, temp], axis=1)
# temp = pd.get_dummies(df['ic'], drop_first=True) 
# df = pd.concat([df, temp], axis=1)
# temp = pd.get_dummies(df['ac'], drop_first=True) 
# df = pd.concat([df, temp], axis=1)
# # df.drop(['Education', 'ic', 'ac'], axis=1, inplace = True)
# df.head()


# In[21]:


y, X = dmatrices('Marks ~ Education + Age + Gender + ss + poc + ocd + ic + ac + buc + ata + smu+ bc + ce', df, return_type='dataframe')


# In[22]:





# In[23]:


vif = pd.DataFrame()
vif["VIF Factor"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
vif["features"] = X.columns


# In[24]:


print(vif)


# In[13]:


temp = ols('Marks ~ Education + ss + ocd + ic + ac', df);
model5= temp.fit()
print(model5.params)


# In[15]:


print(model5.summary2())


# In[ ]:




