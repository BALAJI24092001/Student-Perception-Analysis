#!/usr/bin/env python
# coding: utf-8

# <h1> <i> <u> Student Perception Analysis using Multiple Linear Regression

# ## Importing libaries and understanding the data

# In[1]:


import numpy as np
import pandas as pd
from pandas.plotting import table
import matplotlib.pyplot as plt
import matplotlib.colors as pltcol
import matplotlib.ticker as ticker
import seaborn as sns
# get_ipython().run_line_magic('matplotlib', 'inline')
import math
import statsmodels.api as sm
from statsmodels.formula.api import ols
import statsmodels.tools.eval_measures as ev
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, make_scorer
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler,MinMaxScaler


# In[2]:


data = pd.read_csv('StudentData.csv')
print(len(data))


# In[3]:


print(data.info())
# Data is processed through feature engineering techniques using bivariable analysis


# In[4]:


print(data.describe())
# comparing median, max and min, there may be outliers in Age, tdu, doc and marks


# #### ocd = Online class duration (H0)
# #### eocd = expected online class duration
# #### tdu = Total data usage
# #### ss = self study
# #### doc = Data online classes (H0)
# #### ac = Academic Outcome (H0)
# #### is = Internet speed (H0)
# <!--     (5 point likert scale data) to measure satisfaction-->
# #### buc = beter in understanding the concept (H0) 
# <!--     (ordinal scale) to measure degree fo occurence-->
# #### poc = Participation in online classes (H0)
# #### ata = availability of teacher's assistance (H0)
# #### smu = social media usage (H0)
# #### bc = bored in class (H0) 
# #### ce = chear in exams (H0) 
# 
#     after testing different models:
#     buc variable has no impact on response variable
#     doc has many outliers and also not impacting the variable

# In[5]:


for i in data.columns:
    print(i)
    print(data[i].value_counts())
    print('------------------------------')


# ## Exploratory Data Analysis

# In[6]:


# plt.figure(figsize=(12, 10)) # code 
# sns.heatmap(data=data.corr(), annot=True, vmin=-1, cmap='winter') # code
# ss, ocd, eocd, doc has cosiderable correlation
# selected these variables and validating using exploratory data analysis considering ocd and eocd has significant correlation, colleniearity must be removed


# ### EDA / Univariate
# To detect outliers or anomolies in the data to manipulate accordingly by comparing using bivariate data analysis

# In[7]:


# plt.figure(figsize=(10, 10)) # code
# data['Age'].plot() # code 
#Age predictor has consistent line graph with possible outliers at age of 27-30 
#(because of less data available from phd students)


# In[8]:


# plt.figure(figsize=(10, 10)) # code 
# data.ss.plot() # code 
# consistent graph with no possible outliers
# possible for right skewed distribution


# In[9]:


# plt.figure(figsize=(10, 10)) # code 
# sns.histplot(data=data,binwidth=0.9, x='ss') # code 


# In[10]:


# plt.figure(figsize=(10, 10)) # code 
# data.ocd.plot() # code 
# cosistent graph with possible outliers at 1 or consistent


# In[11]:


# plt.figure(figsize=(10, 10)) # code
# sns.histplot(data=data, x= 'ocd', binwidth=1.4) # code
# left skewed


# In[12]:


# plt.figure(figsize=(10, 10))# code
# data.eocd.plot() # code 
# cosistent graph, possible outliers at 0


# In[13]:


# plt.figure(figsize=(10, 10)) # code
# sns.histplot(data=data, x='eocd', binwidth=0.9, kde=True) # code
# possibly left skewed with most of the dist. in right part of the dist.


# In[14]:


# plt.figure(figsize=(10, 10)) # code
# data.tdu.plot() # code
# outlier at 6 and possibly right skewed


# In[15]:


# plt.figure(figsize=(10, 10)) # code
# sns.histplot(data=data, x='tdu', binwidth=1) # code
# right skewed with ouliers on right end of dist.


# In[16]:


# plt.figure(figsize=(10, 10)) # code
# data.doc.plot() # code
# possible outliers at 0 and 3 and possible to be right skewed


# In[17]:


# plt.figure(figsize=(10, 10)) # code
# sns.histplot(data=data, x='doc', binwidth=0.5) # code


# In[18]:


# plt.figure(figsize=(12, 10)) # code
# data.drop('Marks', axis=1).boxplot(grid = False)
# plt.xticks(size=11);
# plt.yticks(size=13);
# plt.xlabel('Predictor variables')
# plt.title('Box plot for outlier analysis', size=20)
# Inter quartile range 


# ### EDA / Bivariate data analysis
# Compare the response variable with avialable ordianal variables to hypothesise the impact and to select the variable for predicting the response variable.
# plt.figure(figsize=(10, 10)) # code
# sns.histplot(x=data['Marks'], hue=data['Gender'], multiple='stack', binwidth=5)
# Gender ordinal variable has no significan factoring impact on the response variable
# variable not selected
# In[19]:


# plt.figure(figsize=(10, 10))
# sns.histplot(data=data, x = 'Marks', bins=10, hue= 'Education', multiple='stack');
# Due to less avialability of data from phd students and no significant difference in impacting the response variable
# variable no selected
# testing models, pg students has less marks and compared to other grads, even though its not significant, it helped incressing 2% more accuracy


# In[20]:


# plt.figure(figsize=(10, 10))
# sns.histplot(data=data, x='Marks', hue='ic', multiple='stack');
# plt.legend( fontsize='x-large', title = "Internet speed", loc='upper left')
# Internet speed variable has impact on the response variable, people with the best and good internet connection are more likely to get good marks and agrees online classes are better
# variable selec


# In[21]:


# plt.figure(figsize=(10, 10))
# sns.histplot(data=data, x='Marks', hue='ac', multiple='stack', hue_order=['Yes', 'No'])
# Academic outcome has a significant impact on the response variable
# variable selected


# In[22]:


# plt.figure(figsize=(10, 10))
# sns.histplot(data=data, x='Marks', hue='buc', multiple='stack')
# even though there is no significant difference of impact, most of the student with above 80 marks has agreed that online lernign is better that offline learning
# variable selected


# In[23]:


# plt.figure(figsize=(10, 10))
# sns.histplot(data=data, x='Marks', hue='poc', multiple='stack');
# no impact on response variable
# variable not selected


# In[24]:


# plt.figure(figsize=(10, 10))
# sns.histplot(data=data, x='Marks', hue='ata', multiple='stack')
# The higher the marks the most people agreed they are getting teachers assistance
# Even though there is no significant impact, the diffecrence in acceptence in good marks region can impact the response variable moderately
# variable selected


# In[25]:


# plt.figure(figsize=(10, 10))
# sns.histplot(data=data, x='Marks', hue='smu', multiple='stack')
# some people of above 75 marks has not uses socail media
# variable selected


# In[26]:


# plt.figure(figsize=(10, 10))
# sns.histplot(data=data, x='Marks', hue= 'bc', multiple='stack')
# some people above 75 have never got bored in online classes
# varible selected


# In[27]:


# plt.figure(figsize=(10, 10))
# sns.histplot(data=data, x='Marks', hue='ce', multiple='stack')
# some students with more than 75 marks says, they never cheated in exams


# In[28]:


# sns.jointplot(data=data, x='Marks', y='doc', kind='reg')


# ## Feature Engineering

#     # Missing values and alomolies were alredy processed and manipulated sucessfully

# In[29]:


print(data.info())

temp = pd.get_dummies(data['Gender'], drop_first=True)
data = pd.concat([data, temp], axis=1)
# variable not selected
# In[30]:


temp = pd.get_dummies(data['Education'], drop_first=True)
data = pd.concat([data, temp], axis=1)


# In[31]:


temp = pd.get_dummies(data['ic'], drop_first=True)
data = pd.concat([data, temp], axis=1)
# tw, b, g, b, tb


# In[32]:


temp = pd.get_dummies(data['ac'], drop_first=True)
data = pd.concat([data, temp], axis=1)
# Yes, No


# In[33]:


temp = pd.get_dummies(data['buc'], drop_first=True)
data = pd.concat([data, temp], axis=1)
# d, n, a

temp = pd.get_dummies(data['poc'], drop_first=True)
data = pd.concat([data, temp], axis=1)
# variable not selected
# In[34]:


temp = pd.get_dummies(data['ata'], drop_first=True)
data = pd.concat([data, temp], axis=1)
# st2, n2, a2


# In[35]:


temp = pd.get_dummies(data['smu'], drop_first=True)
data = pd.concat([data, temp], axis=1)
# st3, n3, a3


# In[36]:


temp = pd.get_dummies(data['bc'], drop_first=True)
data = pd.concat([data, temp], axis=1)
# st4, n4, a4


# In[37]:


temp = pd.get_dummies(data['ce'], drop_first=True)
data = pd.concat([data, temp], axis=1)
# st5, n5, a5


# In[38]:


print(data.info())


# # MLR model

# In[39]:


model = ols('Marks ~ ss + doc + d + n + phd + ug + eocd + Yes + g + no + tw + tb + n2 + n3 + n4 + n5 + st2 + st3 + st4 + st5', data).fit();
print(model.params)
# 5th model selected
# 13 dependent variables
# 4 numeric variabels


# In[40]:


print(model.summary2())
# model 1 64.7%
# model 2 68.2%
# model 3 62.0%
# model 4 78.6%
# model 5 81.2%


# In[41]:


def evaluateModel(model):
    print("RSS = ", ((data.Marks - model.predict())**2).sum())
    print("R2 = ", model.rsquared)


# In[42]:


evaluateModel(model);
# our model is 81.2% accurate


# In[ ]:





# In[ ]:





# ==================================================================================================================

# In[ ]:





# In[43]:


df1 = pd.read_csv('SFab.csv')
print(df1)


# In[44]:


model1 = ols('Moi ~ Hum', df1).fit()
print(model1.params)


# In[45]:


print(model1.summary2())


# In[46]:


model2 = ols('Hum ~ Moi', df1).fit()
print(model2.params)


# In[47]:


# sns.jointplot(data=df1, x='Hum', y='Moi', kind='reg') # code


# In[ ]:


predicted=model1.predict([38, 39])
print(predicted)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[48]:


df2 = pd.read_csv('profit.csv')
print(df2)


# In[49]:


model3 = ols('y ~ x1 + x2', df2).fit()
print(model3.params)


# In[50]:


print(model3.summary2(alpha=0.01))


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[51]:


df3 = pd.read_csv('SP.csv')
print(df3.info())


# In[52]:


# plt.figure(figsize=(30, 12))
# plt.xticks(size=25)
# plt.yticks(size=25)
# sns.histplot(data=df3, x='Which device do you use for online classes.   (multiple options available)', binwidth=0.5,hue='Which device do you use for online classes.   (multiple options available)', stat='percent')


# In[ ]:




