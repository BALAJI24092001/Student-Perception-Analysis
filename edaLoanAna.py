#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
# get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


train = pd.read_csv('LoanPredictionTrainDataSet.csv')
train.head(10)


# In[ ]:


train.dtypes


# In[ ]:


Gender=pd.crosstab(train['Gender'],train['Loan_Status'])
palette = {'N':'C1', 'Y':'C0'}
Gender.div(Gender.sum(1).astype(float), axis=0).plot(kind='bar',stacked=True,figsize=(10,10))
plt.title('Gender vs Loan Status')
plt.legend(['No', 'Yes'], loc="center",title='Loan Status')
plt.xticks(rotation=0)
plt.show();
print(Gender)
# sns.barplot(x=Gender.Gender, y=Gender.values)
# sns.histplot(data=train, x="Gender", hue="Loan_Status", stacked=True)


# In[ ]:


type(train['Gender'])


# In[ ]:


Married=pd.crosstab(train['Married'], train['Loan_Status'])
Married.div(Married.sum(1).astype(float), axis=0).plot(kind='bar',stacked=True,figsize=(10, 10), color={'N':'#FF9B0B', 'Y':'#1F6993'});
plt.title('Marital Status vs Loan Status')
plt.legend(['No', 'Yes'], loc="center",title='Loan Status')
plt.xticks(rotation=0)
plt.show()


# In[ ]:


Dependents=pd.crosstab(train['Dependents'],train['Loan_Status'])
Dependents.div(Dependents.sum(1).astype(float), axis=0).plot(kind='bar',stacked=True,figsize=(10,10), color={'N':'#FF9B0B', 'Y':'#1F6993'})
plt.title('Dependents vs Loan Status')
plt.legend(['No', 'Yes'], loc="center",title='Loan Status')
plt.xticks(rotation=0)
plt.show()


# In[ ]:


Education=pd.crosstab(train['Education'],train['Loan_Status']);
Education.div(Education.sum(1).astype(float), axis=0).plot(kind='bar',stacked=True,figsize=(10,10), color={'N':'#FF9B0B', 'Y':'#1F6993'})
plt.title('Education vs Loan Status')
plt.legend(['No', 'Yes'], loc="center",title='Loan Status')
plt.xticks(rotation=0)
plt.show()


# In[ ]:


Self_Employed=pd.crosstab(train['Self_Employed'],train['Loan_Status'])
Self_Employed.div(Self_Employed.sum(1).astype(float), axis=0).plot(kind='bar',stacked=True,figsize=(10, 10), color={'N':'#FF9B0B', 'Y':'#1F6993'});
plt.title('Self Employment vs Loan Status')
plt.legend(['No', 'Yes'], loc="center",title='Loan Status')
plt.xticks(rotation=0)
plt.show()


# In[ ]:


Credit_History=pd.crosstab(train['Credit_History'],train['Loan_Status'])
Credit_History.div(Credit_History.sum(1).astype(float), axis=0).plot(kind='bar',stacked=True,figsize=(10, 10), color={'N':'#FF9B0B', 'Y':'#1F6993'});
plt.title('Credit History vs Loan Status')
plt.legend(['No', 'Yes'], loc="center",title='Loan Status')
plt.xticks(rotation=0)
plt.show()


# In[ ]:


Property_Area=pd.crosstab(train['Property_Area'],train['Loan_Status'])
Property_Area.div(Property_Area.sum(1).astype(float), axis=0).plot(kind='bar',stacked=True,figsize=(10,10), color={'N':'#FF9B0B', 'Y':'#1F6993'})
plt.title('Property Area vs Loan Status')
plt.legend(['No', 'Yes'], loc="upper right",title='Loan Status')
plt.xticks(rotation=0)
plt.show()


# In[ ]:


bins=[0,2500,4000,6000,81000]
group=['Low','Average','High','Very high']
train['Income_bin']=pd.cut(train['ApplicantIncome'],bins,labels=group)
Income_bin=pd.crosstab(train['Income_bin'],train['Loan_Status'])
Income_bin.div(Income_bin.sum(1).astype(float), axis=0).plot(kind='bar',stacked=True, figsize=(10,10), color={'N':'#FF9B0B', 'Y':'#1F6993'})
plt.xlabel('ApplicantIncome')
plt.title('Applicant income vs Loan Status')
plt.legend(['No', 'Yes'], loc="center",title='Loan Status')
plt.xticks(rotation=0)
plt.show()


# In[ ]:


bins=[0,1000,3000,42000]
group=['Low','Average','High']
train['Coapplicant_Income_bin']=pd.cut(train['CoapplicantIncome'],bins,labels=group)
Coapplicant_Income_bin=pd.crosstab(train['Coapplicant_Income_bin'],train['Loan_Status'])
Coapplicant_Income_bin.div(Coapplicant_Income_bin.sum(1).astype(float), axis=0).plot(kind='bar',stacked=True, figsize=(10,10), color={'N':'#FF9B0B', 'Y':'#1F6993'})
plt.xlabel('CoapplicantIncome')
plt.title('Coapplicant Income vs Loan Status')
plt.legend(['No', 'Yes'], loc="upper right",title='Loan Status')
plt.xticks(rotation=0)
plt.show()


# In[ ]:


train['Total_Income']=train['ApplicantIncome']+train['CoapplicantIncome']
bins=[0,2500,4000,6000,81000]
group=['Low','Average','High','Very high']
train['Total_Income_bin']=pd.cut(train['Total_Income'],bins,labels=group)
Total_Income_bin=pd.crosstab(train['Total_Income_bin'],train['Loan_Status'])
Total_Income_bin.div(Total_Income_bin.sum(1).astype(float), axis=0).plot(kind='bar',stacked=True, figsize=(10, 10), color={'N':'#FF9B0B', 'Y':'#1F6993'})
plt.xlabel('Total_Income')
plt.title('Total Income vs Loan Status')
plt.legend(['No', 'Yes'], loc="center",title='Loan Status')
plt.xticks(rotation=0)
plt.show()


# In[ ]:


bins=[0,100,200,700]
group=['Low','Average','High']
train['LoanAmount_bin']=pd.cut(train['LoanAmount'],bins,labels=group)
LoanAmount_bin=pd.crosstab(train['LoanAmount_bin'],train['Loan_Status'])
LoanAmount_bin.div(LoanAmount_bin.sum(1).astype(float), axis=0).plot(kind='bar',stacked=True, figsize=(10,10), color={'N':'#FF9B0B', 'Y':'#1F6993'})
plt.xlabel('LoanAmount')
plt.title('Loan Amount vs Loan Status')
plt.legend(['No', 'Yes'], loc="upper right",title='Loan Status')
plt.xticks(rotation=0)
plt.show()


# In[ ]:


plt.figure(figsize=(10,10))
sns.heatmap(vmax=1, square=True, cmap='Blues',annot=True,data=train.corr())


# In[ ]:


def cc(a, b, color= None, figsize=(5, 5), title='stacked bar plot'):
    '''
    a and b are pandas series, either categorical or numerical but atleast one variable must be categorical
    '''
    temp = pd.crosstab(a, b)
    temp.div(temp.sum(1).astype(float), axis=0).plot(kind='bar', stacked=True, figsize=figsize)
    plt.title(title)
    plt.show()
a = np.random.randint(1, 3, size=20)
b = pd.Series(['a', 'b', 'b', 'b', 'b', 'b', 'a', 'b', 'b', 'b', 'b', 'b', 'b', 'a', 'b', 'b', 'a', 'a', 'b', 'b'])
a = pd.Series(a)
cc(a, b)


# In[ ]:




