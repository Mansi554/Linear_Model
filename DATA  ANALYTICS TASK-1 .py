#!/usr/bin/env python
# coding: utf-8

#  BY: MANSI CHOPRA
#     GRIP@THE SPARKS FOUNDATION
# TASK-1 :Predict the percentage of a student based on no.of hours studies using Supervised Machine Learning.
# PROBLEM STATEMENT: What will be predicted score if a student studies for 9.25 hrs/day?

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns 
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


# In[2]:


url="https://raw.githubusercontent.com/AdiPersonalWorks/Random/master/student_scores%20-%20student_scores.csv"
data= pd.read_csv(url)
print("Data Imported!")
data.head()


# In[7]:


data.describe()


# In[8]:


data.info()


# In[10]:


data.shape


# In[11]:


#check null values
data.isnull().sum()


# In[12]:


#plotting the distribution
data.plot(x='Hours',y='Scores',style='*')
plt.title('Hours vs % Score')
plt.xlabel('Hours')
plt.ylabel('% Score')
plt.show()


# In[13]:


x= data.iloc[:,:-1].values
y= data.iloc[:,-1].values


# In[14]:


x_train,x_test,y_train,y_test= train_test_split(x,y,test_size=0.2, random_state=0)
regressor= LinearRegression()
regressor.fit(x_train,y_train)
print("Training Done")


# In[16]:


line= regressor.coef_*x + regressor.intercept_
plt.scatter(x,y)
plt.plot(x,line,color='black')
plt.show()


# In[17]:


print(x_test)
y_pred= regressor.predict(x_test)


# In[18]:


#COMAPRISON OF ACTUAL VS PREDICTED
df=pd.DataFrame({'Actual Result': y_test,'Predicted Result':y_pred})
df


# In[19]:


#training score
print("Training Score:", regressor.score(x_train,y_train))
#test score
print("Test Score:", regressor.score(x_test,y_test))


# In[21]:


df.plot(kind='bar',figsize=(6,6))
plt.grid(linewidth='0.5',color='blue')
plt.grid(linewidth='0.5',color='green')


# In[27]:


Hours= 9.5
test=np.array([Hours]).reshape(-1,1)
prediction=regressor.predict(test)
print("No. of Hours ={}".format(Hours))
print("Predicted Score={}".format(prediction[0]))


# In[28]:


from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
print("MSE:", mean_absolute_error(y_test,y_pred))
print("MSE:",mean_squared_error(y_test,y_pred))
print("R2 score", r2_score(y_test,y_pred))


# In[ ]:




