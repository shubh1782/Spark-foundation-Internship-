#!/usr/bin/env python
# coding: utf-8

# In[1]:


#THE SPARKS FOUNDATION #GRIPOCTOBER21

#Author - SHUBHAM PATIL, DATA SCIENCE & BUSINESS ANALYTICS INTERN

#TASK-1 Prediction using Supervised ML

#Perform exploratory Data Analysis on dataset 'Student' To Predict the percentage of marks of the students based on the number of hours they studied.

#Dataset Sample: http://bit.ly/w-data


# In[2]:


# importing the required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error


# In[3]:


# Reading the Data 
data = pd.read_csv('http://bit.ly/w-data')
data.head(10)


# In[4]:


# Check if there any null value in the Dataset
data.isnull == True


# In[5]:


sns.set_style('darkgrid')
sns.scatterplot(y= data['Scores'], x= data['Hours'])
plt.title('Marks Vs Study Hours',size=20)
plt.ylabel('Marks Percentage', size=12)
plt.xlabel('Hours Studied', size=12)
plt.show()


# In[6]:


sns.regplot(x= data['Hours'], y= data['Scores'])
plt.title('Regression Plot',size=20)
plt.ylabel('Marks Percentage', size=12)
plt.xlabel('Hours Studied', size=12)
plt.show()
print(data.corr())


# In[7]:


# Defining X and y from the Data
X = data.iloc[:, :-1].values  
y = data.iloc[:, 1].values

# Spliting the Data in two
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state = 0)


# In[8]:


regression = LinearRegression()
regression.fit(train_X, train_y)
print("---------Model Trained---------")


# In[9]:


pred_y = regression.predict(val_X)
prediction = pd.DataFrame({'Hours': [i[0] for i in val_X], 'Predicted Marks': [k for k in pred_y]})
prediction


# In[10]:


compare_scores = pd.DataFrame({'Actual Marks': val_y, 'Predicted Marks': pred_y})
compare_scores


# In[11]:


plt.scatter(x=val_X, y=val_y, color='blue')
plt.plot(val_X, pred_y, color='Black')
plt.title('Actual vs Predicted', size=20)
plt.ylabel('Marks Percentage', size=12)
plt.xlabel('Hours Studied', size=12)
plt.show()


# In[12]:


# Calculating the accuracy of the model
print('Mean absolute error: ',mean_absolute_error(val_y,pred_y))


# In[ ]:




