#!/usr/bin/env python
# coding: utf-8

# In[9]:


import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix,classification_report
from sklearn.model_selection import train_test_split #import train_test_split function
from sklearn import metrics #import scikit-learn metrics module for accuracy calculation
from sklearn import tree    #import scikit-learn tree module to plot decision tree


# In[10]:


from sklearn.preprocessing import LabelEncoder
pd.set_option('display.max_columns',None)#displaying long list of columns
pd.set_option('display.max_rows', None)#displaying long list of rows
pd.set_option('display.width', 1000)#width of window


# In[11]:


import os
os.chdir(r'C:\Users\vsingh94\Documents\ASU\Course_work\CIS 508_Data Mining I\Assignments\I_Assignment1')


# In[13]:


#Read training data file 
trainfile = 'SCS_TRAIN.csv'
trainData = pd.read_csv(trainfile)

#Read test data file
testfile = 'SCS_TEST.csv'
testData = pd.read_csv(testfile)


# In[15]:


print(trainData.shape)
print(testData.shape)


# In[16]:


trainData.info()
print()
testData.info()


# In[24]:


import seaborn as sns
import matplotlib.pyplot as plt

print(trainData['TARGET'].value_counts())
plt.figure(figsize=(10,6))
sns.countplot(x=trainData['TARGET'])
plt.xlabel('Target', size=14)
plt.ylabel('Count', size=14 )
plt.title('Distribution of Target column values before split', size=14)


# In[18]:


trainData.isna().sum().sort_values(ascending=False)


# In[106]:


testData.isna().sum().sort_values(ascending=False)


# In[26]:


# to check basic statistics of data set, columnwise
# trainData.describe().to_csv('Results\submission.csv', index = True, mode='w')
trainData.describe()


# In[97]:


trainData['var3'].replace(-999999, 2, inplace = True)


# In[108]:


# unq_val= trainData.nunique().to_csv('Results/unq_vals.csv', index = True, mode='w')
trainData.nunique()


# In[27]:


unq_val_df = pd.DataFrame(trainData.nunique())
unq_val_df.columns =['Unique_values']
col_drop_df = unq_val_df[unq_val_df['Unique_values'] == 1]
col_drop_df.head()


# In[28]:


#to get list of names of all columns from dataframe
traincols = list(trainData.columns.values)
testcols = list(testData.columns.values)
print(traincols)
print(testcols)


# In[29]:


#Define threshold for dropping columns from Train data
drop_prct=int(0.6*(trainData.shape[0]))
print(drop_prct)
#Drop columns that have less than "thresh" number of non_Nans
td_na=trainData.dropna(thresh=drop_prct,axis=1)
print(td_na.shape)


# In[30]:


#Define threshold for dropping columns from test data
drop_prct=int(0.6*(testData.shape[0]))
print(drop_prct)
#Drop columns that have less than "thresh" number of non_Nans
ts_na=testData.dropna(thresh=drop_prct,axis=1)
print(ts_na.shape)


# In[31]:


traincols = list(td_na.columns.values)
drop_train_cols = []
for i in traincols:
    if td_na[i].nunique() == 1:
        print(td_na[i].unique())
        drop_train_cols.append(i)
print(len(drop_train_cols))
print(drop_train_cols)


# In[32]:


testcols = list(ts_na.columns.values)
drop_test_cols = []
for i in testcols:
    if testData[i].nunique() == 1:
        print(testData[i].unique())
        drop_test_cols.append(i)
print(len(drop_test_cols))
print(drop_test_cols)


# In[33]:


#NOW IMPUTE MISSING VALUES FOR THE OTHER COLUMNS=========================
#IMPUTE (SUBSTITUTE) MEAN VALUES FOR NaN IN NUMERIC COLUMNS 
numeric=td_na.select_dtypes(include=['int','float64']).columns
for num in numeric:
  td_na[num]=td_na[num].fillna(td_na[num].mean())

#IMPUTE (SUBSTITUTE) MODE VALUES FOR NaN IN CATEGORICAL COLUMNS
train_cat_cols = td_na.select_dtypes(exclude=['int','float64']).columns#selecting the categorical columns
for colss in train_cat_cols:
  if(td_na.iloc[0][colss]=="N"):
        td_na[colss]=td_na[colss].fillna("N")
  else:
    td_na[colss]=td_na[colss].fillna(td_na[colss].mode())
  
print(td_na.head())


# In[34]:


traincols = list(td_na.columns.values)
drop_train_cols = []
for i in traincols:
    if td_na[i].nunique() == 1:
        print(td_na[i].unique())
        drop_train_cols.append(i)
print(len(drop_train_cols))
print(drop_train_cols)


# In[35]:


print("First list is:", drop_train_cols)
print("_"*100)
print("Second list is:", drop_test_cols)
print("_"*100)
set1 = set(drop_train_cols)
set2 = set(drop_test_cols)

common_col_list = list(set1.intersection(set2))
print("Number of common columns: ", len(common_col_list))
print("Intersection of the lists is:", common_col_list)

tr_minus_ts = [x for x in set1 if x not in set2]
print("_"*100)
print("Number of train minus test  columns: ", len(tr_minus_ts))
print("train minus test columns: ", tr_minus_ts)

ts_minus_tr = [x for x in set2 if x not in set1]
print("_"*100)
print("Number of train minus test  columns: ", len(ts_minus_tr))
print("train minus test columns: ", ts_minus_tr)


# In[37]:


#DROP COLUMNS THAT STILL HAVE NULL/only one unique VALUES in Training data
print(td_na.shape)
for col in drop_train_cols:
    if col in td_na:
        td_na = td_na.drop(columns= [col])
td_na.isnull().sum()
print(td_na.shape)


# In[38]:


#DROP COLUMNS THAT STILL HAVE NULL/only one unique VALUES in Test data
print(ts_na.shape)
for col in drop_train_cols:
    if col in ts_na:
        ts_na = ts_na.drop(columns= [col])
ts_na.isnull().sum()
print(ts_na.shape)


# In[39]:


col_train = list(td_na.columns.values)
col_test = list(ts_na.columns.values)
# print(col_train)
# print(col_test)

#Seprate target column from train data
Xtrain = td_na[col_train[0:len(col_train)-1]].copy()
Ytrain = td_na[['TARGET']].copy()

print("Xtrain: ", Xtrain.shape)
print("Ytrain: ", Ytrain.shape)

Xtest = ts_na.copy()
print("Xtest: ", Xtest.shape)

#converting to integer for classifier
Ytrain['TARGET'] = Ytrain['TARGET'].astype(int)


# In[40]:


# initializing Decision tree algorithm and fitting the model on train set
dt = DecisionTreeClassifier(criterion='gini', max_depth=8, max_leaf_nodes=300)
dt.fit(Xtrain, Ytrain)
#Y_Pred = dt.predict(Xtest)
#Y_Pred = pd.DataFrame(Y_Pred,columns=['TARGET'])
#Y_Pred.to_csv(index=False)
# Use this Y_Pred on Kaggle website to get accuracy result.


# In[41]:


#Basic Analysis
print("Count of 0 & 1 in target for the Train data")
print(Ytrain['TARGET'].value_counts())


# In[42]:


# Accuracy of the algorithm, we need to predict the dataset for which we have available target
# predict the Xtrain and check the accuracy with target values we have in order to judge our model

X_Pred = dt.predict(Xtrain)
# accuracy
print("accuracy: ", metrics.accuracy_score(Ytrain,X_Pred))

# we also need to consider result from traintestslpit dataset 


# In[53]:


# slpit the dataset 
X_train, X_test, Y_train, Y_test = train_test_split(Xtrain, Ytrain, test_size = 0.4, random_state = 42, shuffle=True)
#fit model on new training dataset
dt = DecisionTreeClassifier(criterion='gini', max_depth=8, max_leaf_nodes=300)
dt = dt.fit(X_train, Y_train)
#Predict the responce on new testing dataset
Y_PredNew = dt.predict(X_test)
#Model accuracy
print("accuracy: ",metrics.accuracy_score(Y_test,Y_PredNew) )

dt = DecisionTreeClassifier(criterion='gini', max_depth=8, max_leaf_nodes=300)
dt = dt.fit(X_train, Y_train)


# In[54]:


# Actual VS predict matrix and TP, FP, FN , TN evaluation
print(metrics.confusion_matrix(Y_test, Y_PredNew))
print()
print('Printing the precision and recall, among other metrics')
print(metrics.classification_report(Y_test, Y_PredNew))


# In[102]:


#plotting the decision Tree
tree.plot_tree(dt)


# In[55]:


#get the prediction on Test dataset as column Target
pred = pd.DataFrame(dt.predict(Xtest),columns=['TARGET'])
pred.head()


# In[56]:


#write into a file with actual prediction and ID column\
final_df = pd.DataFrame({'ID': Xtest['ID'], 'TARGET': pred['TARGET']})
final_df.head()


# In[57]:


submit_file = 'Results\kaggle_submit_6.csv'
final_df.to_csv(submit_file, index=False)


# 
