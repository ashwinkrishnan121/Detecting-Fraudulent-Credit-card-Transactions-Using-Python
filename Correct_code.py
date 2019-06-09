# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
pip install xgboost

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns 

import datetime # to dela with date and time
from sklearn.preprocessing import StandardScaler # for preprocessing the data
from sklearn.ensemble import RandomForestClassifier # Random forest classifier
from sklearn.tree import DecisionTreeClassifier # for Decision Tree classifier
from sklearn.svm import SVC # for SVM classification
from xgboost import XGBClassifier # For XG-Boost Classifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split # to split the data
from sklearn.model_selection import KFold # For cross vbalidation
from sklearn.model_selection import GridSearchCV # for tunnig hyper parameter it will use all combination of given parameters
from sklearn.model_selection import RandomizedSearchCV # same for tunning hyper parameter but will use random combinations of parameters
from sklearn.metrics import confusion_matrix,recall_score,precision_recall_curve,auc,roc_curve,roc_auc_score,classification_report
import warnings

#Impoting the data
data = pd.read_csv("creditcard.csv")
data.head()

#Data Cleaning
print(data.info())
data['normAmount'].describe()
data['normAmount'].plot(kind='hist', rot=70, logx=True, logy=True)
data.boxplot(column='normAmount', by='Borough', rot=90)
data.Class = data.Class.astype('category')
data = data.drop_duplicates()

#Count missing values
data.isnull().sum()

#To show the skewness of the data

sns.countplot("Class",data=data)

Normal_trans = len(data[data["Class"]==0]) 
Fraud_trans = len(data[data["Class"]==1]) 
print("Number of normal transactions", Normal_trans)
print("Number of Fraudulent transactions",Fraud_trans)
Perc_Normal_trans = Normal_trans/(Normal_trans+Fraud_trans)
print("percentage of normal transacation is",Perc_Normal_trans*100)
Perc_Fraud_trans = Fraud_trans/(Normal_trans+Fraud_trans)
print("percentage of fraud transacation",Perc_Fraud_trans*100)


#Checking Amount distribution for Normal and Fraud Transactions

Fraud_trans_df = data[data["Class"]==1]
Normal_trans_df = data[data["Class"]==0]
plt.figure(figsize=(10,6))
plt.subplot(121)
Fraud_trans_df[Fraud_trans_df["Amount"]<= 2500].Amount.plot.hist(title="Fraud Tranascation")
plt.subplot(122)
Normal_trans_df[Normal_trans_df["Amount"]<=2500].Amount.plot.hist(title="Normal Transaction")

# Normlising the Amount Column
data['normAmount'] = (data['Amount']-min(data['Amount']))/(max(data['Amount'])-min(data['Amount']))
data = data.drop(['Time','Amount'],axis=1)
data.head()


#Resampling the Data Set

X = data.ix[:, data.columns != 'Class']
y = data.ix[:, data.columns == 'Class']


def undersample(i): 
       # Number of data points in the minority class
       no_fraud_trans = len(data[data.Class == 1])
       fraud_index = np.array(data[data.Class == 1].index)
       # Picking the indices of the normal classes
       normal_index = data[data.Class == 0].index
       # Out of the indices we picked, randomly select "x" number (number_records_fraud)
       rand_normal_index = np.random.choice(normal_index, no_fraud_trans*i, replace = False)
       rand_normal_index = np.array(rand_normal_index)
       # Appending the 2 indices
       under_sample_index = np.concatenate([fraud_index,rand_normal_index])
       # Under sample dataset
       under_sample_data = data.iloc[under_sample_index,:]
       return(under_sample_data)

    

## first make a model function for modeling with confusion matrix
def model(model,train_data,test_data,train_output,test_output):
    clf= model
    clf.fit(train_data,train_output.values.ravel())
    pred=clf.predict(test_data)
    cnf_matrix=confusion_matrix(test_output,pred)
    print("the recall for this model is :",cnf_matrix[1,1]/(cnf_matrix[1,1]+cnf_matrix[1,0]))
    fig= plt.figure(figsize=(6,3))# to plot the graph
    print("TP",cnf_matrix[1,1,]) # no of fraud transaction which are predicted fraud
    print("TN",cnf_matrix[0,0]) # no. of normal transaction which are predited normal
    print("FP",cnf_matrix[0,1]) # no of normal transaction which are predicted fraud
    print("FN",cnf_matrix[1,0]) # no of fraud Transaction which are predicted normal
    sns.heatmap(cnf_matrix,cmap="coolwarm_r",annot=True,linewidths=0.5)
    plt.title("Confusion_matrix")
    plt.xlabel("Predicted_class")
    plt.ylabel("Real class")
    plt.show()
    print("\n----------Classification Report------------------------------------")
    print(classification_report(test_output,pred)) 
    
    
## Creating the training and testing data
def data_prepration(x): # preparing data for training and testing as we are going to use different data 
    #again and again so make a function
    x_data = x.ix[:,x.columns != "Class"]
    x_output =x.ix[:,x.columns=="Class"]
    x_data_train,x_data_test,x_output_train,x_output_test = train_test_split(x_data,x_output,test_size=0.3)
    print("length of training data")
    print(len(x_data_train))
    print("length of test data")
    print(len(x_data_test))
    return(x_data_train,x_data_test,x_output_train,x_output_test)


#train this model using undersample data and test for the whole data test set 
#Logistic Regression
for i in range(1,4):
    Undersample_data = undersample(i)
    US_train_input,US_test_input,US_train_output,US_test_output=data_prepration(Undersample_data)
    train_input,test_input,train_output,test_output=data_prepration(data) 
    #the partion for whole data
    print()
    clf=LogisticRegression()
    model(clf,US_train_input,test_input,US_train_output,test_output)
 

 #train this model using undersample data and test for the whole data test set 
 #SVM
for i in range(1,4):
    Undersample_data = undersample(i)
    US_train_input,US_test_input,US_train_output,US_test_output=data_prepration(Undersample_data)
    train_input,test_input,train_output,test_output=data_prepration(data) 
    #the partion for whole data
    print()
    clf=SVC()
    model(clf,US_train_input,test_input,US_train_output,test_output)   


 #train this model using undersample data and test for the whole data test set 
 #Random Forest
for i in range(1,4):
    Undersample_data = undersample(i)
    US_train_input,US_test_input,US_train_output,US_test_output=data_prepration(Undersample_data)
    train_input,test_input,train_output,test_output=data_prepration(data) 
    #the partion for whole data
    print()
    clf=RandomForestClassifier(n_estimators=100)
    model(clf,US_train_input,test_input,US_train_output,test_output) 
    
  #train this model using undersample data and test for the whole data test set 
  #XG-Boost
for i in range(1,4):
    Undersample_data = undersample(i)
    US_train_input,US_test_input,US_train_output,US_test_output=data_prepration(Undersample_data)
    train_input,test_input,train_output,test_output=data_prepration(data) 
    #the partion for whole data
    print()
    clf=XGBClassifier()
    model(clf,US_train_input,test_input,US_train_output,test_output)   
    
    
    

 #oversample the train data and predict for test data
 data1 = pd.read_csv("creditcard.csv",header = 0)
     

#oversampling of training data 
# duplicate many times the value of fraud data
def oversample(k):
     train_input,test_input,train_output,test_output=data_prepration(data1)
     train_input.columns
     train_output.columns
     # ok Now we have a traing data
     train_input["Class"]= train_output["Class"] # combining class with original data
     traindata = train_input.copy() # for naming conevntion
     print("length of training data",len(traindata))
     # Now make data set of normal transction from train data
     normal_data = traindata[traindata["Class"]==0]
     print("length of normal data",len(normal_data))
     fraud_data = traindata[traindata["Class"]==1]
     print("length of fraud data",len(fraud_data))
     #for that we have to standrdize the normal amount and drop the time from it
     test_input['normAmount'] = (test_input['Amount']-min(test_input['Amount']))/(max(test_input['Amount'])-min(test_input['Amount']))
     test_input.drop(["Time","Amount"],axis=1,inplace=True)
     #Over Sampling
     for i in range (100*k): 
           normal_data= normal_data.append(fraud_data)
     os_data = normal_data.copy() 
     print("Proportion of Normal data in oversampled data is ",len(os_data[os_data["Class"]==0])/len(os_data))
     print("Proportion of fraud data in oversampled data is ",len(os_data[os_data["Class"]==1])/len(os_data))
     # before applying any model standerdize our data amount 
     os_data['normAmount'] = (os_data['Amount']-min(os_data['Amount']))/(max(os_data['Amount'])-min(os_data['Amount']))
     os_data.drop(["Time","Amount"],axis=1,inplace=True)
     # now take all over sampled data as trainging and test it for test data
     os_data_X = os_data.ix[:,os_data.columns != "Class"]
     os_data_y = os_data.ix[:,os_data.columns == "Class"]
     return(os_data_X,os_data_y,test_input,test_output)

#train this model using undersample data and test for the whole data test set 
#Logistic Regression
for k in range(1,4):
    os_input, os_output, test_input, test_output = oversample(8)
    clf=LogisticRegression()
    model(clf,os_input,test_input,os_output,test_output)
 

 #train this model using undersample data and test for the whole data test set 
 #SVM
for k in range(1,4):
  os_input, os_output,test_input, test_output = oversample(2)
  clf=SVC()
  model(clf,os_input,test_input,os_output,test_output)


 #train this model using undersample data and test for the whole data test set 
 #Random Forest
for i in range(1,4):
   os_input, os_output,test_input, test_output = oversample(7)
   clf=RandomForestClassifier(n_estimators=100)
   model(clf,os_input,test_input,os_output,test_output)
   
# To check the feature importance
featimp = pd.Series(clf.feature_importances_,index=os_input.columns).sort_values(ascending=False)
print(featimp)

    
  #train this model using undersample data and test for the whole data test set 
  #XG-Boost
for i in range(1,4):
   os_input, os_output,test_input, test_output = oversample(2)
   clf=XGBClassifier()
   model(clf,os_input,test_input,os_output,test_output)
  
