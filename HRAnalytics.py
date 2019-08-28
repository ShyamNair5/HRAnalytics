# -*- coding: utf-8 -*-
"""
Created on Sun Apr  7 21:35:05 2019

@author: Shyam
"""
# Load libraries
import matplotlib.pyplot as plt
#setting dimension of graph
plt.rcParams["figure.figsize"]= (12, 7)

import pandas as pd
import numpy as np
import seaborn as sns

from pandas.plotting import scatter_matrix

from sklearn.decomposition import PCA
from sklearn.preprocessing import scale
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score,roc_auc_score

from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn import preprocessing
from imblearn.over_sampling import ADASYN
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler


data=pd.read_csv('D:/Shyam/Datasets/PM/train_LZdllcl.csv')



def impute_unknowns(df, column):
    df[column] = df[column].replace(np.NaN,0)
    col_values = df[column].values
    df[column] = np.where(col_values==0, data[column].mode(), col_values)
    return df


#education
#previous_year_rating

data=impute_unknowns(df=data,column='education')


data=impute_unknowns(df=data,column='previous_year_rating')



#XY

data_X=data[data.columns[1:13]]
data_X=data_X.drop('region',axis=1)
data_Y=data['is_promoted']

dataset_X_dummy = pd.get_dummies(data_X)
X=dataset_X_dummy.values
Y=data_Y.values



validation_size=0.3
seed=7
X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, test_size=validation_size, random_state=seed)

X_t = scale(X_train)


# scatter plot matrix
scatter_matrix(dataset_X_dummy)
plt.show()

plt.matshow(dataset_X_dummy.corr())
plt.show()

corr = dataset_X_dummy.corr()
sns.heatmap(corr, 
            xticklabels=corr.columns.values,
            yticklabels=corr.columns.values)



#initially lets create 40 components which is actual number of Variables we have
pca = PCA(n_components=24)

pca.fit(X_t)

#The amount of variance that each PC explains
var= pca.explained_variance_ratio_
#Cumulative Variance explains
var1=np.cumsum(np.round(pca.explained_variance_ratio_, decimals=4)*100)

#lets see Cumulative Variance plot
plt.plot(var1)



#Looking at above plot I'm taking 54 variables
pca = PCA(n_components=18)
pca.fit(X_t)
X_train_PC=pca.fit_transform(X_t)


X_v = scale(X_validation)


X_validation_PC=pca.fit_transform(X_v)


le = preprocessing.LabelEncoder()

seed=42
scoring = 'accuracy'

rfc=RandomForestClassifier()
rfc.fit(X_train_PC, Y_train)
prediction=rfc.predict(X_validation_PC)
kfold = model_selection.KFold(n_splits=10, random_state=seed)
cv_results = model_selection.cross_val_score(rfc, X_train_PC, Y_train, cv=kfold, scoring=scoring)
a=accuracy_score(Y_validation, prediction)
confusionmatrix = confusion_matrix(Y_validation, prediction)
cr=classification_report(Y_validation, prediction)

le.fit(Y_validation)
list(le.classes_)
y_=le.transform(Y_validation) 
p_=le.transform(prediction)
area=roc_auc_score(y_,p_)

print('Training Accuracy')
print(cv_results.mean())
print('Testing Accuracy')
print(a)
print('Confusion Matrix')    
print(confusionmatrix)    
print('Report')    
print(cr)    
print('AUC ROC Score')    
print(area)    

sm = SMOTE()
X_train_PC_s, Y_train_s = sm.fit_sample(X_train_PC, Y_train)

rfc=RandomForestClassifier()
rfc.fit(X_train_PC_s, Y_train_s)
prediction=rfc.predict(X_validation_PC)
kfold = model_selection.KFold(n_splits=10, random_state=seed)
cv_results = model_selection.cross_val_score(rfc, X_train_PC_s, Y_train_s, cv=kfold, scoring=scoring)
a=accuracy_score(Y_validation, prediction)
confusionmatrix = confusion_matrix(Y_validation, prediction)
cr=classification_report(Y_validation, prediction)

le.fit(Y_validation)
list(le.classes_)
y_=le.transform(Y_validation) 
p_=le.transform(prediction)
area=roc_auc_score(y_,p_)

print('Training Accuracy')
print(cv_results.mean())
print('Testing Accuracy')
print(a)
print('Confusion Matrix')    
print(confusionmatrix)    
print('Report')    
print(cr)    
print('AUC ROC Score')    
print(area)    



rus = ADASYN()
X_train_PC_r, Y_train_r = rus.fit_sample(X_train_PC, Y_train)

   
rfc=RandomForestClassifier()
rfc.fit(X_train_PC_r, Y_train_r)
prediction=rfc.predict(X_validation_PC)
kfold = model_selection.KFold(n_splits=10, random_state=seed)
cv_results = model_selection.cross_val_score(rfc, X_train_PC_r, Y_train_r, cv=kfold, scoring=scoring)
a=accuracy_score(Y_validation, prediction)
confusionmatrix = confusion_matrix(Y_validation, prediction)
cr=classification_report(Y_validation, prediction)

le.fit(Y_validation)
list(le.classes_)
y_=le.transform(Y_validation) 
p_=le.transform(prediction)
area=roc_auc_score(y_,p_)

print('Training Accuracy')
print(cv_results.mean())
print('Testing Accuracy')
print(a)
print('Confusion Matrix')    
print(confusionmatrix)    
print('Report')    
print(cr)    
print('AUC ROC Score')    
print(area)    


nb=GaussianNB()
nb.fit(X_train_PC_r, Y_train_r)
prediction=nb.predict(X_validation_PC)
kfold = model_selection.KFold(n_splits=10, random_state=seed)
cv_results = model_selection.cross_val_score(nb, X_train_PC_r, Y_train_r, cv=kfold, scoring=scoring)
a=accuracy_score(Y_validation, prediction)
confusionmatrix = confusion_matrix(Y_validation, prediction)
cr=classification_report(Y_validation, prediction)

le.fit(Y_validation)
list(le.classes_)
y_=le.transform(Y_validation) 
p_=le.transform(prediction)
area=roc_auc_score(y_,p_)


print('Training Accuracy')
print(cv_results.mean())
print('Testing Accuracy')
print(a)
print('Confusion Matrix')    
print(confusionmatrix)    
print('Report')    
print(cr)   
print('AUC ROC Score')    
print(area)





nb=GaussianNB()
nb.fit(X_train_PC_s, Y_train_s)
prediction=nb.predict(X_validation_PC)
kfold = model_selection.KFold(n_splits=10, random_state=seed)
cv_results = model_selection.cross_val_score(nb, X_train_PC_s, Y_train_s, cv=kfold, scoring=scoring)
a=accuracy_score(Y_validation, prediction)
confusionmatrix = confusion_matrix(Y_validation, prediction)
cr=classification_report(Y_validation, prediction)

le.fit(Y_validation)
list(le.classes_)
y_=le.transform(Y_validation) 
p_=le.transform(prediction)
area=roc_auc_score(y_,p_)


print('Training Accuracy')
print(cv_results.mean())
print('Testing Accuracy')
print(a)
print('Confusion Matrix')    
print(confusionmatrix)    
print('Report')    
print(cr)   
print('AUC ROC Score')    
print(area)

   
     


pred_data=pd.read_csv('D:/Shyam/Datasets/PM/test_2umaH9m.csv')


pred_data=impute_unknowns(df=pred_data,column='education')


pred_data=impute_unknowns(df=pred_data,column='previous_year_rating')
dataset_X_pred_dummy = pd.get_dummies(pred_data)
X_p=dataset_X_pred_dummy.values
X_pred = scale(X_p)


X_pred=pca.fit_transform(X_pred)

output=rfc.predict(X_pred)
output=pd.DataFrame(output)
empid=pred_data['employee_id']
empid=pd.DataFrame(empid)
output=pd.concat([empid,output],ignore_index=True)


output.to_csv('D:/Shyam/Datasets/PM/output.csv',index=False)
