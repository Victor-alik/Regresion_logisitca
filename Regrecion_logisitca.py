from os import sep
#%%
import pandas as pd 
import numpy as np 
df=pd.read_csv('bank.csv', sep=';')
df.head()
#%%
#Determinar el balance de la variable de respuesta
import matplotlib.pyplot as plt 
plt.bar(df.y.unique(),df.y.value_counts());
#%%
##Resampling:
#El submuestreo consiste en eliminar registros aleaotrios de la 
#calse mayorista (causa predidad e informacion)
#El sobremuestreo es duplicar registros aleatorios de la clase
#minorista (causa sobreajuste)

df_no=df[df['y']=='no']
df_yes=df[df['y']=='yes']

print('No_shape', df_no.shape)
print('Yes_shape', df_yes.shape)
#%%
df_no_reduced=df_no.sample(521,random_state=103)
df_no_reduced.shape
#%%
df_reduced=pd.concat([df_no_reduced,df_yes],axis=0)
df_reduced.head()
#%%
df_reduced.tail()
#%%
df_reduced=df_reduced.sample(frac=1,random_state=103) #Revolvemos los datos (si y no)
df_reduced.head()
#%%
df_reduced.shape
#%%
X=df_reduced['duration'].values.reshape(-1,1)
y=df_reduced['y']
#Creamos el split de datos
from sklearn.model_selection import train_test_split 
X_train,X_test,y_train,y_test=train_test_split(X,y, test_size=0.33,random_state=103)
#%%
print(X_train.shape)
print(X_test.shape)
#%%
from sklearn.linear_model import LogisticRegression 

clf=LogisticRegression().fit(X_train,y_train)
y_train_hat=clf.predict(X_train)
y_test_hat=clf.predict(X_test)
clf.score(X_train,y_train)
#%%
clf.score(X_test,y_test)
#%%
proba=clf.predict_proba(X_train)[:,0]
proba
#%%
plt.scatter(X_train,proba)
#%%
####Matriz de confusion ################
from sklearn.metrics import confusion_matrix 
import seaborn as sns 
labels=['yes','no']
cm=confusion_matrix(y_train,y_train_hat,labels=labels)
sns.heatmap(cm,annot=True,
            fmt='d',
            xticklabels=labels,
            yticklabels=labels,
            cmap='Greens')
plt.ylabel('Predicted')
plt.xlabel('ACtual')
#%%
TP,FP,FN,TN=cm.flatten()
print(TP,FP,FN,TN)
#%%
#recall o true positive rate (TPR)
TPR=TP/(TP+FN)
TPR
#%%
#Precision (PPV)
PPV=TP/(TP+FP)
PPV
#%%%
#Accuracy (ACC)
(TP+TN)/cm.sum()
#%%
#F_measure
F=(2*TPR*PPV)/(TPR+PPV) #media armonica entre TPR y PPV
F
#%%
from sklearn.metrics import classification_report 
from sklearn.metrics import accuracy_score
print(accuracy_score(y_test,y_test_hat))
print(classification_report(y_test,y_test_hat))