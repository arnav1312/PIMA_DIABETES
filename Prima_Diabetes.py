
# coding: utf-8

# In[1]:

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import gc
import matplotlib.pyplot as plt
import seaborn as sns
#matplotlib inline


# In[7]:

INPUT_FOLDER='C:/Kaggle_Prima/pima-indians-diabetes-database/'
print ('File Sizes:')
for f in os.listdir(INPUT_FOLDER):
    if 'zip' not in f:
       print (f.ljust(30) + str(round(os.path.getsize(INPUT_FOLDER +  f) / 1000, 2)) + ' KB')


# In[19]:

main_file=pd.read_csv(INPUT_FOLDER + 'diabetes.csv')
main_file.shape


# In[17]:

main_file.describe()


# In[18]:

main_file.head()


# In[29]:

main_file.groupby("Outcome").size()


# In[28]:

main_file.hist(figsize=(10,8))
plt.figure()
plt.show()


# In[33]:

get_ipython().magic('matplotlib inline')
main_file.plot(kind= 'box' , subplots=True, layout=(3,3), sharex=False, sharey=False, figsize=(10,8))


# In[48]:

import seaborn as sns
sns.set(style="ticks")
sns.pairplot(main_file, hue="Outcome")


# In[64]:

X = main_file.ix[:,0:8]
Y = main_file["Outcome"]
from sklearn import model_selection
X_train, X_test, Y_train, Y_test= model_selection.train_test_split(X, Y, test_size=0.2)


# In[65]:

len(X_train)


# In[66]:

len(X_test)


# In[67]:

len(Y_train)


# In[68]:

len(Y_test)


# In[77]:

from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC

models = []
models.append(('LR', LogisticRegression()))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC()))


# In[81]:

results = []
names = []
for name,model in models:
    kfold = model_selection.KFold(n_splits=10)
    cv_result = model_selection.cross_val_score(model,X_train,Y_train, cv = kfold,scoring = "accuracy")
    kfold = model_selection.KFold(n_splits=10)
    names.append(name)
    results.append(cv_result)
for i in range(len(names)):
    print(names[i],results[i].mean())


# In[82]:

ax = sns.boxplot(data=results)
ax.set_xticklabels(names)


# In[83]:

lda = LinearDiscriminantAnalysis()
lda.fit(X_train,Y_train)
predictions_lda = lda.predict(X_test)


# In[84]:

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score


# In[88]:

print("Accuracy Score is:")
print(accuracy_score(Y_test, predictions_lda))
print()


# In[89]:

print("Classification Report:")
print(classification_report(Y_test, predictions_lda))


# In[94]:

conf = confusion_matrix(Y_test,predictions_lda)


# In[95]:

conf


# In[96]:

label = ["0","1"]
sns.heatmap(conf, annot=True, xticklabels=label, yticklabels=label)


# In[ ]:



