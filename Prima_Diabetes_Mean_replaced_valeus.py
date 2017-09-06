
# coding: utf-8

# In[6]:

#IMPORTING THE BASIC LIBRARIES
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import gc
import matplotlib.pyplot as plt
import seaborn as sns
#matplotlib inline


# In[5]:

INPUT_FOLDER='/Users/as186194/Documents/DOCUMENTS/TRIALS/Kaggle/Kaggle_Prima/pima-indians-diabetes-database/'
print ('File Sizes:')
for f in os.listdir(INPUT_FOLDER):
    if 'zip' not in f:
       print (f.ljust(30) + str(round(os.path.getsize(INPUT_FOLDER +  f) / 1000, 2)) + ' KB')


# In[7]:

#CREATING A DATAFRAME FOR THE MAIN FILE TO BE USED IN THE CODE
main_file=pd.read_csv(INPUT_FOLDER + 'diabetes.csv')
main_file.shape


# In[8]:

main_file.describe()


# In[9]:

main_file.head()


# In[10]:

#COUNTING THE PEOPLE WITH AND WITHOUT DIABETES
main_file.groupby("Outcome").size()


# In[11]:

main_file.hist(figsize=(10,8))
plt.figure()
plt.show()


# ###### Replacing '0' values of the columns mentioned below with their respective column mean.

# * BMI
# * BLOOD PRESSURE
# * GLUCOSE

# In[39]:

bmi_mean=main_file["BMI"].mean(skipna= True)
main_file=main_file.replace({'BMI': {0: bmi_mean}}) 


# In[44]:

bp_mean=main_file["BloodPressure"].mean(skipna= True)
main_file=main_file.replace({'BloodPressure': {0: bp_mean}}) 


# In[51]:

glu_mean=main_file["Glucose"].mean(skipna= True)
main_file=main_file.replace({'Glucose': {0: glu_mean}}) 


# In[58]:

main_file.describe()


# In[59]:

main_file.hist(figsize=(10,8))
plt.figure()
plt.show()


# In[60]:

get_ipython().magic('matplotlib inline')
main_file.plot(kind= 'box' , subplots=True, layout=(3,3), sharex=False, sharey=False, figsize=(10,8))


# In[61]:

import seaborn as sns
sns.set(style="ticks")
sns.pairplot(main_file, hue="Outcome")


# ###### Separating the data into Train & Test (80/20 split)

# In[91]:

X = main_file.ix[:,0:8]
Y = main_file["Outcome"]
from sklearn import model_selection
X_train, X_test, Y_train, Y_test= model_selection.train_test_split(X, Y, test_size=0.2)


# In[92]:

len(X_train)


# In[93]:

len(X_test)


# In[94]:

len(Y_train)


# In[95]:

len(Y_test)


# ###### Importing different models to check for the best accuracy 

# In[105]:

from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn import model_selection
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier


models = []
models.append(('LR', LogisticRegression()))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('RF', RandomForestClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC()))


# In[106]:

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


# ###### Visualizing the different model accuracies using a box plot

# In[107]:

ax = sns.boxplot(data=results)
ax.set_xticklabels(names)


# In[111]:

#FITTING THE LDA MODEL ON THE TEST DATASET
lda = LinearDiscriminantAnalysis()
lda.fit(X_train,Y_train)
predictions_lda = lda.predict(X_test)


# In[112]:

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score


# In[113]:

print("Accuracy Score is:")
print(accuracy_score(Y_test, predictions_lda))
print()


# In[104]:

print("Classification Report:")
print(classification_report(Y_test, predictions_lda))


# ###### Creating a Confusion Matrix

# In[114]:

conf = confusion_matrix(Y_test,predictions_lda)


# In[115]:

conf


# In[116]:

label = ["0","1"]
sns.heatmap(conf, annot=True, xticklabels=label, yticklabels=label)

