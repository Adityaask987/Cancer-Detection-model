#!/usr/bin/env python
# coding: utf-8

# In[1]:


# !pip install medmnist


# In[2]:


import medmnist
import numpy as np


# In[3]:


# from medmnist import pneumoniaMNIST3D
# dataset1 = pneumoniaMNIST3D(split="train", download=True)
# dataset2 = pneumoniaMNIST3D(split="val", download=True)
# dataset3 = pneumoniaMNIST3D(split="test", download=True)


# In[4]:


data=np.load('pneumoniamnist.npz')


# In[5]:


data.files


# In[6]:


lst=data.files 
xtrain=np.array(data[lst[0]])
xtrainval=np.array(data[lst[1]])
xtest=np.array(data[lst[2]])
ytrain=np.array(data[lst[3]])
ytrainval=np.array(data[lst[4]])
ytest=np.array(data[lst[5]])


# In[7]:


print(xtrain.shape)
print(xtrainval.shape)
print(xtest.shape)


# In[8]:


# combining all three dataset and shuffling them randomly

dim1=xtrain.shape[0]+xtrainval.shape[0]+xtest.shape[0]
dim2=xtrain.shape[1]
x=np.zeros((dim1,dim2,dim2,1))
y=np.zeros((dim1,1))

dim=xtrain.shape[0]+xtrainval.shape[0]

x[0:xtrain.shape[0],:,:,0]=xtrain
x[xtrain.shape[0]:dim,:,:,0]=xtrainval
x[dim:,:,:,0]=xtest
y[0:xtrain.shape[0]]=ytrain
y[xtrain.shape[0]:dim]=ytrainval
y[dim:]=ytest


# In[9]:


import tensorflow as tf 
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Flatten, MaxPool2D
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import minmax_scale


# In[10]:


xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.15)


# In[11]:


model=Sequential()

model.add(Conv2D(32,kernel_size=(3,3),activation='relu', input_shape=(28,28,1)))
model.add(Conv2D(32,kernel_size=(3,3),activation='relu'))
model.add(MaxPool2D(pool_size=(2,2)))

model.add(Conv2D(32,kernel_size=(3,3),activation='relu'))
model.add(Conv2D(32,kernel_size=(3,3),activation='relu'))
model.add(MaxPool2D(pool_size=(2,2)))

model.add(Flatten())

model.add(Dense(128,activation='relu'))
model.add(Dense(64,activation='relu'))
model.add(Dense(1,activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(xtrain,ytrain,batch_size=128, epochs=30, validation_split=0.1)
ypred=model.predict(xtest)


# In[12]:


Ypred=np.zeros(len(ypred))
for i in range(len(ypred)):
    if ypred[i]>=0.5:
        Ypred[i]=1
    else:
        Ypred[i]=0


# In[14]:


accuracy_score(ytest,Ypred)


# In[15]:


# model.save('pneumonia.h5')


# In[16]:


import sklearn.metrics as metrics
# calculate the fpr and tpr for all thresholds of the classification
fpr, tpr, threshold = metrics.roc_curve(ytest, ypred)
roc_auc = metrics.auc(fpr, tpr)

# method I: plt
import matplotlib.pyplot as plt
plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()


# In[ ]:



#!/usr/bin/env python
# coding: utf-8

# Import necessary libraries
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, roc_auc_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Flatten, MaxPool2D
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier

# Load data
data = np.load('pneumoniamnist.npz')
lst = data.files 
xtrain = np.array(data[lst[0]])
xtrainval = np.array(data[lst[1]])
xtest = np.array(data[lst[2]])
ytrain = np.array(data[lst[3]])
ytrainval = np.array(data[lst[4]])
ytest = np.array(data[lst[5]])

# Combine datasets and shuffle them
dim1 = xtrain.shape[0] + xtrainval.shape[0] + xtest.shape[0]
dim2 = xtrain.shape[1]
x = np.zeros((dim1, dim2, dim2, 1))
y = np.zeros((dim1, 1))

dim = xtrain.shape[0] + xtrainval.shape[0]
x[0:xtrain.shape[0], :, :, 0] = xtrain
x[xtrain.shape[0]:dim, :, :, 0] = xtrainval
x[dim:, :, :, 0] = xtest
y[0:xtrain.shape[0]] = ytrain
y[xtrain.shape[0]:dim] = ytrainval
y[dim:] = ytest

# Split data into training and testing sets
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.15)

# Function to create a Keras model
def create_model(optimizer='adam'):
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)))
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu'))
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model

# Creating KerasClassifier
keras_model = KerasClassifier(build_fn=create_model, epochs=30, batch_size=128, verbose=0)

# Defining hyperparameters to tune
param_grid = {
    'optimizer': ['adam', 'sgd', 'rmsprop'],
}

# Performing grid search
grid = GridSearchCV(estimator=keras_model, param_grid=param_grid, cv=3)
grid_result = grid.fit(xtrain, ytrain)

# Summarizing results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']

for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))

best_model = grid_result.best_estimator_.model

# Evaluating
ypred = best_model.predict(xtest)
Ypred = np.round(ypred)  # Convert probabilities to binary predictions

# Calculating and printing accuracy and ROC AUC score
accuracy = accuracy_score(ytest, Ypred)
roc_auc = roc_auc_score(ytest, ypred)
print("Test Accuracy:", accuracy)
print("Test ROC AUC:", roc_auc)

