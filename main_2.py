#!/usr/bin/env python
# coding: utf-8

# In[1]:


# !pip install medmnist


# In[2]:


import medmnist
import numpy as np


# In[3]:


# from medmnist import PathMNIST
# dataset1 = PathMNIST(split="train", download=True)
# dataset2 = PathMNIST(split="val", download=True)
# dataset3 = PathMNIST(split="test", download=True)


# In[4]:


data=np.load('pathmnist.npz')


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
x=np.zeros((107180,28,28,3))
y=np.zeros((107180,1))
x[0:89996,:,:,:]=xtrain
x[89996:100000,:,:,:]=xtrainval
x[100000:,:,:,:]=xtest
y[0:89996]=ytrain
y[89996:100000]=ytrainval
y[100000:]=ytest


# In[9]:


y


# In[10]:


import tensorflow as tf 
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Flatten, MaxPool2D
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import minmax_scale


# In[11]:


from sklearn.preprocessing import OneHotEncoder
enc = OneHotEncoder()
y=enc.fit_transform(y).toarray()


# In[12]:


xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.15)


# In[13]:


model=Sequential()

model.add(Conv2D(32,kernel_size=(3,3),activation='relu', input_shape=(28,28,3),padding='same'))
model.add(Conv2D(32,kernel_size=(3,3),activation='relu',padding='same'))
model.add(MaxPool2D(pool_size=(2,2)))

model.add(Conv2D(32,kernel_size=(3,3),activation='relu',padding='same'))
model.add(Conv2D(32,kernel_size=(3,3),activation='relu',padding='same'))
model.add(MaxPool2D(pool_size=(2,2)))

model.add(Conv2D(32,kernel_size=(3,3),activation='relu',padding='same'))
model.add(Conv2D(32,kernel_size=(3,3),activation='relu',padding='same'))
model.add(MaxPool2D(pool_size=(2,2)))

model.add(Conv2D(32,kernel_size=(3,3),activation='relu',padding='same'))
model.add(Conv2D(32,kernel_size=(3,3),activation='relu',padding='same'))
model.add(MaxPool2D(pool_size=(2,2)))

model.add(Flatten())

model.add(Dense(1024,activation='relu'))
model.add(Dense(256,activation='relu'))
model.add(Dense(9,activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['categorical_accuracy'])
model.fit(xtrain,ytrain,batch_size=1024, epochs=10, validation_split=0.1)


# In[14]:


model.evaluate(xtest,ytest)


# In[15]:


from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV

# creating a Keras model
def create_model(optimizer='adam'):
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 3), padding='same'))
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same'))
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same'))
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same'))
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same'))
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same'))
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same'))
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same'))
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(1024, activation='relu'))
    model.add(Dense(256, activation='relu'))
    model.add(Dense(9, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['categorical_accuracy'])
    return model

# Creating KerasClassifier
keras_model = KerasClassifier(build_fn=create_model, epochs=10, batch_size=1024, verbose=0)

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


from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from tensorflow.keras.models import Sequential
from sklearn.preprocessing import OneHotEncoder
import numpy as np


# One-hot encode the labels
enc = OneHotEncoder()
y_one_hot = enc.fit_transform(y.reshape(-1, 1)).toarray()

model = Sequential()

n_splits = 5

kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

num_classes = len(np.unique(y))
overall_conf_matrix = np.zeros((num_classes, num_classes), dtype=int)

for train_index, test_index in kf.split(x, y):
    x_train, x_test = x[train_index], x[test_index]
    y_train, y_test = y_one_hot[train_index], y_one_hot[test_index]

    model.fit(x_train, y_train, epochs=10, batch_size=32, verbose=0)

    y_pred = np.argmax(model.predict(x_test), axis=1)

    conf_matrix = confusion_matrix(np.argmax(y_test, axis=1), y_pred, labels=np.unique(y))

    overall_conf_matrix += conf_matrix

accuracy = accuracy_score(np.argmax(y_one_hot, axis=1), cross_val_predict(model, x, y_one_hot, cv=kf))
classification_report_str = classification_report(np.argmax(y_one_hot, axis=1), cross_val_predict(model, x, y_one_hot, cv=kf))

print("Overall Confusion Matrix:")
print(overall_conf_matrix)
print("Accuracy:", accuracy)
print("Classification Report:")
print(classification_report_str)

from sklearn.metrics import multilabel_confusion_matrix
ypred=np.argmax(model.predict(xtest),axis=1)
Ytest=np.argmax((ytest),axis=1)
cm = multilabel_confusion_matrix(ypred, Ytest)


# In[16]:


model.save('path.h5')

#-----------heatmap
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
from sklearn.metrics import confusion_matrix

fig, axes = plt.subplots(nrows=1, ncols=cm.shape[0], figsize=(15, 5))

for i in range(cm.shape[0]):
    sns.heatmap(cm[i, :, :], annot=True, fmt='d', cmap='Blues', cbar=False, ax=axes[i])
    axes[i].set_title(f'Dimension {i+1}')
    
plt.show()
