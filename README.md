# CNN Cancer Detection Models
This repository contains two Convolutional Neural Network (CNN) models for cancer detection using the PneumoniaMNIST and PathMNIST datasets. These models are implemented in Python using TensorFlow and Keras, and include code for data preprocessing, model training, and evaluation.

# Datasets
1. PneumoniaMNIST: A dataset for pneumonia detection from chest X-ray images.
2. PathMNIST: A dataset for colorectal cancer detection from pathology images.
Both datasets are part of the MedMNIST collection, which is a series of standardized biomedical image datasets.

# Files
## 1. main_1.py - PneumoniaMNIST Model
This script includes the following steps:

### Data Loading: Loads the PneumoniaMNIST dataset from a .npz file.
Data Preprocessing: Combines and shuffles the training, validation, and test sets.
Model Architecture: Defines a CNN model with several convolutional and pooling layers followed by dense layers.
Training: Trains the model using binary cross-entropy loss and the Adam optimizer.
Evaluation: Evaluates the model using accuracy and ROC AUC metrics. The script also includes code to plot the ROC curve.
Hyperparameter Tuning: Performs hyperparameter tuning using GridSearchCV on different optimizers.
## 2. main_2.py - PathMNIST Model
This script includes the following steps:

### Data Loading: Loads the PathMNIST dataset from a .npz file.
Data Preprocessing: Combines and shuffles the training, validation, and test sets, and performs one-hot encoding on the labels.
Model Architecture: Defines a deeper CNN model with additional convolutional layers and dense layers.
Training: Trains the model using categorical cross-entropy loss and the Adam optimizer.
Evaluation: Evaluates the model using categorical accuracy and confusion matrices. The script also includes code to plot heatmaps of confusion matrices.
Hyperparameter Tuning: Performs hyperparameter tuning using GridSearchCV on different optimizers.

# Results
The models achieve high accuracy on both datasets, demonstrating their effectiveness in cancer detection. The results of the models, including accuracy, ROC AUC score, and confusion matrices, are displayed at the end of each script.

# Acknowledgements
The PneumoniaMNIST and PathMNIST datasets are provided by the MedMNIST project.
TensorFlow and Keras are used for building and training the models.
