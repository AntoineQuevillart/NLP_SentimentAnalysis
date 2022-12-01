import pickle
import joblib
from sklearn.svm import SVC
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import metrics

# Load pickle encoded dataset
filename_train = "C:/model/train_selected.pkl"
filename_test = "C:/model/test_selected.pkl"
filename_ytrain = "C:/model/ytrain_selected.pkl"
filename_ytest = "C:/model/ytest_selected.pkl"

infile_train = open(filename_train,'rb')
infile_test = open(filename_test,'rb')
infile_ytrain = open(filename_ytrain,'rb')
infile_ytest = open(filename_ytest,'rb')

train = pickle.load(infile_train)
test = pickle.load(infile_test)
y_train = pickle.load(infile_ytrain)
y_test = pickle.load(infile_ytest)

infile_train.close()
infile_test.close()
infile_ytrain.close()
infile_ytest.close()

# Convert y type from float to integer
y_train[y_train == "positive"] = 1
y_train[y_train == "negative"] = 0
y_test[y_test == "positive"] = 1
y_test[y_test == "negative"] = 0

y_train = y_train.astype('int')
y_test = y_test.astype('int')

# SVM Model
classifierSVM = SVC(C = 1, kernel = 'rbf', gamma = 0.5, probability = True, random_state = 0)
classifierSVM.fit(train, y_train)

# Prediction
y_pred = classifierSVM.predict(test)
y_pred = svm_model.predict(test)


# Model Accuracy: how often is the classifier correct?
print("Accuracy:", metrics.accuracy_score(test, y_pred))
# 81,68% accuracy

# Save the SVM classifier with joblib
joblib.dump(value=classifierSVM, filename='scoring_model.pkl')

