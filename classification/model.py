import pickle
import joblib
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn import metrics

# Load pickle encoded dataset
filename = "C:/dataset/encoded_dataset.pkl"
infile = open(filename,'rb')
dataset = pickle.load(infile)
infile.close()

# Train test splitting
train, test = train_test_split(dataset, test_size=0.2)

# Define X and y
X_train = train[:,1:len(train)]
y_train = train[:,0]
X_test = test[:,1:len(test)]
y_test = test[:,0]

# Convert y type from float to integer
y_train = y_train.astype('int')
y_test = y_test.astype('int')

# SVM model - 5 fold cross validation
clf_svm = svm.SVC(probability=True)
clf_svm.fit(X_train, y_train)

# Prediction
y_pred = clf_svm.predict(X_test)

# Model Accuracy: how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
# 78,8% accuracy

# Save the SVM classifier with joblib
filename = "C:/model/svm_classifier.joblib"
joblib.dump(clf_svm, filename)