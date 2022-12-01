from sklearn.feature_selection import SelectFwe, f_classif
import pickle
import pandas
from sklearn.model_selection import train_test_split

# Load pickle encode dataset
filename = "C:/dataset/encoded_dataset.pkl"
infile = open(filename,'rb')
dataset = pickle.load(infile)
infile.close()

# Train test splitting
train, test = train_test_split(dataset, test_size=0.3, random_state=42)

# Define X and y
X_train = train[:,1:len(train)]
y_train = train[:,0]
X_test = test[:,1:len(test)]
y_test = test[:,0]

# Convert y type from float to integer
y_train = y_train.astype('int')
y_test = y_test.astype('int')

# Select variables with ANOVA
selection = SelectFwe(f_classif, alpha=0.001)
df_train = pandas.DataFrame(train)
variables = df_train[df_train.columns[:384]]
X_train_selected = selection.fit_transform(variables, df_train[0])

# 310 variables selected
print(X_selected.shape)
print(selection.scores_)
print(selection.pvalues_)
print(df_train.columns[:384][selection.pvalues_<0.001])

# Selection for test dataset
df_test = pandas.DataFrame(test)
X_test_selected = selection.transform(df_test[df_test.columns[:384]])
print(X_test_selected.shape)

# Save final train / test dataset
pickle.dump(X_train_selected, open("C:/model/train_selected.pkl", 'wb'))
pickle.dump(X_test_selected, open("C:/model/test_selected.pkl", 'wb'))
pickle.dump(y_train, open("C:/model/ytrain_selected.pkl", 'wb'))
pickle.dump(y_test, open("C:/model/ytest_selected.pkl", 'wb'))

# Save var_selection
joblib.dump(value=selection, filename='select_variable.pkl')


