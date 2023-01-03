
import pandas as pd

# load the model training data
data = pd.read_csv('fertility.csv')

# importing random forest classifier from assemble module
from sklearn.ensemble import RandomForestClassifier

# Create a Random forest Classifier
clf = RandomForestClassifier(n_estimators = 10)
# , max_depth = 3 , criterion="gini",min_samples_split = 3


# Import label encoder
from sklearn import preprocessing
  
# label_encoder object knows how to understand word labels.
label_encoder = preprocessing.LabelEncoder()

data['Season']= label_encoder.fit_transform(data['Season'])
data['childish_disease']= label_encoder.fit_transform(data['childish_disease'])
data['acc_trauma']= label_encoder.fit_transform(data['acc_trauma'])
data['surgical']= label_encoder.fit_transform(data['surgical'])
data['fever_lastyr']= label_encoder.fit_transform(data['fever_lastyr'])
data['alcohol']= label_encoder.fit_transform(data['alcohol'])
data['smoking']= label_encoder.fit_transform(data['smoking'])
data['result']= label_encoder.fit_transform(data['result'])

# upto last columns is X and prediction is y(last column)
X_train = data.iloc[:,:-1]
y_train = data.iloc[:,-1]

print(data)
# Train the model using the training sets
clf.fit(X_train, y_train)

# prediction 
y_predict = clf.predict(X_train)

# check for accuracy of model
from sklearn.metrics import accuracy_score
print(clf.feature_importances_)

print(accuracy_score(y_train, y_predict))

# create joblib file for django
from joblib import dump
dump(clf,'fm.pkl')