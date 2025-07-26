##Import the libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score

##Importing the dataset
# IMPORTANT: You'll need to upload 'breast_cancer.csv' to the same repository
# or provide a link to where it can be downloaded.
ds = pd.read_csv('breast_cancer.csv')

X = ds.iloc[:, 1:-1].values
y = ds.iloc[:, -1].values

## Splitting the dataset into the Training set and Test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

##Training the Logistic Regression model on the Training set
classifier = LogisticRegression()
model = classifier.fit(X_train, y_train)

##Predicting the Test set results
y_pred = model.predict(X_test)

##Making the Confusion Matrix/Accuracy Score
cm = confusion_matrix(y_test, y_pred)
accs = accuracy_score(y_test, y_pred)

print("Confusion Matrix:")
print(cm)
print("\nAccuracy Score:")
print(accs)

##Computing the accuracy with k-Fold Cross Validation

from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = model, X = X_train, y = y_train, cv = 10)
print("Accuracy: {:.2f} %".format(accuracies.mean()* 100))
print("Standard Deviation: {:.2f} %".format(accuracies.std()* 100))
