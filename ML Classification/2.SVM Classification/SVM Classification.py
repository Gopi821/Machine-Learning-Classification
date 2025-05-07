
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

dataset = pd.read_csv(r'C:\Users\Gopi Reddy\NIT7PM\APR\logit classification.csv')

x = dataset.iloc[:,[2,3]].values
y = dataset.iloc[:, -1].values

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y, test_size=0.20,random_state=0)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()

x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

from sklearn.svm import SVC
classifier = SVC()
classifier.fit(x_train, y_train)
y_pred = classifier.predict(x_test)

# Confusion Matrix

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)

from sklearn.metrics import accuracy_score
ac = accuracy_score(y_test, y_pred)
print(ac)

bias = classifier.score(x_train, y_train)
print(bias)

variance = classifier.score(x_test,y_test)
print(variance)

from sklearn.metrics import classification_report
cr = classification_report(y_test, y_pred)
print(cr)


