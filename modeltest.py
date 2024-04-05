import numpy as np
import pandas as pd




dataset = pd.read_csv('diabetesdata.csv')
X = dataset.iloc[:, 1:18].values
y = dataset.iloc[:, 0].values

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


from sklearn.experimental import enable_hist_gradient_boosting  # noqa
from sklearn.ensemble import HistGradientBoostingClassifier

# Assuming X_train and y_train are your training data
classifier = HistGradientBoostingClassifier()
classifier.fit(X_train, y_train)

from sklearn.metrics import confusion_matrix, accuracy_score
y_pred = classifier.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
print(cm)

print(accuracy_score(y_test, y_pred))
print(classifier.predict([[ 1, 1, 1, 32, 1, 1, 0, 1, 0, 1, 0, 5, 0, 30, 1, 1, 8]]))