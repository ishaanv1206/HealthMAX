
import numpy as np
import matplotlib.pyplot as plt
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



print(classifier.predict([[0,0,0,25,1,0,0,1,0,0,0,3,0,0,0,0,7]]))