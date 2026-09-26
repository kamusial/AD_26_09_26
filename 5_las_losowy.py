import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

df = pd.read_csv("iris.csv")
print(df["class"].value_counts())

species = {
    "Iris-setosa":0, "Iris-versicolor":1, "Iris-virginica":2
}
df["class_value"] = df["class"].map(species)

X1 = df[['sepallength', 'sepalwidth']]
X2 = df[['petallength', 'petalwidth']]
X = df.iloc[:,:4]  # wszystkie cechy
y = df.class_value
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2)

model = RandomForestClassifier()   # criterion='entropy', max_depth=6, min_samples_split=2
model.fit(X_train, y_train)
print(model.score(X_test, y_test))
print(pd.DataFrame(confusion_matrix(y_test, model.predict(X_test))))
print(pd.DataFrame(model.feature_importances_, X.columns))

import joblib
model_new = joblib.dump(model, 'Kamil_random_forest.model')
