import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.externals import joblib

model = joblib.load('music-recommender.joblib')

predictions = model.predict([[21, 1]])
predictions
