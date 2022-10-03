# -*- coding: utf-8 -*-

import numpy as np 
import pandas as pd 
import pickle 
import seaborn as sns
data = sns.load_dataset("iris")
dataset = pd.DataFrame(data)
x = dataset.drop(["species"], axis = 1)
y = dataset["species"]
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x,y, train_size = 0.7, test_size = 0.3, random_state = 1)
from sklearn.ensemble import RandomForestClassifier
reg = RandomForestClassifier()
classifier_rf = RandomForestClassifier(n_jobs = -1,max_depth = 2, n_estimators= 100, oob_score = True)
reg.fit(X_train,y_train)
pickle.dump(reg,open("model.pkl","wb"))
model = pickle.load(open('model.pkl',"rb"))