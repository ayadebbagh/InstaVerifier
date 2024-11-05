import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from igInfo import userInfo, username
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.utils.class_weight import compute_class_weight
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV


#TODO: if not downloaded, download -> read from downloaded csv
accountsDS = pd.read_csv('/Users/ayadebbagh/.cache/kagglehub/datasets/rezaunderfit/instagram-fake-and-real-accounts-dataset/versions/1/final-v1.csv')
accountsDS = accountsDS.drop("has_channel", axis=1)
accountsDS = accountsDS.drop("has_guides", axis=1)

#just to look at the correlation graph
corr = accountsDS.corr()
plt.figure(figsize = (15,5))
sns.heatmap(corr, annot = True)

#set X, the features, and y, the target
X = accountsDS.drop(columns = ['is_fake'])
y = accountsDS['is_fake']

#splitting into training and testing sets
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state = 42, train_size = 0.8)

#since there is an imbalance in the amount of fake vs real accounts in the dataset, I compute class weights to balance it out
classes = np.array([0, 1])
class_weights = compute_class_weight('balanced', classes=classes, y=train_y)
class_weight_dict = {0: class_weights[0], 1: class_weights[1]}

#creating the two models to be used 
forest_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', RandomForestClassifier(random_state=1, class_weight=class_weight_dict))
])

linear_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', LogisticRegression(max_iter=1000, class_weight=class_weight_dict))
])


#using a voting classifier to get the best predictions based on the 2 models used
voting_clf = VotingClassifier(estimators=[
    ('forest', forest_pipeline),
    ('linear', linear_pipeline)
], voting='soft')


#using different parameters for each model to optimize predictions, and then feeding the optimized params to the votingClassifier
param_grid = {
    'forest__classifier__n_estimators': [100, 200],
    'forest__classifier__max_depth': [None, 10, 20],
    'linear__classifier__C': [0.1, 1, 10]
}
grid_search = GridSearchCV(voting_clf, param_grid, cv=5, scoring='accuracy')
grid_search.fit(train_X, train_y)


best_model = grid_search.best_estimator_
fake_preds = best_model.predict(val_X)
cm = confusion_matrix(fake_preds, val_y)
scores = cross_val_score(best_model, X, y, cv=5)

#taking the new instagram user inputed by user
user_df = pd.DataFrame([userInfo])
user_df = user_df[X.columns].dropna(axis=1)
user_df = user_df.reindex(columns=X.columns, fill_value=0)


is_fake_prediction = best_model.predict(user_df)
print(f"The account '{username}' is predicted to be {'fake' if is_fake_prediction[0] else 'real'}.")