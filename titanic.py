from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import StackingClassifier
from joblib import dump
import pandas as pd
import numpy as np

titanic_data_master = pd.read_csv('data/Titanic-Dataset.csv')
titanic_data_master['Sex'] = titanic_data_master['Sex'].str.lower().replace({'male': 1, 'female': 0})
#Binary Value allocation for feature 'Sex'
titanic_data_master['FamilySize']=titanic_data_master['SibSp']+titanic_data_master['Parch']+1  
#^^^ Combining two less important features for better optimization ^^^
data_reduced = titanic_data_master.drop("PassengerId",axis=1).drop("Name",axis=1).drop("Ticket",axis=1).drop("Cabin",axis=1).drop("Embarked",axis=1)
#Dropping less important features
data_reduced = data_reduced.drop("SibSp",axis=1).drop("Parch",axis=1) #Feature Engineering
## SibSp and Parch are combined to form FamilySize feature to reduce number of features and improve model performance
train_set, test_set = train_test_split(data_reduced, test_size=0.10, random_state=69, stratify=data_reduced["Survived"])
x_train = train_set.drop("Survived",axis=1)
y_train = train_set["Survived"].copy()
x_test = test_set.drop("Survived",axis=1)
y_test = test_set["Survived"].copy()
my_pipeline = Pipeline([("imputer", SimpleImputer(strategy="median")),
                       ("scaler", StandardScaler())])
x_train = my_pipeline.fit_transform(np.array(x_train))
x_test = my_pipeline.transform(np.array(x_test))
model_rf = RandomForestClassifier()
model_lr = LogisticRegression()
model = StackingClassifier(
    estimators=[('random-forest', model_rf)],
    final_estimator=model_lr
) 
model.n_estimators=70
model.fit(x_train, y_train)
#print(f"Training C.V. score = {cross_val_score(model,x_train, y_train, cv=8).mean()}")
#print(f"Testing C.V. score = {cross_val_score(model, x_test, y_test, cv=8).mean()}")
dump(model, "titanic.joblib")
dump(my_pipeline, "my_pipeline.joblib")