![Titanic](https://raw.githubusercontent.com/Masterx-AI/Project_Titanic_Survival_Prediction_/main/titanic.jpg)
# The Unsinkable Sinks !!!
On April 15th, 1912, the so called **unsinkable** RMS Titanic sunk after colliding with an iceberg, leading to one of the world's most fatal shipwrecks. I collected the data from [Kaggle](https://www.kaggle.com/datasets/yasserh/titanic-dataset) and created a Machine Learning Model that will predict whether a person bearing the details provided as user-input had survived or not.<br/>


# Approach I took to deal with the dataset
I found several missing values under the **Cabin** column almost 77% of the data is reported to be missing... The **PassengerId** column did nothing but provided serial number to our dataset, the **Name** column provided unique name of each of the passenger similarly the **Ticket** column provided unique ticket number, these are just an identifier but not a significant contributor, also the **Embarked** column denotes the embarkment of passengers from 3 different locations, again, a noisy column.. The above columns clearly can not contribute to predict whether a person survived or not... and are not considered in training our model and also for prediction<br/>
The feature **Sbisp** denotes number of siblings/spouses that accompany a respective passenger similarly **Parch** denotes number of parent/children the indivisual is travelling with. These features had < 0.05 importance to the model, but after combining these features into **FamilySize**, where **FamilySize = Sibsp + Parch + 1**, the model gave 0.08 importance to FamilySize. The **Survived** column is our required label. Thus, we know all possible features and also all the possible values of the labels, thus we need **Supervised Machine Learning** here.
<br/>
The cardinal number of passengers are plotted with respect to the above features :
![Hist](https://github.com/debayanroy676/titanic_machine_learning/blob/master/graphs/feature_histogram.jpg?raw=true)
</br>

# Notably Important features and trends
The **Pclass**, **Sex**,  **Age**, and **Fare** had been the deciding factor here. Pclass represents the class of the passenger (1st class, 2nd class or 3rd class) and Fare denotes the amount paid by the passenger to buy tickets. The graphs represents that 1st class passengers and ones who paid higher price for tickets had more survival probability, also more female passengers survived as compared to males.. <br/>
![Plot](https://github.com/debayanroy676/titanic_machine_learning/blob/master/graphs/pair-plot.jpg?raw=true)
</br>

# Solution description :
I used :
- **Python** programming language
- **Sci-Kit learn** module
- **StandardScaler** for scaling the data
- **SimpleImputer** with *strategy="median"* to assign the median of the particular feature to any blank data encountered
- **RandomForestClassifier** + **LogisticRegression** hybrid
- **cross validation** to ensure model doesnot overfit

# Problems Encountered :
I faced these problems that are enlisted below : 
- Firstly, I tried using only RandomForestClassifier but the cross_val_score was not satisfactory.
- I adjust cross_validation parameters and also tried hyper-parameter tuning and adjusted n_estimators to 70, but the problem was still unsatisfied.
- I tried Feature Engineering by combining *Sibsp* and *Parch* to *FamilySize* as told above, still the score didn't tend to improve.
- Lastly, I tried a hybrid of RandomForestClassifier and LogisticRegression which did **NOT** improved the situation. The final cv score is stuck at nearly 81% for training and testing.
- **Conclusion :**  *i might have reached the ceiling of this dataset, and i selected deployment anyways. But i will return and try to improve it as i learn more about Machine Learning.*

# Files :
- [titanic.py](https://github.com/debayanroy676/titanic_machine_learning/blob/master/titanic.py) creates, trains and tests the model and dumps model.joblib making it ready for the CLI ([UI.py](https://github.com/debayanroy676/titanic_machine_learning/blob/master/UI.py))
- [UI.py](https://github.com/debayanroy676/titanic_machine_learning/blob/master/UI.py) serves as the Command Line Interface that accepts User Input and gives the predicted output.

# Usage :
```bash
git clone https://github.com/debayanroy676/titanic_machine_learning.git
cd titanic_machine_learning
#if sklearn and pandas are not installed
pip install scikit-learn
pip install pandas
#else directly run
python titanic.py
python UI.py
```
---





