from joblib import load
pclass_ = float(input("Enter the Passenger Class\n1 for First Class\n2. for Second Class\nor 3. for Third Class\nEnter your choise : "))
sex, pclass = 0, 0
if pclass_ == 1: pclass = 1
elif pclass_ == 2: pclass = 2
elif pclass_ == 3: pclass = 3
else: print("Invalid Input...."); exit()
sex_ = int(input("Enter the Gender\n1 for Male\n0 for Female\nEnter your choise : "))
if sex_ == 1: sex = 1
elif sex_ == 0: sex = 0
else: print("Invalid Input...."); exit()
age = float(input("Enter the Age of the Passenger : "))
if age < 1: print("Invalid Input...."); exit()
fare = float(input("Enter the Fare paid by the Passenger : "))
family_size = int(input("Enter the Family Size of the Passenger : "))
if family_size < 1: print("Invalid Input...."); exit()
features = [pclass, sex, age, fare, family_size]
my_pipeline = load("my_pipeline.joblib")
features = my_pipeline.transform([features])
model = load("titanic.joblib")
if model.predict(features) == 1: print("Passenger Survived")
else: print("Passenger did not Survive")