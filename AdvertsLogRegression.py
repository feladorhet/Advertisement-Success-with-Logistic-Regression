import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

adverts = pd.read_csv("advertising.csv")

#This data set contains the following features:
#'Daily Time Spent on Site': consumer time on site in minutes
#'Age': cutomer age in years
#'Area Income': Avg. Income of geographical area of consumer
#'Daily Internet Usage': Avg. minutes a day consumer is on the internet
#'Ad Topic Line': Headline of the advertisement
#'City': City of consumer
#'Male': Whether or not consumer was male
#'Country': Country of consumer
#'Timestamp': Time at which consumer clicked on Ad or closed window
#'Clicked on Ad': 0 or 1 indicated clicking on Ad

##DATA EXPLORATION
##STARTS HERE

ageHist = sns.histplot(adverts["Age"])
age_areaIncome = sns.jointplot(data=adverts, x="Age", y="Area Income")
timeSpent_age = sns.jointplot(data=adverts, x="Age", y="Daily Time Spent on Site", kind="kde", color="red")
timeSpent_intUsage = sns.jointplot(data=adverts, x="Daily Time Spent on Site", y="Daily Internet Usage", color="green")
clickedOnAd = sns.pairplot(data=adverts, hue="Clicked on Ad", palette="bwr")
#plt.show()

##DATA EXPLORATION 
##ENDS HERE

##NOW TO TRAIN A LOGISTIC REGRESSION MODEL
X = adverts[["Daily Time Spent on Site", "Age", "Area Income","Daily Internet Usage", "Male"]]
y = adverts["Clicked on Ad"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=50)

logModel = LogisticRegression(max_iter=1000)
logModel.fit(X=X_train, y=y_train)

predictions = logModel.predict(X_test)

#THE CONFUSION MATRIX AND REPORTS TO SEE HOW WELL OUR MODEL DOES
confMat = confusion_matrix(y_true=y_test, y_pred=predictions)
classReps = classification_report(y_true=y_test, y_pred=predictions)

print("Confusion Matrix:")
print(confMat)
print("")
print("Classification Reports:")
print(classReps)
