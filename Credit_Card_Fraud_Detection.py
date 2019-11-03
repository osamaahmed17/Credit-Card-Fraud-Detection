import sys
import numpy
import pandas
import matplotlib
import seaborn
import scipy

print("Python",format(sys.version))
print("numpy",format(numpy.__version__))
print("pandas",format(pandas.__version__))
print("matplotlib",format(matplotlib.__version__))
print("seaborn",format(seaborn.__version__))
print("scipy",format(scipy.__version__))

#Importing the Specific Required Packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#Load the Dataset from using Pandas
data=pd.read_csv('../creditcard.csv')

#Exploring the Dataset
print(data.columns)

#V version represent PCA dimensional reductions which protects the sensitivity of the user's dataset, 
#it hides the confedentiality of the user

#In Class, the Zero would represents a valid credit card transaction and 1 would represent Fraudaluent credit card transaction
print(data.shape)

print(data.describe())

data = data.sample(frac=0.1, random_state=1) #To Save the computational power , we are going to use much smaller dataset

print(data.shape)

#Plot a histogram of each parameter
data.hist(figsize=(20,20))
plt.show()

# Determine number of fraudlent cases in the dataset
Fraud=data[data['Class']==1]
Valid=data[data['Class']==0]
    
outlier_fraction=len(Fraud)/float(len(Valid))
print(outlier_fraction)

print("Fraud Cases: {}".format(len(Fraud)))
print("Valid Cases:{}".format(len(Valid)))

#Building a coorelation Matrix to see if there is any strong relation between the variables, it  show that feature are important
cormat=data.corr()
fig=plt.figure(figsize=(12,9))
sns.heatmap(cormat,vmax=0.8,square=True)
plt.show()

#Get all the columns from the dataset
columns=data.columns.tolist()

#Filter the columns which we do not want
columns= (c for c in columns if c not in ["Class"])

#Store the variables we are predicting
target='Class'
X=data[columns]
Y=data[target]

#Print the shapes of X and Y
print(X.shape)  #It would be having class labels and data
print(Y.shape)  #It would be having class labels

from sklearn.metrics import classification_report, accuracy_score
from sklearn.ensemble import IsolationForest # Its gonna return isolation score of each sample by using Random Forest method
from sklearn.neighbors import LocalOutlierFactor #Its an unsupervised outlier detection method and calculates the anamoly score
#Of each sample.

#Define a Random state
state=1

#Define outlier detection Methods
classifiers={
    "Isolation Forest":IsolationForest(max_samples=len(X),contamination=outlier_fraction,random_state=state),
    "Local Outlier Factor":LocalOutlierFactor(n_neighbors=20,contamination=outlier_fraction) #The Higher the percentage of outlier, the higher the n_neigbours would be
}

#Fitting the models
n_outliers=len(Fraud)
for i , (clf_name,clf) in enumerate(classifiers.items()):
    #Fit the data and tag the outlier
    if(clf_name=="Local Outlier Factor"):
       y_pred= clf.fit_predict(X)
       scores_pred=clf.negative_outlier_factor_
    else:
       clf.fit(X)
       scores_pred= clf.decision_function(X)
       y_pred= clf.fit_predict(X)
       
    #Reshape the Prediction values to 0 for Valid and 1 for Fraud
    y_pred[y_pred==1]=0
    y_pred[y_pred==-1]=1
       
    n_errors= (y_pred!=Y).sum()
    
    #Run Classification Matrix
    print('{}:{}'.format(clf_name,n_errors))
       
    #For Printing Accuracy Score
    print(accuracy_score(Y,y_pred))
    print(classification_report(Y,y_pred))

