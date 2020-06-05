# EODClassifier
A binary classifier based on ensemble of decision of individual features.

## Pre-requisites 
To use this EODclassifier few libraries/modules need to be pre-installed such as -
NumPy, Pandas and sklearn (for metrics and performance analysis)

## Installation
To use the classifier you are simply required to dowload the **patana.py** file.
Keep this file in the same working directory where your program is running in.

## Usage
To import the Classifier file write -
> from patana import EODClassifier

To create object of EODClassifier class -
> eod = EODClassifier()

For training the dataset use the **.fit** function and similarly for testing use the **.predict** function.
> eod.fit(X_train, y_train)

> eod.predict(X_test)


### Parameters
There are two parameters available in this classifier i.e. **p(degree)** and **nof(number of feautures)**. To use these parameters put your desired values of degree and/or nof.
For Example:
> eod = EODClassifier(p=2, nof=5)

By default, **p=1** and **nof='all'** which means it takes all the features for training and prediction. You can also use **nof='half'** to take top 50% features. Also, you can mention how many features you want and it will take that many features with the highest fitness values.

## Sample code snippet 
A ready to run code snippet for beginner's / quick understanding.
```
from sklearn import datasets
from patana import EODClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

dataset = datasets.load_breast_cancer()
X = dataset.data
y = dataset.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

eod = EODClassifier()
eod.fit(X_train,y_train)
y_pred = eod.predict(X_test)   

print(accuracy_score(y_test,y_pred)) 
```

## How to Contribute
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update tests as appropriate.
