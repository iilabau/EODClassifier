# EODClassifier
A binary classifier based on ensemble of decision of individual features.

## Pre-requisites 
To use our EOD classifier few libraries/modules needs to be pre-installed those are -
NumPy, Pandas and sklearn(for metrics and performance analysis)

## Installing
To use the classifier you are simply required to dowload the **patana.py** file on the same location as the program it is being used on.

## Usage
To import the Classifier file write -
> from patana import EODClassifier

For training the dataset use the **.fit** function and similarly for testing use the **.predict** function.
For more information about how to use the module see the **usage.py** file for more clear understanding.

### Parameters
There are two parameters available on our Classifier Degree(p) and nof(Number of feautures). To use the Parameter put your desired value of degree and/or nof.
For Example:
> eod = EODClassifier(p=2,nof=5)

By default degree(p)=1 and nof='all' which means it takes all the features for predicting the data. You can also use 'half' for nof to take half of all the features which have the most fitness value. Also you can mention how many features you want and it will take that many features with the highest fitness value.

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update tests as appropriate.

