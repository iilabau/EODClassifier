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