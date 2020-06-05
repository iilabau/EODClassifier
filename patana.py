import numpy as np
import pandas as pd

# Ensemble of Decisions classifier
class EODClassifier:
    def __init__(self, nof='all', p=1):                
        self.nof = nof
        self.p = p
        self.m0 = []
        self.m1 = []
        self.threshold = []
        self.fitness = []

    def fit(self, X_train, Y_train):
        df=pd.DataFrame(np.column_stack((X_train, Y_train)))

        self.m0 = []
        self.m1 = []
        self.threshold = []
        self.fitness = []
       
        n = df.shape[1] - 1        
        dfs = dict(tuple(df.groupby(df.columns[n])))    
        class0=dfs[0]
        class1=dfs[1]
        self.m0=class0.mean(axis = 0, skipna = True)
        self.m1=class1.mean(axis = 0, skipna = True)
        sd0=class0.std(axis = 0, skipna = True)
        sd1=class1.std(axis = 0, skipna = True)
        for i in range(df.shape[1]-1):
            t=((self.m0[i]*sd1[i])+(self.m1[i]*sd0[i]))/(sd0[i]+sd1[i])
            self.threshold.append(t)
        for i in range(df.shape[1]-1):
            r=abs(self.m0[i]-self.m1[i])/(sd0[i]+sd1[i])
            self.fitness.append(r)
        if self.nof == 'all':
            self.nof = n
        if self.nof == 'half':
            self.nof = int(n/2)
        
        #select the best features
        fits = np.sort(np.array(self.fitness))        
        for i in range(n-self.nof):
            pos = np.argwhere(np.array(self.fitness)==fits[i])
            self.fitness[pos[0][0]] = 0.0            
            

    def predict(self, X_test):
        sum_of_fit=0
        for i in range(X_test.shape[1]):
            sum_of_fit+=np.power(self.fitness[i], self.p)
        
        predictClass=[]
        list2=[]
        for i in range(X_test.shape[0]):
            classVariable=0
            for j in range(X_test.shape[1]):            
                if(self.m0[j] < self.m1[j]):
                    if(X_test[i][j] <= self.threshold[j]):
                        list2.append(0)
                    else:
                        list2.append(1)
                else:
                    if(X_test[i][j] <= self.threshold[j]):
                        list2.append(1)
                    else:
                        list2.append(0) 
        finlist=np.array(list2).reshape(X_test.shape[0], X_test.shape[1])
        for i in range(X_test.shape[0]):
            sum1 = 0
            for j in range(X_test.shape[1]):            
                sum1 += finlist[i][j]*np.power(self.fitness[j],self.p)
            cls = sum1/sum_of_fit
            
            if(cls<0.5):
                predictClass.append(0)
            else:
                predictClass.append(1)
        
        return np.array(predictClass)