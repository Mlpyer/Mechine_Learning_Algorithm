import numpy as np
from sklearn import datasets
# from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import KFold

class KNN:
    def __init__(self,k,X,y):
        self.k = k
        self.train_data = X
        self.train_lable = y


    def distence_calculation(self,sample):
        distences = list()
        for train in self.train_data:
            res = np.sqrt(np.sum((train - sample) ** 2))
            distences.append(res)
        return distences
    
    def majority_label(self,topk_index):
        label_freq_dict = dict()    
        for index in topk_index:
            if self.train_lable[index] in label_freq_dict:
                label_freq_dict[self.train_lable[index]] += 1
            else:
                label_freq_dict[self.train_lable[index]] = 1
    
        majority_label = max(label_freq_dict.items(), key=lambda x: x[1])[0]
        return majority_label


    
    def predict(self,X_test,y_test):
        y_pred = list()
        for sample in X_test:
            distence =  self.distence_calculation(sample)
            topk_indexs = np.argsort(distence)[:self.k] 
            label = self.majority_label(topk_indexs)
            y_pred.append(label)
        #print(y_test)
        #print(y_pred)
        #print('acc = {}'.format(accuracy_score(y_test, y_pred)))
        return accuracy_score(y_test,y_pred)

        
    
def accuracy_score(y_true, y_pred):
    correct = (y_pred == y_true).astype(int)
    return np.average(correct)


def process_features(X):
    scaler = MinMaxScaler(feature_range=(0,1))
    X = scaler.fit_transform(1.0*X)
    return X


iris = datasets.load_iris()
X = iris['data']
y = iris['target']
#X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=None)


#X_train = process_features(X_train)
#X_test = process_features(X_test)



#model = KNN(k=6,X=X_train,y=y_train)
#model.predict(X_test, y_test)


KF = KFold(n_splits=10,random_state=None)
res = 0
for k in range(1,71):
     for train,test in KF.split(X):
          X_train,X_test = X[train],X[test]
          y_train,y_test = y[train],y[test]
          process_features(X_train)
          process_features(X_test)
          model = KNN(k = k,X=X_train,y=y_train)
          res += model.predict(X_test,y_test)
     print('k = ',k)
     print('average accuracy = ',res/10)
     res = 0