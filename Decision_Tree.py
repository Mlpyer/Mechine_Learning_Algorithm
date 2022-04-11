import numpy as np
from sklearn import datasets
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

class Node:
    j = None#特征
    theta = None#阈值
    left = None#左子树
    right = None#右子树
    
    
class Decision_tree:
    
    
    def Entropy(self,y,idx):
        feature_label_dict = dict()
        for i in idx:
            if y[i] in feature_label_dict:
                feature_label_dict[y[i]]+=1
            else:
                feature_label_dict[y[i]]=1
        res = 0
        for value in feature_label_dict.values():
            res -= value/len(idx)*np.log(value/len(idx))
        return res
    
    
    def Entropy_Increase(self,y,idx1,idx2):
        datasize = len(idx1) + len(idx2)
        
        res1 = len(idx1)/datasize*self.Entropy(y, idx1)
        res2 = len(idx2)/datasize*self.Entropy(y, idx2)
        
        return res1+res2
    
    
    def Entropy_Increase_Rate(self,y,idx1,idx2):
        
        datasize = len(idx1) + len(idx2)
        Entropy_Increase = self.Entropy_Increase(y, idx1, idx2)
        rate = -len(idx1)/datasize*np.log(len(idx1)/datasize)-len(idx2)/datasize*np.log(len(idx2)/datasize)
        return Entropy_Increase/rate
        
       
    def Gini(self,y,idx):
        feature_label_dict = dict()
        for i in idx:
            if y[i] in feature_label_dict:
                feature_label_dict[y[i]]+=1
            else:
                feature_label_dict[y[i]]=1
        res = 0
        for value in feature_label_dict.values():
            res += (value/len(idx))**2
        return 1-res
    
    
    def split_data(self,X,j,theta,idx):
        idx1 = list()
        idx2 = list()
        for i in idx:
            split = X[i][j]
            if split<=theta:
                idx1.append(i)
            else:
                idx2.append(i)
        return idx1,idx2
    
    def find_best_split(self,X,y,idx):
        nodeGini = self.Gini(y,idx)
        best_score = -float('inf')
        best_j = -1
        best_idx1 = list()
        best_idx2 = list()
        best_theta = float('inf')
        for j in self.feature_names:
            thetas = set(x[j] for x in X)
            for theta in thetas:
                idx1,idx2 = self.split_data(X,j,theta,idx)
                if(len(idx1)>=1 and len(idx2)>=1):
                    # score = self.Entropy_Increase(y,idx1,idx2)
                    datasize = len(idx1)+len(idx2)
                    score = len(idx1)/datasize*self.Gini(y,idx1)+len(idx2)/datasize*self.Gini(y,idx2)
                    if(nodeGini-score>=best_score):
                        best_j = j
                        best_idx1 = idx1
                        best_idx2 = idx2
                        best_theta = theta
                        best_score = nodeGini-score
        return best_j,best_idx1,best_idx2,best_theta

    

    def generate_tree(self,X,y,idx):
        r=Node()
        if self.Gini(y, idx)==0:
            r.left = None
            r.right = None
            r.j = y[idx[0]]
            r.theta = float('inf')
        # print('-------------------------------')
        # print('idx',idx)
        j,idx1,idx2,theta = self.find_best_split(X,y,idx)
        # print("fearture_name",iris.feature_names[j])
        # print('theta',theta)
        # print('idx1',idx1)
        # print('idx2',idx2)
        
        if j==-1:
            return r
        # self.feature_names.remove(j)
        r.j = j
        r.theta = theta         
        r.left = self.generate_tree(X, y, idx1)
        r.right = self.generate_tree(X, y, idx2)
        

        return r

    def fit(self,X,y):
        k,m = X.shape
        y_hat = self.convert_to_vectors(y)
        k,n = y_hat.shape
        self.feature_names = list(i for i in range(m))
        self.target_names = list(j for j in range(n))
        self.idx = list(p for p in range(k))
        self.root = self.generate_tree(X, y, self.idx)

    
    def get_prediction(self,r,x):
        if r.left == None or r.right == None:
            return r.j
        value = x[r.j]
        if value <= r.theta:
            return self.get_prediction(r.left, x)
        else:
            return self.get_prediction(r.right, x)
        
    def predict(self,X):
        y = list()
        for i in range(len(X)):
            y.append(self.get_prediction(self.root, X[i]))
        return y


    def convert_to_vectors(self,c):
        m = len(c)
        k = np.max(c) + 1
        y = np.zeros(m * k).reshape(m,k)
        for i in range(m):
            y[i][c[i]] = 1
        return y
    

def process_features(X):
    scaler = MinMaxScaler(feature_range=(0,1))
    X = scaler.fit_transform(1.0*X)
    return X

def accuracy_score(y_true, y_pred):
    correct = (y_pred == y_true).astype(int)
    return np.average(correct)

iris = datasets.load_iris()
X = iris['data']
y = iris['target']
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=0)
# X_train = process_features(X_train)
# X_test = process_features(X_test)
X_train = process_features(X_train)
X_test = process_features(X_test)

model = Decision_tree()
model.fit(X_train,y_train)
y_pred = model.predict(X_test)
print(accuracy_score(y_test, y_pred))