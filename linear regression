from sklearn.datasets import make_regression
import numpy as np
import matplotlib.pyplot as plt
 


class linear(object):
    def __init__(self):
        self.W = None
 
        
    def Compute_loss(self,X,y):
        self.W = ((X.T*X).I) * (X.T*y)
        return self.W
    
    def local_linear(self,test_point,X,y,k=1):
        m = X.shape[0]
        weights = np.mat(np.eye(m))
        for i in range(m):
            distance = test_point - X[i,1]
            weights[i,i] = np.exp(distance * distance.T/(-2.0 * k**2))
            self.W = ((X.T*weights*X).I) * (X.T*weights*y)
            
            return test_point * self.W[1] +self.W[0]
   
    def local_linear_predict(self,X,y,k = 1):
        m = X.shape[0]
        y_hat = np.zeros(m)
        for i in range(m):
            y_hat[i] = self.local_linear(X[i,1],X,y,k)
        print(y_hat)
        return y_hat    
      
    def generate_data(self):
        X,y,coef=make_regression(n_samples=50,n_features=1,noise=30,coef=True,bias = 100)
        plt.scatter(X,y,c='r',s=3)
        #plt.plot(X,X*coef+100,c='black') 
        
        X = np.mat(X)
        y = np.mat(y).T
        ones = np.mat(np.ones([X.shape[0],1]))
        X = np.concatenate((ones,X),1)


classify = linear()
classify.generate_data()
y_hat =  classify.local_linear_predict(X,y,1)
plt.plot(X[:,1].T.tolist()[0],y_hat ,c='g')         
plt.show() 
