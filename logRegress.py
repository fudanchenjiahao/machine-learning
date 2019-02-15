import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets

class logistic(object):
    def __init__(self):
        self.W = None
        self.b = None
    def sigmoid(self,z):
        return 1/(1+np.exp(-z))
        

    
    def output(self,X):
        z = np.dot(X,self.W) + self.b
        return self.sigmoid(z)
    
    def Compute_loss(self,X,y):
        data_size = X.shape[0]
        a = self.output(X)
        total_loss = np.sum(-y*np.log(a)-(1-y)*np.log(1-a))
        loss = total_loss/data_size
        dw =  X.T*(a-(y.T)) / data_size
        db = np.sum(a-(y.T)) / data_size
        print(dw,db)
        return loss,dw,db
    
    def train(self,X,y,learn_rate = 0.01,num_iters =10000,gradient_type = "all",batch_size = 25):
        data_size,feature_num = X.shape
        self.W = 1 * np.random.randn(feature_num,1)
        self.b = 1 * np.random.randn(1)
   
#        self.W = np.array([[-0.1],[0.3]])
#        self.b = [0.6]
        loss_list = []
        if gradient_type == "all":
            for i in range(num_iters):
                loss,dw,db = self.Compute_loss(X,y)
                loss_list.append(loss)
                self.W -= learn_rate*dw
                self.b -= learn_rate*db
                print("i = %d,error = %f" %(i,loss))
            return loss_list
        
        elif gradient_type == "mini batch":
            for i in range(num_iters):
                k = 0
                while k < data_size:
                    mini_batch_X = X[k:k + batch_size,:]
                    mini_batch_y = y[:,k:k + batch_size]
                    k += batch_size
                    loss,dw,db = self.Compute_loss(mini_batch_X,mini_batch_y)
                    loss_list.append(loss)
                    self.W -= learn_rate*dw
                    self.b -= learn_rate*db
                    print("i = %d,error = %f" %(i,loss))
            return loss_list   
    def predict(self,X_test):
        a = self.output(X_test)
        y_pred = np.where(a>=0.5,1,0)
        return y_pred,a


    def generate_data(self):
        
        data,y=datasets.make_classification(n_samples=100, n_features=2, n_informative=2, n_redundant=0,n_repeated=0, n_classes=2, n_clusters_per_class=1)
        plt.scatter(data[:,1],data[:,0],c=y)
        data = np.mat(data)
        y = np.mat(y)
        plt.show() 
        return data,y    
    def decision_boundary(self):
        x0 = np.arange(-2,3,0.1)
        x1 = ( -classify.b - classify.W[0]*x0) / classify.W[1]
        plt.figure(1)
        plt.plot(x0,x1,color = 'red')
        plt.xlabel('X0')
        plt.ylabel('X1')
        plt.legend(loc = 'upper left')
     
classify = logistic()
data,y = classify.generate_data()
loss_list = classify.train(data,y,gradient_type = "mini batch")
classify.decision_boundary()


 

#plt.figure(2) 
#plt.plot(loss_list)
#plt.xlabel('Iteration number')
#plt.ylabel('Loss value')

#show the decision boundary






 

