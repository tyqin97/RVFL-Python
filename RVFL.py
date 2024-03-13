import numpy as np
import scipy.special as sp
from math import sqrt

from sklearn import metrics
import sklearn.datasets as skd
import sklearn.model_selection as skm
import sklearn.preprocessing as skp

from dataclasses import asdict, astuple, dataclass, field, replace

# std = skp.MinMaxScaler(feature_range=(0,1))

# def load_diabetes():
#     dataset = skd.load_diabetes()
#     X = std.fit_transform(dataset['data'])
#     T = dataset['target'].reshape(-1, 1)
    
#     X1, X2, T1, T2 = skm.train_test_split(X, T, test_size=0.3)
    
#     return X1, X2, T1, T2

@dataclass
class Model:
    Weight: np.ndarray = field(default_factory=list)
    Biases: np.ndarray = field(default_factory=list)
    Beta: np.ndarray = field(default_factory=list)
    
class ActivationFunction:
    @staticmethod
    def sigmoid(x):
        return sp.expit(x)
    
    @staticmethod
    def brelu(x):
        return np.where(x <= 0, 0, np.where(x > 1, 1, x))
    
    @staticmethod
    def tanh(x):
        return np.tanh(x)
    
    @staticmethod
    def softmax(x):
        return sp.softmax(x)
    
    @staticmethod
    def relu(x):
        return np.maximum(0, x)
    
    @staticmethod
    def binary(x):
        return np.where(x <= 0, -1, 1)

class RVFL:
    def __init__(self, n_nodes, lmbdas, act_func, task):
        task_list = ['regression', 'classification']
        acfn_list = ['sigmoid', 'brelu', 'tanh', 'softmax', 'relu', 'binary']

        assert task.lower() in task_list, 'Task Type Not Defined!'
        assert act_func.lower() in acfn_list, 'Activation Function Not Found!'
        
        self.model = Model()
        
        self.n_nodes = n_nodes
        self.lmbdas = lmbdas
        self.act_func = getattr(ActivationFunction(), act_func)
        self.task = task
        
    def train(self, X1, T1):
        self.D = None
        self.H = None
        n_sample, n_feature = X1.shape
        
        self.model.Weight, self.model.Biases = \
            self.generateNodes(self.lmbdas, n_feature, self.n_nodes)
        
        self.H = self.act_func(X1 @ self.model.Weight + self.model.Biases)
        self.D = self.combineD(self.H, X1)
        
        self.model.Beta = self.calcBeta(T1, self.D)
        
        resErr, Y1 = self.calcYresult(self.model.Beta, T1, self.D)
        score = self.calcRMSE(resErr, n_sample)
        
        if self.task == "classification":
            acc = self.calcAccuracy(T1, Y1)
            return {"RMSE" : score, "Accuracy" : acc}
        return {"RMSE" : score}
    
    def predict(self, X2, T2):
        n_sample, n_feature = X2.shape
        
        H2 = self.act_func(X2 @ self.model.Weight + self.model.Biases)
        D2 = self.combineD(H2, X2)
        
        resErr2, Y2 = self.calcYresult(self.model.Beta, T2, D2)
        score = self.calcRMSE(resErr2, n_sample)
        
        if self.task == "classification":
            acc = self.calcAccuracy(T2, Y2)
            return {"RMSE": score, "Accuracy": acc}
        return {"RMSE": score}
        
    def generateNodes(self, lmbda, n_feature, n_nodes):
        W = lmbda * (2 * np.random.rand(n_feature, n_nodes) - 1)
        b = lmbda * (2 * np.random.rand(1, n_nodes) - 1)
        return W, b
    
    @staticmethod
    def combineD(H, X):
        return np.concatenate([np.ones_like(X[:,0:1]), H, X], axis=1)
    
    def calcBeta(self, T1, D):
        return np.linalg.pinv(D) @ T1
    
    @staticmethod
    def calcYresult(Beta, T1, D):
        Y = D @ Beta
        resErr = Y - T1
        return resErr, Y
    
    @staticmethod
    def calcRMSE(resErr, n_sample):
        return sqrt(np.sum(np.sum(resErr ** 2, axis=0) / n_sample, axis=0))
    
    @staticmethod
    def calcAccuracy(T1, Y1):
        Y1 = np.argmax(Y1, axis=1)
        T1 = np.argmax(T1, axis=1)
        return metrics.accuracy_score(T1, Y1)
    
# if __name__ == "__main__":
#     X1, X2, T1, T2 = load_diabetes()
#     M = RVFL(500, 2, 'sigmoid', 'regression')
    
#     train_result = M.train(X2, T2)
#     valid_result = M.predict(X1, T1)
    
#     print(train_result, valid_result)