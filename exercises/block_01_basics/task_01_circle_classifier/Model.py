import numpy as np
import pandas as pd

class MLPClassifier:
    def __init__(self,train_set,val_set,Learning_rate=0.01,batch_size=80,epochs=1000):
        self.train_set = train_set
        self.val_set = val_set
        self.X_full = self.train_set[['x','y']].values
        self.y_full = pd.get_dummies(self.train_set['label']).values
        self.X = None
        self.y = None
        self.W1 = np.random.randn(2, 4) * np.sqrt(2. / 4) # He初始化
        self.b1 = np.zeros((1, 4))
        self.W2 = np.random.randn(4, 4) * np.sqrt(2. / 4)
        self.b2 = np.zeros((1, 4))
        self.W3 = np.random.randn(4, 2) * np.sqrt(2. / 4) 
        self.b3 = np.zeros((1, 2))
        self.loss = []
        self.H1 = None
        self.H2 = None
        self.logits = None
        self.Learning_rate = Learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
    
    def softmax(self):
        _ = np.exp(self.logits - np.max(self.logits, axis=1, keepdims=True))
        softmax_logits = _ / np.sum(_, axis=1, keepdims=True)
        return softmax_logits

    def relu(self, M):
        return np.maximum(0, M) 
    
    def deriv_relu(self, M):
        return (M > 0).astype(float) 
    
    def CE(self,M_pred,M_true):
        epsilon = 1e-10
        M_pred = np.clip(M_pred, epsilon, 1. - epsilon)
        M_CE = -(M_true*np.log(M_pred))
        return M_CE
    def compute_loss(self):
        return np.sum(np.mean(self.CE(self.softmax(),self.y),axis=1))
    
    def forward(self):
        self.H1 = self.relu(self.X @self.W1 + self.b1)
        self.H2 = self.relu(self.H1 @self.W2 + self.b2)
        self.logits = self.H2 @self.W3 + self.b3

    def backward(self):
        self.dL_dlogits = self.softmax() - self.y
        self.dL_dH2 = self.dL_dlogits @ self.W3.T * self.deriv_relu(self.H2)
        self.dL_dH1 = self.dL_dH2 @ self.W2.T * self.deriv_relu(self.H1)
        self.dL_dW3 = self.H2.T @ self.dL_dlogits
        self.dL_dW2 = self.H1.T @ self.dL_dH2
        self.dL_dW1 = self.X.T @ self.dL_dH1
        self.dL_db3 = np.sum(self.dL_dlogits, axis=0, keepdims=True)
        self.dL_db2 = np.sum(self.dL_dH2, axis=0, keepdims=True)
        self.dL_db1 = np.sum(self.dL_dH1, axis=0, keepdims=True)

        self.W3 -= self.Learning_rate * self.dL_dW3
        self.W2 -= self.Learning_rate * self.dL_dW2
        self.W1 -= self.Learning_rate * self.dL_dW1
        self.b3 -= self.Learning_rate * self.dL_db3
        self.b2 -= self.Learning_rate * self.dL_db2
        self.b1 -= self.Learning_rate * self.dL_db1

    def fit(self):
        sample_count = len(self.X_full)
        for epoch in range(self.epochs):
            indices = np.arange(sample_count)
            np.random.shuffle(indices)
            X_shuffled = self.X_full[indices]
            y_shuffled = self.y_full[indices]
            epoch_loss = []
            for start_idx in range(0, sample_count, self.batch_size):
                end_idx = min(start_idx + self.batch_size, sample_count)
                self.X = X_shuffled[start_idx:end_idx]
                self.y = y_shuffled[start_idx:end_idx]
                self.forward()
                epoch_loss.append(self.compute_loss())
                self.backward()
            self.loss.append(np.mean(epoch_loss))
            if epoch % 100 == 99:
                print(f'Epoch {epoch+1}, Loss: {self.loss[-1]}')

    def predict(self):
        self.X = self.val_set[['x','y']].values
        self.forward()
        return np.argmax(self.softmax(), axis=1)