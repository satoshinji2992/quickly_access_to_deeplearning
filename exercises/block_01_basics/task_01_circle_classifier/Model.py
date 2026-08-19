import numpy as np


class MLPClassifier:
    def __init__(
        self,
        train_set,
        val_set,
        Learning_rate=0.01,
        batch_size=80,
        epochs=1000,
        seed=0,
    ):
        self.train_set = train_set
        self.val_set = val_set
        self.X_full, self.y_full, _ = self._prepare_data(self.train_set)
        self.X = None
        self.y = None
        self.rng = np.random.default_rng(seed)
        self.W1 = self.rng.standard_normal((2, 4)) * np.sqrt(2.0 / 2)  # He 初始化
        self.b1 = np.zeros((1, 4))
        self.W2 = self.rng.standard_normal((4, 4)) * np.sqrt(2.0 / 4)
        self.b2 = np.zeros((1, 4))
        self.W3 = self.rng.standard_normal((4, 2)) * np.sqrt(2.0 / 4)
        self.b3 = np.zeros((1, 2))
        self.loss = []
        self.train_accuracy = []
        self.val_loss = []
        self.val_accuracy = []
        self.H1 = None
        self.H2 = None
        self.logits = None
        self.Learning_rate = Learning_rate
        self.batch_size = batch_size
        self.epochs = epochs

    @staticmethod
    def _prepare_data(dataframe):
        X = dataframe[["x", "y"]].to_numpy(dtype=float)
        labels = dataframe["label"].to_numpy(dtype=np.int64)
        if not np.isin(labels, (0, 1)).all():
            raise ValueError("circle-classifier labels must be 0 or 1")
        one_hot = np.eye(2, dtype=float)[labels]
        return X, one_hot, labels

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
        M_CE = -np.sum(M_true*np.log(M_pred), axis=1)
        return M_CE
    def compute_loss(self):
        return np.mean(self.CE(self.softmax(),self.y))

    def forward(self):
        self.H1 = self.relu(np.dot(self.X, self.W1) + self.b1)
        self.H2 = self.relu(np.dot(self.H1, self.W2) + self.b2)
        self.logits = np.dot(self.H2, self.W3) + self.b3
        return self.logits

    def backward(self):
        batch_size = self.X.shape[0]
        self.dL_dlogits = (self.softmax() - self.y) / batch_size
        self.dL_dH2 = np.dot(self.dL_dlogits, self.W3.T) * self.deriv_relu(self.H2)
        self.dL_dH1 = np.dot(self.dL_dH2, self.W2.T) * self.deriv_relu(self.H1)
        self.dL_dW3 = np.dot(self.H2.T, self.dL_dlogits)
        self.dL_dW2 = np.dot(self.H1.T, self.dL_dH2)
        self.dL_dW1 = np.dot(self.X.T, self.dL_dH1)
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
            indices = self.rng.permutation(sample_count)
            X_shuffled = self.X_full[indices]
            y_shuffled = self.y_full[indices]
            for start_idx in range(0, sample_count, self.batch_size):
                end_idx = min(start_idx + self.batch_size, sample_count)
                self.X = X_shuffled[start_idx:end_idx]
                self.y = y_shuffled[start_idx:end_idx]
                self.forward()
                self.backward()

            train_loss, train_accuracy = self.evaluate(self.train_set)
            val_loss, val_accuracy = self.evaluate(self.val_set)
            self.loss.append(train_loss)
            self.train_accuracy.append(train_accuracy)
            self.val_loss.append(val_loss)
            self.val_accuracy.append(val_accuracy)
            if epoch == 0 or (epoch + 1) % 100 == 0 or epoch + 1 == self.epochs:
                print(
                    f"Epoch {epoch + 1:4d} | "
                    f"train_loss={train_loss:.4f} train_acc={train_accuracy:.3f} | "
                    f"val_loss={val_loss:.4f} val_acc={val_accuracy:.3f}"
                )
        return self

    def evaluate(self, dataframe):
        self.X, self.y, labels = self._prepare_data(dataframe)
        self.forward()
        predictions = np.argmax(self.softmax(), axis=1)
        return self.compute_loss(), float(np.mean(predictions == labels))

    def predict(self, dataframe=None):
        if dataframe is None:
            dataframe = self.val_set
        self.X, _, _ = self._prepare_data(dataframe)
        self.forward()
        return np.argmax(self.softmax(), axis=1)
