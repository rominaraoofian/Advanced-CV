import numpy as np
import matplotlib.pyplot as plt
import os

Eps = 1e-8

class Ex01_2:
    def __init__(self, gamma=1e-2, lr=1e-3, epochs=100, earlty_stopping_patience=200):
        self.gamma = gamma
        self.lr = lr
        self.epochs = epochs
        self.early_stopping_patience = earlty_stopping_patience
        
        self.alpha = None
        self.best_alpha = None
        self.best_epoch = None
        self.best_val_loss = float('inf')
        self.train_losses = []
        self.val_losses = []
        
    def heuristic_gamma(self, X):
        sq_norms = np.sum(X**2, axis=1).reshape(-1, 1)
        dist_sq = sq_norms + sq_norms.T - 2 * np.dot(X, X.T)
        
        triu_indices = np.triu_indices(dist_sq.shape[0], k=1)
        relevant_distances = dist_sq[triu_indices]
        
        median_sq_dist = np.median(relevant_distances)
        
        gamma = 1.0 / (2 * median_sq_dist)
        return gamma
    
    # Radial Basis Function kernel giving similairty between two sets of samples so the outputs is
    # a matrix of shape (N1, N2) where N1 and N2 are the number of samples in X1 and X2 respectively
    def rbf_kernel(self, X1, X2):
        X1_sq = np.sum(X1**2, axis=1, keepdims=True)   
        X2_sq = np.sum(X2**2, axis=1)                  
        dist = X1_sq + X2_sq - 2 * X1 @ X2.T  
        dist   = np.maximum(dist, 0.0) # avoid negatice dist         
        return np.exp(-self.gamma * dist)

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    #BCELoss
    # bernouli distributation for a training sample is p^y * (1-p)^(1-y) 
    # for update we see it on all of our training samples so we get mulitiplicateion of all of them a
    # for calculating argmax we first take log and make it summation as follow
    def Negative_log_likelihood(self, y, p):
        p = np.clip(p, Eps, 1 - Eps)
        return -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))

    
    def compute_gradient(self, K, y, p):
        N = y.shape[0]
        return (K.T @ (p - y)) / N

    def  train(self, X_train, y_train, X_val, y_val):

        N = X_train.shape[0]
        self.alpha = np.zeros(N)

        # Precomputing  RBFs
        if self.gamma == None:
            gamma = self.heuristic_gamma(X_train)
            self.gamma = gamma
        K_train = self.rbf_kernel(X_train, X_train)
        K_val = self.rbf_kernel(X_val, X_train)

        # self.best_val_loss = float('inf')
        counter = 0
        for epoch in range(self.epochs):

            pred_train = self.sigmoid(K_train @ self.alpha)
            pred_val = self.sigmoid(K_val @ self.alpha)

            train_loss = self.Negative_log_likelihood(y_train, pred_train)
            val_loss = self.Negative_log_likelihood(y_val, pred_val)

            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)

            # Save best model
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_alpha = self.alpha.copy()
                self.best_epoch = epoch
                counter = 0  
            else:
                counter += 1  

            if counter >= self.early_stopping_patience:
                print(f"Early stopping at epoch {epoch}")
                break

            grad = self.compute_gradient(K_train, y_train, pred_train)

            #  optimization step, simple gradient descent
            self.alpha -= self.lr * grad

            print(f"Epoch {epoch}: Train={train_loss:.4f}, Val={val_loss:.4f}")

        self.alpha = self.best_alpha


    def compute_accuracy(self, TP, FP, TN, FN):
        return (TP + TN) / (TP + FP + TN + FN + Eps) 
    
    def evaluate(self, X_test, y_test, X_train):
        K_test = self.rbf_kernel(X_test, X_train)
        pred_prob = self.sigmoid(K_test @ self.alpha)
        y_pred = np.where(pred_prob > 0.5, 1, 0)

        TP = np.sum((y_test == 1) & (y_pred == 1))
        TN = np.sum((y_test == 0) & (y_pred == 0))
        FP = np.sum((y_test == 0) & (y_pred == 1))
        FN = np.sum((y_test == 1) & (y_pred == 0))
        
        accuracy = self.compute_accuracy(TP, FP, TN, FN)

        print("\n--- Evaluation ---")
        print(f"Accuracy: {accuracy:.4f}")

        return accuracy

    def plot_losses(self):
        plt.plot(self.train_losses, label='Train Loss')
        plt.plot(self.val_losses, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.title('Training vs Validation Loss')

        # create folder if it doesn't exist
        save_dir = "plots_02"
        os.makedirs(save_dir, exist_ok=True)

        filename = f"plot_{self.gamma}_{self.lr}_{self.epochs}_{self.best_val_loss}_{self.best_epoch}.png"
        filepath = os.path.join(save_dir, filename)

        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close()

        print(f"Saved plot -> {filepath}")