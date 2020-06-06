#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pickle
from time import time, strftime

import numpy as np
import pandas as pd

from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder


class NCMML():
    """
    Nearest Class Mean Metric Learning (NCMML) v0.5
    Modified version of T. Mensink's Matlab code
    @ date : Created on Tue, Dec, 11, 2018
    @ author : Sangjun Han, South Korea
    """
    
    """
    References
    [1] T. Mensink et al., "Distance-Based Image Classification: Generalizing to New Classes at Near Zero Cost,"
    IEEE Transactions on Pattern Analysis and Machine Intelligence, 2013
    [2] T. Mensink et al., "Metric Learning for Large Scale Image Classification: Generalizing to New Classes at Near-Zero Cost,"
    European Conference on Computer Vision, 2012
    """
    
    
    def __init__(self):
        """
        Initialize the model.
        """
        
        self.W_output = None
        self.centroids_output = None
        self.threshold = None
        
        
    def saveModel(self, file_name=""):
        if file_name == "":
            file_name = "./NCMML_model_" + strftime("%Y_%m_%d") + ".pickle"
            
        with open(file_name, 'wb') as f:
            pickle.dump(self, f)
        
        
    def loadModel(self, file_name=""):
        if file_name == "":
            file_name = "./NCMML_model_" + strftime("%Y_%m_%d") + ".pickle"
            
        with open(file_name, 'rb') as f:
            return pickle.load(f)
            
            
    def saveWeights(self, file_name=""):
        if file_name == "":
            np.save("./NCMML_weights_" + strftime("%Y_%m_%d"), self.W_output, allow_pickle=True)
        else:
            np.save(file_name, self.W_output, allow_pickle=True)
           
        
    def setThreshold(self, threshold):
        self.threshold = threshold
        
        
    def fit(self, X, y, X_val, y_val, semantic_dim=500, learning_rate=0.001, init=None, reg=0.001,
            max_iters=100, batch_size=256, decay=0.8, num_decay=5, stop_tol=0.001, num_stop_tol=30):
        """
        Train this model using stochastic gradient descent

        Inputs : N -> numOfData, D -> features, d -> features in semantic space, C -> numOfClass 
        - X : A numpy array of shape (N, D) for train
        - y : A numpy array of shape (N,) for train
        - X_val : For validation
        - y_val : For validation
        - semantic_dim : Feature dimension in semantic space (d,)
        - learning_rate : Learning rate for optimization
        - init : If None, normal random distribution, or weight matrix W
        - reg : Parameter for regularization
        - max_iters : The number of iterating
        - batch_size : The number of randomly sampling data for one iteration
        - decay : The ratio of decaying learning rate
        - num_decay : Epoch cycle for decaying learning rate
        - stop_tol : Tolerance for early stopping 
        - num_stop_tol : The counting of violating tolerance for early stopping

        Outputs:
        - history : History dictionary containing training loss and accuracy
        """
        
        N, D = X.shape     
        stop_count = 0
        
        history = {}
        history["train_loss"] = []
        history["train_acc"] = []
        history["val_loss"] = []
        history["val_acc"] = []
        
        # centroids for each class
        centroids = self.__initCentroids(X, y)
        self.centroids_output = centroids
        
        # initialize W
        if init is None:
            W = np.random.randn(D, semantic_dim) / np.sqrt(D)
        else:
            W = init
        
        # update
        total_time = 0
        start_time = time()
        print("Start training")
        
        for i in range(0, max_iters):
            train_loss = 0
            train_acc = 0
            
            # global W
            self.W_output = W
            
            # randomly sampling batch
            x_batch, y_batch = self.__randomBatch(X, y, batch_size)
            x_batch += (reg * np.random.randn(x_batch.shape[0], x_batch.shape[1])) # regularization term

            # transform to semantic space
            Wx = x_batch.dot(W)
            Wc = centroids.dot(W)

            # softmax activation and loss
            alpha, y_batch_pred, train_loss = self.__softmaxProbLoss(Wx, Wc, y_batch)
            train_acc = accuracy_score(y_batch, y_batch_pred)
            
            # metrics for train
            train_loss = abs(train_loss)
            train_acc = abs(train_acc)
            
            history["train_loss"].append(train_loss)
            history["train_acc"].append(train_acc)
            
            # metrics for val
            if X_val is not None and y_val is not None:
                val_loss, val_acc, _ = self.predict(X_val, y_val)
                history["val_loss"].append(abs(val_loss))
                history["val_acc"].append(abs(val_acc))
            
            # time measure from backward to forward
            epoch_time = time() - start_time
            total_time += epoch_time
            print("Epoch %d (%0.2f sec) - train_loss : %0.4f, train_acc : %0.4f, val_loss : %0.4f, val_acc : %0.4f" % (i+1, epoch_time, train_loss, train_acc, val_loss, val_acc))
            
            # early stopping
            if len(history["train_loss"]) > 2 and (abs(history["train_loss"][-2] - history["train_loss"][-1]) < stop_tol):
                stop_count += 1
                if stop_count == num_stop_tol:
                    print("---------Early stopping\n")
                    break
            
            # learning rate decay
            if (i+1) % num_decay == 0:
                learning_rate *= decay
                print("---------Learning_rate was decayed to %0.6f" % (learning_rate))
            
            # start updating
            start_time = time()
            
            # compute gradient 1
            alpha[range(x_batch.shape[0]), y_batch] -= 1

            # compute gradient 2
            tmp = (Wc * np.sum(alpha, axis=0)[np.newaxis].T) - (alpha.T.dot(Wx))
            grad = centroids.T.dot(tmp)

            # compute gradient 3
            tmp = alpha.dot(Wc)
            grad -= x_batch.T.dot(tmp)
            grad *= -(2 / batch_size)

            # update
            W -= learning_rate * grad

        print("Total time for training - %0.2f sec" % (total_time))
              
        return history
        
        
    def __initCentroids(self, X, y):
        """
        K-means clusteing for each class
        
        Outputs :
        - centroids : A numpy array of mean cetroid for each class (C, D)
        """
        
        N, D = X.shape
        
        classes = np.unique(y) # it is sorted unique label
        num_classes = len(classes)
    
        print("Initializing mean-centroids for each class")
        
        # output numpy array
        centroids = np.empty([num_classes, D])
        
        start_time = time()
        for i, class_i in enumerate(classes):
            centroids[i, :] = np.mean(X[y == class_i], axis=0)

        print("%0.2f sec for initializng\n" % (time() - start_time))

        return centroids


    def __randomBatch(self, X, y, batch_size):
        """
        Sampling batch data 
        """
        
        N = X.shape[0]
        
        idx = np.arange(N)
        np.random.shuffle(idx)
        idx = idx[:batch_size]

        return np.copy(X[idx]), np.copy(y[idx])

    
    def __squaredDist(self, data, centroids):
        """
        Fast computing euclidean distance XX - 2XY + YY = (X - Y)(X - Y)
        Referenced from sklearn.metrics.pairwise import euclidean_distances
        
        Inputs :
        - data : data in semantic space (N, d)
        - centroids : centroids in semantic space (C, d)
        
        Outputs :
        - dist : euclidian distance between two numpy array (N, C)
        """
        
        dist = -2 * data.dot(centroids.T)
        dist = dist + np.sum(data ** 2, axis=1)[np.newaxis].T
        dist = dist + np.sum(centroids ** 2, axis=1)[np.newaxis]

        return dist
    
    
    def __softmax(self, data):
        """
        Softmax activation
        
        Inputs :
        - data : negative euclidean distance matrix (N, C)
        
        Outputs :
        - data : probability for distance
        """
        
        data = np.exp(data - np.amax(data, axis=1, keepdims=True))
        data /= np.sum(data, axis=1, keepdims=True)

        return data
    
    
    def __softmaxProbLoss(self, data, centroids, label=None, eps=0.000001):
        """
        Loss function
        
        Inputs :
        - data : data in semantic space (N, d)
        - centroids : centroids in semantic space (C, d)
        - label : y (N,)
        
        Outputs :
        - softmax_prob : probability matrix
        - label_pred : predicted label y
        - loss (if label is not contained, it returns 0)
        """
        
        N = data.shape[0]
        
        softmax_prob = self.__softmax(-self.__squaredDist(data, centroids))
        label_pred = np.argmax(softmax_prob, axis=1)
        
        if label is None:
            return softmax_prob, label_pred, 0
        
        softmax_prob_class = softmax_prob[range(N), label.astype("int16")]
        loss = -np.mean(np.log(softmax_prob_class + eps))

        return softmax_prob, label_pred, loss
    
    
    def predict(self, X, y=None):
        """
        Prediction
        
        Inputs :
        - X : A numpy array of shape (N, D)
        - y : A numpy array of shape (N,)

        Outputs :
        - loss (if label is not contained, it returns 0)
        - accuracy (if label is not contained, it returns softmax matrix)
        - y_pred
        """
        
        Wx = X.dot(self.W_output)
        Wc = self.centroids_output.dot(self.W_output)

        softmax, y_pred, loss = self.__softmaxProbLoss(Wx, Wc, y)
        
        if (y is None) or (y.shape[0] == 1): # if it predicts one data
            return 0, softmax, y_pred
        
        accuracy = accuracy_score(y, y_pred)
        
        return loss, accuracy, y_pred
    
    
    def getDist(self, X):
        """
        Get distance
        
        Inputs :
        - X : A numpy array of shape (N, D)

        Outputs:
        - dist
        """
        
        Wx = X.dot(self.W_output)
        Wc = self.centroids_output.dot(self.W_output)

        # distance
        dist = self.__squaredDist(Wx, Wc)
        
        # normalized distance
        dist /= np.sum(dist, axis=1)[:, np.newaxis]
        
        return dist
    
    
    def predictDist(self, X, y=None, new_class=999):
        """
        Prediction by distance
        Before implement this function, required to decide a threshold that separates unseen class data
        
        Inputs :
        - X : A numpy array of shape (N, D)
        - y : A numpy array of shape (N,)
        - new_class : assign new class to the data which is far from each centroids 

        Outputs:
        - accuracy (if label is not contanied, it returns 0)
        - dist (if label is not contanied, it returns distance matrix)
        - y_pred
        """
        
        dist = self.getDist(X)

        # predict
        y_pred = np.argmin(dist, axis=1)
        y_pred[np.amin(dist, axis=1) > self.threshold] = new_class # unknown
        
        if (y is None) or (y.shape[0] == 1): # if it predicts one data
            return 0, dist, y_pred
        
        accuracy = accuracy_score(y, y_pred)
        
        return accuracy, dist, y_pred