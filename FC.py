import numpy as np
import random
from tqdm import tqdm

class fullyConnectedNN(object):
    def __init__(self, layers_sizes):
        """
        this function init the NN
        :param layers_sizes(list): list of the sizes of the layers
        """
        self.layers_sizes = layers_sizes
        self.n_layers = len(layers_sizes)
        self._weights = [np.random.randn(layer_size_in + 1, layer_size_out)  # the + 1 is for the bias
                         for layer_size_in, layer_size_out in zip(layers_sizes[:-1], layers_sizes[1:])]
        self.layer_out = None
        self.deltas = None
        self.lr = 0.01

    def feed_forward(self, features, train=False):
        """
        this function returns the output of the net for given features
        :param features:(np array) the features that we want to predict on
        :param train:(bool) if the net is on train mode
        :return: the out put of the net
        """
        # add one to the features for the bias
        res = features
        if train:
            self.layer_out = []
        for weight_mat in self._weights:
            ones_column = np.ones((len(res), 1))
            res = np.concatenate((res, ones_column), axis=1)
            if train:
                self.layer_out.append(res)
            res = np.tanh(res @ weight_mat)
        if train:
            self.layer_out.append(res)
        return res

    def back_prop(self, features, labels):
        """
        this function cumpute the deltas of the net
        :param features: (np array) The input features for training.
        :param labels: (np array) The corresponding labels for the input features.
        """
        #init deltas and computing der
        self.deltas = [np.zeros((len(features), layer_size), dtype=float) for layer_size in self.layers_sizes[1:]]
        net_res = self.softmax(self.feed_forward(features, train=True))
        error_der = self.cross_entropy_derivative(net_res, labels)

        # compute the first delta
        self.deltas[-1] = (1 - np.tanh(self.layer_out[-1])**2) * error_der

        # implementing the back prop rule
        for layer_i in range(2, self.n_layers):
            sum_of_Del_W = self._weights[-layer_i + 1][:-1] @ self.deltas[-layer_i + 1].T
            res = 1 - np.tanh(self.layer_out[-layer_i])**2
            res = np.delete(res, -1, axis=1)
            self.deltas[-layer_i] = res * sum_of_Del_W.T


    def update_weights(self):
        for layer_i in range(0, self.n_layers - 1):
            der = self.layer_out[layer_i][:, np.newaxis, :] * self.deltas[layer_i][:, :, np.newaxis]
            der = np.sum(der, axis=0)
            self._weights[layer_i] -= self.lr*der.T

    def predict_proba(self, features):
        return self.softmax(self.feed_forward(features))

    def predict(self, features):
        proba_vec = self.predict_proba(features)
        return np.argmax(proba_vec, axis=1)

    def train(self, features, labels, epochs=100, lr=0.01, batch_size=32):
        self.lr = lr
        for epoch in tqdm(range(epochs)):
            # Shuffle the training data
            combined_data = list(zip(features, labels))
            random.shuffle(combined_data)
            features_shuffled, labels_shuffled = zip(*combined_data)

            # Create mini-batches
            mini_batches = [
                (features_shuffled[k:k + batch_size], labels_shuffled[k:k + batch_size])
                for k in range(0, len(features), batch_size)
            ]

            # Iterate over mini-batches and update model parameters
            for mini_batch_features, mini_batch_labels in mini_batches:
                self.back_prop(mini_batch_features, mini_batch_labels)
                self.update_weights()

    def score(self, features, labels):
        res = self.predict(features)
        correct_predictions = np.sum(labels == res)
        return correct_predictions / len(labels)

    @staticmethod
    def cross_entropy(pred_prob, true_label):
        return -sum(np.log(pred_prob) @ true_label)

    @staticmethod
    def softmax(values):
        exponent_val = np.exp(values)
        sum_of_vals = np.sum(exponent_val, axis=1)
        sum_of_vals = sum_of_vals.reshape(-1, 1)
        return exponent_val / sum_of_vals

    @staticmethod
    def cross_entropy_derivative(y_pred, y_true):
        y_pred = np.array(y_pred)
        y_true = np.array(y_true)
        # Avoid division by zero
        epsilon = 1e-7
        # Calculate the derivative of cross-entropy loss
        d_loss = - (y_true / (y_pred + epsilon)) + (1 - y_true) / (1 - y_pred + epsilon)

        return d_loss

