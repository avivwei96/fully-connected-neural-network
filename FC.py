import numpy as np
import random
from tqdm import tqdm

class fullyConnectedNN(object):
    def __init__(self, layers_sizes, l1_reg=0, l2_reg=0, momentum=0, activation='sig', loss='CE', lr_change=1):
        """
        this function init the NN
        :param layers_sizes: (np array like) the struct of the net
        :param l1_reg: (float) the coefficient of the l1 reg
        :param l2_reg: (float) the coefficient of the l2 reg
        :param momentum: (float) the coefficient of the grad momentum
        :param activation: (string) the name of the activation function
        :param loss: (string) the name of the loss function
        :param lr_change: (float) the coefficent of the changing lr if the net doesnt progress
        """
        self.layers_sizes = layers_sizes
        self.n_layers = len(layers_sizes)
        self._weights = [np.random.normal(loc=0, scale=np.sqrt(2 / layer_size_in + layer_size_out), size=(layer_size_in, layer_size_out))  # the + 1 is for the bias
                         for layer_size_in, layer_size_out in zip(layers_sizes[:-1], layers_sizes[1:])]
        self.layer_in = None
        self.deltas = None
        self.gradients = [np.zeros_like(w) for w in self._weights]
        self.l1 = l1_reg
        self.l2 = l2_reg
        self.loss = 0
        self.lr = 0.01
        self.lr_change = lr_change
        self.momentum = momentum

        # chose the activation function and the loss function
        self.loss_func_name = loss
        self.activation_func = activation
        if activation == 'sig':
            self.activation = self.sigmoid
            self.grad = self.sigmoid_grad
        if activation == 'tan_h':
            self.activation = self.tan_h
            self.grad = self.tan_h_grad
        if activation == 'relu':
            self.activation = self.relu
            self.grad = self.relu_gradient
        if loss == "CE":
            self.loss_func = self.cross_entropy
            self.loss_grad = self.cross_entropy_derivative
        if loss == 'hinge':
            self.loss_func = self.hinge_loss
            self.loss_grad = self.hinge_loss_gradient


    def feed_forward(self, features, train=False):
        """
        this function returns the output of the net for given features
        :param features:(np array) the features that we want to predict on
        :param train:(bool) if the net is on train mode
        :return: the out put of the net
        """
        # add one to the features for the bias
        res = np.array(features)
        if train:
            self.layer_in = []
        for weight_mat in self._weights:
            if train:
                self.layer_in.append(res)
            res = self.activation(res @ weight_mat)
        if train:
            self.layer_in.append(res)
        return res

    def back_prop(self, features, labels):
        """
        this function cumpute the deltas of the net
        :param features: (np array) The input features for training.
        :param labels: (np array) The corresponding labels for the input features.
        """
        #init deltas and computing der
        net_res = self.softmax(self.feed_forward(features, train=True))
        self.loss += np.mean(self.loss_func(net_res, labels))
        error_der = self.loss_grad(net_res, labels)

        # compute the first delta
        delta = self.grad(self.layer_in[-1]) * error_der
        self.gradients[-1] = self.layer_in[-2].T @ delta + self.gradients[-1] * self.momentum

        # implementing the back prop rule
        for layer_i in range(2, self.n_layers):
            sum_of_Del_W = self._weights[-layer_i + 1] @ delta.T
            grad = self.grad(self.layer_in[-layer_i])
            delta = grad * sum_of_Del_W.T
            self.gradients[-layer_i] = self.layer_in[-layer_i - 1].T @ delta + self.gradients[-layer_i] * self.momentum

    def update_weights(self):
        # this function update the weight due to the gradient (SGD)
        for layer_i in range(0, self.n_layers - 1):
            self._weights[layer_i] -= self.lr * self.gradients[layer_i] + self.l2*self._weights[layer_i] + \
                                      self.l1*(self._weights[layer_i]/abs(self._weights[layer_i]))

    def predict_proba(self, features):
        # this function return a proba vector with th net res
        return self.softmax(self.feed_forward(features))

    def predict(self, features):
        # this function return the label prediction of the net
        proba_vec = self.predict_proba(features)
        return np.argmax(proba_vec, axis=1)

    def train(self, features, labels, X_vladition, y_vladition, epochs=100, lr=0.01, batch_size=128):
        """

        :param features: (np array) the vectors of the features of the train data
        :param labels: (np array) one hot vector of the true label of the train data
        :param X_vladition: (np array) the vectors of the features of the validation data
        :param y_vladition:(np array) one hot vector of the true label of the validation data
        :param epochs: (int) the number of max epochs
        :param lr: (float) the learning rate for changing the weights
        :param batch_size: (int) the size of the batch
        """
        self.lr = lr
        counter = 0
        score = 0
        for epoch in tqdm(range(epochs)):
            self.loss = 0
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
            n_score = self.score(X_vladition, np.argmax(y_vladition, axis=1))
            if n_score >= score + 0.01:
                counter = 0
                score = n_score
            else:
                self.lr *= self.lr_change  # if the score doesnt increase the lr will change due to the lr_change
                counter += 1
            if counter > 10:
                break

    def score(self, features, labels):
        # return the score of the net for those labels and features
        res = self.predict(features)
        correct_predictions = np.sum(labels == res)
        return correct_predictions / len(labels)

    def print_net(self):
        # print the net details
        print(f"| lr = {self.lr} | activtion function = {self.activation_func} | loss = {self.loss_func_name} | \n"
              f"| reg l1 = {self.l1} | reg l2 = {self.l2} | momentum = {self.momentum} | lr chamge = {self.lr_change} |")

    # loss function and their grad
    @staticmethod
    def cross_entropy(pred_prob, true_label):
        return -sum(np.log(pred_prob).T @ true_label)/len(true_label)

    @staticmethod
    def cross_entropy_derivative(y_pred, y_true):
        y_pred = np.array(y_pred)
        y_true = np.array(y_true)
        d_loss = y_pred - y_true
        return d_loss

    @staticmethod
    def hinge_loss(pred_labels, true_labels):
        y_true_modified = np.where(true_labels == 0, -1, true_labels)
        loss = np.maximum(0, 1 - y_true_modified * pred_labels)
        loss = np.sum(loss, axis=1) - 1  # Subtract 1 to exclude the correct class
        loss = np.maximum(0, loss)
        return loss

    @staticmethod
    def hinge_loss_gradient(y_pred, y_true):
        y_pred = np.array(y_pred)
        y_true = np.array(y_true)
        grad = np.where(y_true * y_pred < 1, -y_true, 0)
        return grad

    @staticmethod
    def softmax(values):
        exponent_val = np.exp(values)
        sum_of_vals = np.sum(exponent_val, axis=1)
        sum_of_vals = sum_of_vals.reshape(-1, 1)
        return exponent_val / sum_of_vals

    # activtion functions and their grad
    @staticmethod
    def sigmoid(features):
        return 1 / (1 + np.exp(-features))

    @staticmethod
    def sigmoid_grad(sigmoid):
        return sigmoid * (1 - sigmoid)
    @staticmethod
    def relu(features):
        return np.maximum(0, features)

    @staticmethod
    def relu_gradient(relu_res):
        grad = np.where(relu_res > 0, 1, 0)
        return grad

    @staticmethod
    def tan_h(features):
        return np.tanh(features)

    @staticmethod
    def tan_h_grad(tan_h_res):
        return 1 - tan_h_res ** 2


