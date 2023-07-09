import numpy as np

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

    def feed_forward(self, features):
        """
        this function returns the output of the net for given features
        :param features(np array): the features that we want to predict on
        :return: the out put of the net
        """
        # add one to the features for the bias
        res = np.append(features, 1)
        for weight_mat in self._weights:
            res = np.tanh(res @ weight_mat)
        return res

