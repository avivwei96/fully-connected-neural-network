from FC import fullyConnectedNN as fc
import numpy as np
from sklearn.model_selection import train_test_split

X = np.load('MNIST-data.npy')
y = np.load("MNIST-lables.npy")

labels = np.zeros((len(y), 10))
labels[np.arange(len(y)), y] = 1
features = X.reshape((X.shape[0], -1))

X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

X_vladition, X_test, y_vladition, y_test = train_test_split(X_test, y_test, test_size=0.5, random_state=42)

lrs = [0.000001, 0.00001, 0.001, 0.1, 1]
hiden_layer_sizes = [200, 100, 50, 10]
amount_of_layers = [1, 2, 3]
input_dim = len(X_train[0])
output_dim = len(y_test[0])
best_score = 0
for lr in lrs:
    for hiden_lay in hiden_layer_sizes:
        for amount_of_lay in amount_of_layers:
            net_struct = [input_dim] + [hiden_lay]*amount_of_lay + [output_dim]
            net = fc(net_struct)
            net.train(X_train, y_train, epochs=1000//(hiden_lay * amount_of_lay), lr=lr)
            net_score = net.score(X_vladition, np.argmax(y_vladition, axis=1))
            if net_score > best_score:
                print(f"\nNew best!!! {net_score} lr={lr}, hiden_layer={hiden_lay}, amount_of_lay={amount_of_lay}")
                best_net = net
                best_score = net_score
best_net.score()