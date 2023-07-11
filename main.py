from FC import fullyConnectedNN as fc
import numpy as np
from sklearn.model_selection import train_test_split
from experiments import experiment


def normalize_features(features):
    means = np.mean(features, axis=0)  # Calculate the mean along axis 0 (columns)
    normalized_features = features - means
    return normalized_features


X = np.load('MNIST-data.npy')
y = np.load("MNIST-lables.npy")

# make the features ready for the net
labels = np.zeros((len(y), 10))
labels[np.arange(len(y)), y] = 1
features = X.reshape((X.shape[0], -1))
input_dim = len(features[0])
output_dim = len(labels[0])

# split to train and test
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

# normalize the feature to to the avg of the train
mean = np.mean(np.mean(X_train, axis=0))
X_train = X_train - mean
X_test = X_test - mean

# split the test to validation and test
X_vladition, X_test, y_vladition, y_test = train_test_split(X_test, y_test, test_size=0.5, random_state=42)


# experiment 1
# finding the best activation and their best lr
activations = ['sig', 'tan_h', 'relu']
best_score = 0
for func in activations:
    # looking for the best lr
    best_lr = None
    lrs = [0.5, 0.1, 0.05, 0.01, 5 * 10 ** 3, 10 ** -3]
    for lr in lrs:
        net_struct = [input_dim, 512, output_dim]
        exper = experiment(net_struct, activation=func, lr=lr, max_epochs=20)
        score = exper.run_experiment(X_train, y_train, X_vladition, y_vladition)
        if score > best_score:
            print("----------------------new-best-model--------------------------")
            best_lr = lr
            best_score = score
            best_func = func

# experiment 2
# finding best lost function
losses = ['hinge', 'CE']
best_loss = 'hinge'
for loss in losses:
    # looking for the best lr
    net_struct = [input_dim, 512, output_dim]
    exper = experiment(net_struct, activation=best_func, lr=best_lr, loss=loss, max_epochs=20)
    score = exper.run_experiment(X_train, y_train, X_vladition, y_vladition)
    if score > best_score:
        print("----------------------new-best-model--------------------------")
        best_loss = loss
        best_score = score

# experiment 3
# finding the best struct for the net
hiden_layer_sizes = [1024, 512, 256]
amount_of_layers_l = [1, 2, 3]
best_struct = [input_dim, 512, output_dim]

for hiden_layer_size in hiden_layer_sizes:
    for amount_of_layers in amount_of_layers_l:
        net_struct = [input_dim] + [hiden_layer] * amount_of_layers + [output_dim]
        exper = experiment(net_struct, activation=best_func, lr=best_lr, loss=best_loss)
        score = exper.run_experiment(X_train, y_train, X_vladition, y_vladition)
        if score > best_score:
            print("----------------------new-best-model--------------------------")
            best_struct = net_struct
            best_score = score

momentums = [0.9, 0.8, 0.7]
best_momentum = 1

for momentum in momentums:
    net_struct = [input_dim] + [hiden_layer] * amount_of_layers + [output_dim]
    exper = experiment(best_struct, activation=best_func, lr=best_lr, loss=best_loss, momentum=momentum)
    score = exper.run_experiment(X_train, y_train, X_vladition, y_vladition)
    if score > best_score:
        print("----------------------new-best-model--------------------------")
        best_momentum = momentum
        best_score = score


net = fc(best_struct, activation=best_func, lr=best_lr, loss=best_loss, )
#reg1 = [10**-6, 10**-5, 10**-4, 10**-3, 5*10**-2, 10**-1]
#reg2 = [10**-6, 10**-5, 10**-4, 10**-3, 5*10**-2, 10**-1]
# best_score = 0
