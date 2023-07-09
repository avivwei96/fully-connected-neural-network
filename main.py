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
# Convert the tuples to arrays
X_train = np.array(X_train)
X_test = np.array(X_test)
y_train = np.array(y_train)
y_test = np.array(y_test)

lr = [0.01]
net = fc([784, 100, 10])
net.train(X_train, y_train, epochs=10)
print(net.score(X_test, np.argmax(y_test, axis=1)))
