from FC import fullyConnectedNN as fc
import numpy as np

net = fc([784, 200, 150, 10])
random_list = np.random.randn(2,784)
labels = [1, 0, 0, 1]
net.back_prop(random_list,labels = [0, 0, 0, 0, 0, 0, 1, 0, 0, 1])
