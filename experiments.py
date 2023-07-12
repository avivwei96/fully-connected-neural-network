from FC import fullyConnectedNN as fc
import numpy as np

class experiment(object):
    def __init__(self, net_struct, l1_reg=0, l2_reg=0, momentum=0, activation='sig',
                 loss='CE', lr_change=1, max_epochs=100, lr=0.01, batch_size=128):
        """
        :param net_struct:  (np array like) the struct of the net
        :param l1_reg: (float) the coefficient of the l1 reg
        :param l2_reg: (float) the coefficient of the l2 reg
        :param momentum: (float) the coefficient of the grad momentum
        :param activation: (string) the name of the activation function
        :param loss: (string) the name of the loss function
        :param lr_change: (float) the coefficent of the changing lr if the net doesnt progress
        :param max_epochs: (int) the max epochs of experment
        :param lr: (float) the learing rate of the expirement
        """
        self.net = fc(net_struct, l1_reg, l2_reg, momentum, activation, loss, lr_change)
        self.max_epoches = max_epochs
        self.lr = lr
        self.best_net = None
        self.best_score = 0
        self.batch_size = batch_size
        
    def run_experiment(self, X_train, y_train, X_vald, y_vald):
        # this fucnction create a net and run the experiment
        # returns the score and the net
        print()
        print("-------------------starting-to-new-experiment------------------")
        print()
        self.net.train(X_train, y_train, X_vald, y_vald, epochs=self.max_epoches, lr=self.lr, batch_size=self.batch_size)
        val_score = self.net.score(X_vald, np.argmax(y_vald, axis=1))
        train_score = self.net.score(X_train, np.argmax(y_train, axis=1))
        print()
        print('--------------------------results------------------------------')
        print(f"validation score={val_score} train score={train_score}")
        self.net.print_net()
        print('----------------------experiment-over--------------------------')
        print()
        return self.net, val_score





