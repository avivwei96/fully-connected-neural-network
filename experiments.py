from FC import fullyConnectedNN as fc

class experiment(object):
    def __init__(self, net_struct, l1_reg=0, l2_reg=0, momentum=0, activation='sig', loss='CE', lr_change=1, max_epochs=100, lr=0.01):
        self.net = fc(net_struct, l1_reg, l2_reg, momentum, activation, loss, lr_change)
        self.max_epoches = max_epochs
        self.lr = lr
        self.best_net = None
        self.best_score = 0

    def run_experiment(self, X_train, y_train, X_vald, y_vald):
        print("-------------------starting-to-new-experiment------------------")
        self.net.train(X_train, y_train, X_vald, y_vald, epochs=self.max_epoches, lr=self.lr)
        self.net.print_net()
        score = self.net.score(X_vlad, np.argmax(y_vlad, axis=1))
        print(f"validation score = {score}")
        self.net.print_net()
        print('----------------------experiment-over--------------------------')
        return score





