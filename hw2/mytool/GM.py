import numpy as np

class GM(object):
    def __init__(self, sizes):
        # set info
        self.sizes = sizes

        # set activate function and loss functino
        self.activate_func = self.act_sigmoid
        self.loss_func = self.loss_MSE

        # init weights [w1, ... , wn, wb]
        self.weight = np.zeros((sizes[0],sizes[1]))
        

    def save(self, filepath):
        np.save(filepath, self.weights)

    def load(self, filepath):
        self.weights = np.load(filepath)
        self.weights = [np.vstack(nparray) for nparray in self.weights]

    def train(self, trainset):
        # format data row by row
        X = np.vstack([x for x,y in trainset])
        X0 = np.vstack([x for x,y in trainset if y == [0.0]])
        X1 = np.vstack([x for x,y in trainset if y == [1.0]])

        # compute mean and covariance
        mean0 = np.mean(X0, axis=0).reshape((1,-1))
        mean1 = np.mean(X1, axis=0).reshape((1,-1))
        cov_inv = np.linalg.pinv(np.cov(X.T).reshape((len(X[0]),len(X[0]))))

        # compute weight (w, b)
        w = (mean1 - mean0).dot(cov_inv).T
        b = (
                - 0.5 * mean1.dot(cov_inv).dot(mean1.T)
                + 0.5 * mean0.dot(cov_inv).dot(mean0.T)
                + np.log(float(len(X1))/float(len(X0))) 
            )

        # set model weight
        self.weight = np.vstack([w,b])

    def feedforward(self, x):
        # init input
        x = np.array([x])
        
        # forward
        x = np.append(x, [[1]], axis=1)
        o = x.dot(self.weight)
        o = self.activate_func(o)
    
        return o[0]

    def evaluate(self, evaluate_set):
        correct = 0.0
        for x, y in evaluate_set:
            o = self.feedforward(x)
            equal = True
            for i in xrange(len(o)):
                if int(y[0] + 0.5) != int(o[0] + 0.5):
                    equal = False
                    break
            if equal:
                correct += 1
        correct /= len(evaluate_set)
        print "Correct: %e" % (correct)
        return correct
                

    def test(self, x, printout = True):
        o = self.feedforward(x)
        if printout:    
            print "Test:%s\t=>\t%s" % (x, o)
        return o

    def act_sigmoid(self, z):
        return 1.0/(1.0+np.exp(-z))

    def loss_MSE(self, o, y):
        loss = 0.0
        for act, des in zip(o, y):
            loss += (act - des) * (act - des)
        return loss




