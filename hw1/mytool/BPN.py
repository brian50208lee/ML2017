import random,math
import numpy as np

class BPN(object):

    def __init__(self, sizes, learn_alpha = 0.5, learn_reg = 0.0, print_iter = 1000):
        # set info
        self.sizes = sizes
        self.num_layers = len(sizes)
        self.learn_alpha = learn_alpha
        self.learn_reg = learn_reg

        # set activate function and loss functino
        self.activate_func = self.act_identity
        self.deactivate_func = self.act_deidentity
        self.loss_func = self.loss_MSE

        # init weights [w1, ... , wn, wb]
        self.weights = [np.random.randn(x+1, y) for x, y in zip(sizes[:-1], sizes[1:])]
     
        # tmp training data [a1, ... , an, base = 1]
        self.activate = [np.zeros((1, num_neural+1)) for num_neural in sizes]
        
        # tmp error data [e1, ... , en]
        self.error = [np.zeros((1, num_neural)) for num_neural in sizes]

        # loss record
        self.print_iter = print_iter
        self.loss = 0.0
        self.iter = 0

    def save(self, filepath):
        np.save(filepath, self.weights)

    def load(self, filepath):
        self.weights = np.load(filepath)
        self.weights = [np.vstack(nparray) for nparray in self.weights]


    def train(self, trainset):
        for data, desired in trainset:
            out = self.feedforward(data)
            self.calError(desired)
            self.reweight()
            
            # updata error
            self.iter += 1
            self.loss += self.loss_func(out, desired)
            if (self.iter % self.print_iter) == 0:
                self.loss /= self.print_iter
                print "iter: %-10dAvgError: %e" % (self.iter ,self.loss)
                #print "%e" % (self.loss)
                self.loss = 0.0

    def evaluate(self, trainset):
        m_loss = 0.0
        for data, desired in trainset:
            out = self.feedforward(data)
            self.calError(desired)
            m_loss += self.loss_func(out, desired)
        m_loss /= len(trainset)
        print "AvgError: %e" % (m_loss)
        return m_loss

    def traingroup(self, trainset):
        datasize = len(trainset)
        act = [np.zeros((1, num_neural+1)) for num_neural in self.sizes]
        err = [np.zeros((1, num_neural)) for num_neural in self.sizes]
        tar = [0.0]
        m_loss = 0.0
        for data, desired in trainset:
            tar = [x + y for x, y in zip(tar, desired)]
            out = self.feedforward(data)
            self.calError(desired)
            act = [x + y for x, y in zip(act, self.activate)]
            err = [x + y for x, y in zip(err, self.error)]
            m_loss += self.loss_func(out, desired)

        self.activate = [a/datasize for a in act]
        self.error = [e/datasize for e in err]
        tar = [t/datasize for t in tar]
        self.reweight()
        
        self.iter += 1
        out = self.activate[self.num_layers-1]
        out = np.delete(out, out.size - 1, 1)[0]
        m_loss /= len(trainset) 
        print "iter: %-10d\tAvgError: %e" % (self.iter ,m_loss)
                

    def test(self, data, printout = True):
        out = self.feedforward(data)
        if printout:    print "data:%s\n=>\t%.20f" % (data,out)
        return out

    def feedforward(self, data):

        self.activate[0] = np.insert(np.array([data]), len(data), 1, axis=1)
        
        for layer in xrange(1, len(self.activate)):
            self.activate[layer] = self.activate[layer-1].dot(self.weights[layer-1])
            self.activate[layer] = self.activate_func(self.activate[layer])
            self.activate[layer] = np.insert(self.activate[layer], self.activate[layer].size, 1, axis=1)

        result = self.activate[self.num_layers-1]
        result = np.delete(result, result.size - 1, 1)
        return result[0]

    def calError(self, desired):
        # error in output layer
        target = np.array([desired])
        a = self.activate[self.num_layers - 1]
        a = np.delete(a, a.size - 1, 1)
        e = target - a
        self.error[self.num_layers - 1] = e * self.deactivate_func(a)

        # error in other layer
        for lay in xrange(self.num_layers - 2, 0, -1): # each layer
            e = self.weights[lay].dot(self.error[lay + 1].T).T
            e = np.delete(e, e.size - 1, 1)
            a = self.activate[lay]
            a = np.delete(a, a.size - 1, 1)
            self.error[lay] = e * self.deactivate_func(a)

    def reweight(self):
        # reweight output layer
        lay = self.num_layers - 2
        self.weights[lay] = (
                self.weights[lay]
                + self.learn_alpha * self.activate[lay].T.dot(self.error[lay + 1]) 
                - 2 * self.learn_reg * self.weights[lay]
            )
        
        # reweight other layer
        for lay in reversed(xrange(self.num_layers - 2)):
            self.weights[lay] = (
                    self.weights[lay]
                    + self.learn_alpha * self.activate[lay].T.dot(self.error[lay + 1])
                    - 2 * self.learn_reg * self.weights[lay]
                )


    def act_sigmoid(self, x):
        return 1.0/(1.0+np.exp(-x))

    def act_desigmoid(self, y):
        return y*(1-y)

    def act_identity(self, x):
        return x

    def act_deidentity(self, y):
        return 1

    def loss_MSE(self, out_actual, out_desired):
        loss = 0.0
        for des, act in zip(out_desired, out_actual):
            loss += (des - act) * (des - act)
        return loss

    def loss_RMSE(self, out_actual, out_desired):
        loss = self.loss_MSE(out_actual, out_desired)
        loss = math.sqrt(loss)
        return loss

def example():
    print   '''------------------------------
>>> bpn = BPN(sizes = [3,3,3,1], learn_alpha = 0.001, learn_reg = 0.0, print_iter = 1000)
>>> for i in xrange(1000):
...     bpn.train([([1,1,1],[5])])
...     bpn.train([([-1,-1,-1],[-5])])
...     bpn.train([([5,1,1],[20])])
>>> bpn.test([1,1,1])
>>> bpn.test([-1,-1,-1])
>>> bpn.test([5,1,1])
--------------------'''
    bpn = BPN(sizes = [3,3,3,1], learn_alpha = 0.001, learn_reg = 0.0, print_iter = 1000)
    for i in xrange(1000):
        bpn.train([([1,1,1],[5])])
        bpn.train([([-1,-1,-1],[-5])])
        bpn.train([([5,1,1],[20])])
    bpn.test([1,1,1])
    bpn.test([-1,-1,-1])
    bpn.test([5,1,1])





