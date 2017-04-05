import random,math
import numpy as np

class BPN(object):
    def __init__(self, sizes, learn_rate = 1, learn_reg = 0.0, dec_adadelta = 0.95 , print_iter = 1000):
        # set info
        self.sizes = sizes
        self.lay_num = len(sizes)
        self.learn_rate = learn_rate
        self.learn_reg = learn_reg
        self.adadelta_dec = dec_adadelta
        self.adadelta_eps = 1E-8

        # set activate function and loss functino
        self.activate_func = self.act_sigmoid
        self.deactivate_func = self.act_desigmoid
        self.loss_func = self.loss_MSE

        # init weights [w1, ... , wn, wb]
        self.weights = [np.random.randn(x+1, y) for x, y in zip(sizes[:-1], sizes[1:])]
        self.adadelta_G = [np.zeros((x+1, y)) for x, y in zip(sizes[:-1], sizes[1:])]
        self.adadelta_D = [np.zeros((x+1, y)) for x, y in zip(sizes[:-1], sizes[1:])]
       
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
        err = [0.0 for i in xrange(len(self.error))]
        for x, y in trainset:
            o = self.feedforward(x)
            self.calError(y)
            err = [err[i] + self.error[i] for i in xrange(len(self.error))]
            self.iter += 1
        err = [e/len(trainset) for e in err]
        self.error = err
        self.reweight()

        # print error
        print "iter: %-10dAvgError: %e" % (self.iter, err[-1])

    def train_sgd(self, trainset):
        for x, y in trainset:
            o = self.feedforward(x)
            self.calError(y)
            self.reweight()
            
            # updata error
            self.iter += 1
            self.loss += self.loss_func(o, y)
            if (self.iter % self.print_iter) == 0:
                self.loss /= self.print_iter
                print "iter: %-10dAvgError: %e" % (self.iter, self.loss)
                self.loss = 0.0

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

    def feedforward(self, x):
        # init input lay
        self.activate[0] = np.array([x])
        
        # forward
        for lay in xrange(0, self.lay_num - 1):
            self.activate[lay] = np.append(self.activate[lay], [[1]], axis=1)
            self.activate[lay+1] = self.activate[lay].dot(self.weights[lay])
            self.activate[lay+1] = self.activate_func(self.activate[lay+1])
            self.activate[lay] = np.delete(self.activate[lay], -1, axis=1)
        
        return self.activate[self.lay_num - 1][0]

    def calError(self, y):
        # error in output layer
        a = self.activate[self.lay_num - 1]
        e = np.array([y]) - a
        self.error[self.lay_num - 1] = e * self.deactivate_func(a)

        # error in other layer
        for lay in xrange(self.lay_num - 2, 0, -1): # n - 2 to 1
            a = self.activate[lay]
            e = self.weights[lay][:-1].dot(self.error[lay + 1].T).T
            self.error[lay] = e * self.deactivate_func(a)

    def reweight(self):
        # reweight other layer
        for lay in xrange(self.lay_num - 2, -1, -1): # n - 2 to 0
            # gradient
            g = np.append(self.activate[lay], [[1]], axis=1).T.dot(self.error[lay + 1])
     
            
            # delta w
            self.adadelta_G[lay] = self.adadelta_dec*self.adadelta_G[lay] + (1-self.adadelta_dec)*g*g
            delta_w =   (
                            - np.sqrt(self.adadelta_D[lay] + self.adadelta_eps) 
                            / np.sqrt(self.adadelta_G[lay] + self.adadelta_eps) 
                            * g
                            + self.learn_reg * self.weights[lay]
                        )
            self.adadelta_D[lay] = self.adadelta_dec*self.adadelta_D[lay] + (1-self.adadelta_dec)*delta_w*delta_w
            # reweight
            self.weights[lay] -= self.learn_rate * delta_w
            

    def act_sigmoid(self, x):
        x = np.clip(x, -100, 100)
        return 1.0/(1.0+np.exp(-x))

    def act_desigmoid(self, y):
        return y*(1-y)

    def act_ReLU(self, z):
        return np.maximum(0,z)

    def act_deReLU(self, z):
        return np.minimum(1,np.maximum(0,z))

    def act_identity(self, x):
        return x

    def act_deidentity(self, y):
        return 1

    def loss_cross_entropy(self, o, y):
        loss = 0.0
        # unimplement
        return loss

    def loss_MSE(self, o, y):
        loss = 0.0
        for act, des in zip(o, y):
            loss += (act - des) * (act - des)
        return loss

    def loss_RMSE(self, o, y):
        loss = self.loss_MSE(o, y)
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
        bpn.train([([1,1,1],[1])])
        bpn.train([([-1,-1,-1],[0])])
        bpn.train([([5,1,1],[0])])
    bpn.test([1,1,1])
    bpn.test([-1,-1,-1])
    bpn.test([5,1,1])





