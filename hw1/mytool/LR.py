import random
import numpy as np

class LRegression(object):

    def __init__(self, indim, learn_rate = 0.5):
        # set info
        self.indim = indim
        self.learn_rate = learn_rate

        # init weights [w1, ... , wn, wb]
        self.weights = np.random.randn(self.indim+1)
        
        # tmp training data [a1, ... , an, base = 1]
        self.data = np.array([])
        self.output = 0.0
        self.loss = 0.0

        # error
        self.error = 0.0
        self.count = 0

    def train(self, trainset):
        for data, desired in trainset:
            self.data = np.array(data)
            self.feedforward(data)
            self.reweight(desired)
            self.output = self.lossfunction(desired)
            #print 'loss: ', self.output

            self.count += 1
            if self.count < 5000:
                self.error += self.output
            else:
                self.error /= 5000
                print self.error
                self.count = 0
                self.error = 0.0

    def test(self, data):
        self.data = np.array(data)
        self.feedforward(data)
        return self.output

    def save(self, filepath):
        np.save(filepath, self.weights)

    def load(self, filepath):
        self.weights = np.load(filepath)

    def feedforward(self, data):
        self.data = np.insert(self.data, self.data.size, 1)
        self.output = self.data.dot(self.weights.T)

    def reweight(self, desired):
        self.weights = self.weights + 2*self.learn_rate*(desired - self.output)*self.data

    def lossfunction(self, out_desired):
        return (out_desired - self.output) * (out_desired - self.output)

'''
train = open('train.csv', 'r')
train.readline() # ignore first line

# parsing csv to hour perline
dataset = np.array([[]]).reshape(0,18)
count = 0
line18 = []
for line in train:
    line18.append([s.replace('NR','0').strip() for s in line.split(',')[3:]])
    
    count += 1
    if count == 18:
        dataset = np.append(dataset, np.array(line18).T, axis = 0)
        count = 0
        line18 = []

# parsing to training input format
trainset = []
for i in xrange(len(dataset)-9):
    data = dataset[i:i+9].reshape(dataset[i:i+9].size).astype(float).tolist()
    data = data[9::18]
    #print data[9::18]
    target = [dataset[i+9,9].astype(float)]
    trainset.append((data, target))


# train
LR = LRegression(9, 0.000005)
#LR.load('./model/lr9.npy')

for i in xrange(60):
    LR.train(trainset)




test_X = open('test_X.csv', 'r')
# parsing csv to hour perline
dataset = np.array([[]]).reshape(0,18)
count = 0
line18 = []
for line in test_X:
    line18.append([s.replace('NR','0').strip() for s in line.split(',')[2:]])
    count += 1
    if count == 18:
        dataset = np.append(dataset, np.array(line18).T, axis = 0)
        count = 0
        line18 = []

# parsing to training input format
testset = []
for i in xrange(len(dataset)/9):
    data = dataset[i*9:i*9+9].reshape(dataset[i*9:i*9+9].size).astype(float).tolist()
    testset.append(data)

for i in xrange(len(testset)):
    testset[i] = testset[i][9::18]

out = open('result.csv','w')
out.write("id,value\n")
for i in xrange(len(testset)):
    out.write( 'id_%d,%f\n' % (i, LR.test(testset[i])))
out.close()
'''


