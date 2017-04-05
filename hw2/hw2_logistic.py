import sys, random
import mytool.parser as parser
import mytool.BPN as BPN

# set file
train = sys.argv[1] if len(sys.argv) > 1 else './data/train.csv'
test = sys.argv[2] if len(sys.argv) > 2 else './data/test.csv'
X_train = sys.argv[3] if len(sys.argv) > 3 else './data/X_train'
Y_train = sys.argv[4] if len(sys.argv) > 4 else './data/Y_train'
X_test = sys.argv[5] if len(sys.argv) > 5 else './data/X_test'
outfile = sys.argv[6] if len(sys.argv) > 6 else './res.csv'

# def function
def train(maxiter=10):
	for i in xrange(maxiter):
		random.shuffle(trainset)
		bpn.train(trainset)
		evaluate()

def train_sgd(maxiter=10):
	for i in xrange(maxiter):
		random.shuffle(trainset)
		bpn.train_sgd(trainset)
		evaluate()

def evaluate():
	bpn.evaluate(evaluset[:1000])
	bpn.evaluate(evaluset[1000:2000])
	bpn.evaluate(evaluset[2000:3000])
	bpn.evaluate(evaluset[3000:4000])

def output():
	out = open(outfile, 'w')
	out.write('id,label\n')
	for i in xrange(len(testset)):
		score = bpn.test(testset[i], printout=False)
		out.write('%d,%d\n' % (i+1, int(score+0.5)))
	out.close()

def save():
	bpn.save("./model/tmp")

def load():
	bpn.load("./model/tmp.npy")


# parse file
print "Parsing ..."
trainset, testset = parser.parse(X_train, Y_train, X_test)
trainset, testset = parser.parse_power(trainset, testset, power=3)
trainset, testset = parser.parse_normalize_mean(trainset, testset)
#trainset, testset = parser.parse_feature_scaling(trainset, testset)

evaluset = trainset[-4000:]
trainset = trainset[:]

# train
print "Training ..."
dim = len(trainset[0][0])
bpn = BPN.BPN(sizes=[dim,1], learn_rate=1, learn_reg=0.0, print_iter=len(trainset))
#train_sgd(100)
bpn.load("model_best/085577_3d.npy")
print "Output"
output()

