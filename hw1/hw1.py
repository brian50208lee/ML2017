import sys, random
import mytool.parser as parser
import mytool.LR as LR

# set file
trainfile = sys.argv[1] if len(sys.argv) > 1 else './data/train.csv'
testfile = sys.argv[2] if len(sys.argv) > 2 else './data/test_X.csv'
outfile = sys.argv[3] if len(sys.argv) > 3 else './res.csv'

# parse file
trainset, testset = parser.parse(trainfile, testfile, feature=[9], dim=1)
evaluset = trainset[-1000:]
trainset = trainset[:-1000]

def train(maxiter=10):
	for i in xrange(maxiter):
		random.shuffle(trainset)
		reg.train(trainset)
	print reg.evaluate(evaluset)

def traingroup(maxiter=10,batchsize=5):
	for i in xrange(maxiter):
		random.shuffle(trainset)
		reg.traingroup(trainset[:batchsize])

def output():
	out = open(outfile, 'w')
	out.write('id,value\n')
	for i in xrange(len(testset)):
		out.write('id_%d,%f\n' % (i, reg.test(testset[i], printout=False)))
	out.close()


dim = len(trainset[0][0])
reg = LR.LR(sizes=[dim,1], learn_alpha=0.00001, learn_reg=0.00000, print_iter=len(trainset))
train(20)
print reg.evaluate(evaluset)

