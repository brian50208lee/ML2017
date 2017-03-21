import sys, random
import mytool.parser as parser
import mytool.BPN as BPN

# set file
trainfile = sys.argv[1] if len(sys.argv) > 1 else './data/train.csv'
testfile = sys.argv[2] if len(sys.argv) > 2 else './data/test_X.csv'
outfile = sys.argv[3] if len(sys.argv) > 3 else './res.csv'

# parse file
trainset, testset = parser.parse(trainfile, testfile, feature=[9], dim=1)

def train(maxiter=10):
	for i in xrange(maxiter):
		random.shuffle(trainset)
		reg.train(trainset)

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
reg = BPN.BPN(sizes=[dim,1], learn_alpha=0.000001, learn_reg=0.0, print_iter=len(trainset))
reg.load("./model_best/573638_bpn_9_1.npy")
output()