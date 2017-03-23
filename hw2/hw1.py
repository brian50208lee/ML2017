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
		reg.train(trainset)
	print reg.evaluate(evaluset)

def traingroup(maxiter=10,batchsize=5):
	for i in xrange(maxiter):
		random.shuffle(trainset)
		reg.traingroup(trainset[:batchsize])

def output():
	out = open(outfile, 'w')
	out.write('id,label\n')
	for i in xrange(len(testset)):
		score = reg.test(testset[i], printout=False)
		#out.write('%d,%d\n' % (i+1, int(score+0.5)))
		out.write('%d,%f\n' % (i+1, score))
	out.close()

# parse file
feature = list(set(range(106)).difference(set([1,3,4])))
trainset, testset = parser.parse(X_train, Y_train, X_test, feature, dim=1)
evaluset = trainset[-5000:]
trainset = trainset[:-5000]
#random.shuffle(trainset)
trainset = trainset[:500]

# train
dim = len(trainset[0][0])
reg = BPN.BPN(sizes=[dim,100,1], learn_alpha=0.001, learn_reg=0.00, print_iter=len(trainset))
train(50)

'''
#test
reg = BPN.BPN(sizes=[2,2,1], learn_alpha=1, learn_reg=0.0, print_iter=100)
trainset = [([10,1],[1])
			,([-1,-10],[1])
			,([1,-1],[0])
			,([-1,1],[0])]
train(5000)
for data in trainset:
	print reg.test(data[0])
'''
