import sys, random
import mytool.parser as parser
import mytool.BPN as BPN
import mytool.LR as LR

# set file
trainfile = sys.argv[1] if len(sys.argv) > 1 else './data/train.csv'
testfile = sys.argv[2] if len(sys.argv) > 2 else './data/test_X.csv'
outfile = sys.argv[3] if len(sys.argv) > 3 else './res.csv'

# parse file
trainset, testset = parser.parse(trainfile, testfile, feature=[9], dim=1)
evaluset = trainset[4000:]
trainset = trainset[:4000]


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
reg = BPN.BPN(sizes=[dim,1], learn_alpha=0.0000001, learn_reg=0.0, print_iter=len(trainset))
try:
	reg.load("./model/best.npy")
except:
	reg.save("./model/best")

best = reg.evaluate(evaluset)
print "best:",best

while True:
	reg = BPN.BPN(sizes=[dim,1], learn_alpha=0.0000001, learn_reg=0.0, print_iter=len(trainset))
	reg.load("./model/best.npy")
	reg.save("./model/tmp")
	error = reg.evaluate(evaluset)
	for i in xrange(10):
		count = 0
		while reg.evaluate(evaluset) - error >= 0 and count < 10:
			count += 1
			reg.load("./model/tmp.npy")
			train(1)
		if reg.evaluate(evaluset) - error < 0:
			reg.save("./model/tmp")
			print "save time:",i
		reg.load("./model/tmp.npy")
		error = reg.evaluate(evaluset)
	if error < best:
		best = error
		reg.save("./model/best")
		print "best:",best



'''
best_error = 9999999
for i in [x for x in xrange(2, 201) for i in range(3)]:
	# init bpn
	dim = len(trainset[0][0])
	reg = BPN.BPN(sizes=[dim,i,1], learn_alpha=0.000001, learn_reg=0.0000000, print_iter=len(trainset))
	pre_error = 9999999
	error = reg.evaluate(trainset)
	while abs(pre_error - error) > 0.3:
		train()
		pre_error = error
		error = reg.evaluate(trainset)
		print "error change: ",abs(pre_error - error)
	if error < best_error:
		print "save=>",i
		best_error = error
		reg.save("./model/best")
'''
