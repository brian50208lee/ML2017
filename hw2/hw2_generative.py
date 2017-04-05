import sys, random
import mytool.parser as parser
import mytool.GM as GM

# set file
train = sys.argv[1] if len(sys.argv) > 1 else './data/train.csv'
test = sys.argv[2] if len(sys.argv) > 2 else './data/test.csv'
X_train = sys.argv[3] if len(sys.argv) > 3 else './data/X_train'
Y_train = sys.argv[4] if len(sys.argv) > 4 else './data/Y_train'
X_test = sys.argv[5] if len(sys.argv) > 5 else './data/X_test'
outfile = sys.argv[6] if len(sys.argv) > 6 else './res.csv'

def output():
	out = open(outfile, 'w')
	out.write('id,label\n')
	for i in xrange(len(m_testset)):
		score = gm.test(m_testset[i], printout=False)[0]
		out.write('%d,%d\n' % (i+1, int(score+0.5)))
	out.close()
	
def save():
	gm.save("./model/tmp")

def load():
	gm.load("./model/tmp.npy")


# parse file
print "Parsing ..."
trainset, testset = parser.parse(X_train, Y_train, X_test)
m_trainset, m_testset = parser.parse_power(trainset, testset, power=3)
m_trainset, m_testset = parser.parse_normalize_mean(m_trainset, m_testset)
#m_trainset, m_testset = parser.parse_feature_scaling(m_trainset, m_testset)
#m_trainset, m_testset = parser.parse_feature(trainset, testset, range(106))
m_evaluset = m_trainset[-5000:]
m_trainset = m_trainset[:]

# train
dim = len(m_trainset[0][0])
gm = GM.GM(sizes=[dim,1])

print "Training ..."
gm.train(m_trainset)
gm.evaluate(m_evaluset)

output()