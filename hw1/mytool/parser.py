import numpy as np

def parse(trainfile, testfile, feature=xrange(18), dim=1):
	# parse training data => train = [([d11,...,d1n],[t11,...,t1m]),...,([dk1,...,dkn],[tk1,...,tkm])]
	# format train = [(data list, target list), ...]
	train = np.array([line.strip().replace('NR','0').split(',')[3:] for line in open(trainfile)]).astype(float)[1:]
	train = np.vstack([arr.reshape(18,24).T.reshape(-1,18).tolist() for arr in train.reshape([-1,18*24])])
	train = [(train[i:i+9].reshape(-1).tolist(), [train[i+9][9]]) for i in xrange(len(train)-9)]
	
	# parse testing data => test = [[d11,...,d1n],...,[dk1,...,dkn]]
	# format train = [data list, ...]
	test = np.array([line.strip().replace('NR','0').split(',')[2:] for line in open(testfile)]).astype(float)
	test = np.vstack([arr.reshape(18,9).T.reshape(-1,18)  for arr in test.reshape([-1,18*9])])
	test = test.reshape(-1,18*9).tolist()

	# extract feature => train = [([f11,...,f1n'],[t11,...,t1m]),...,([fk1,...,fkn'],[tk1,...,tkm])]
	# extract feature => test = [[f11,...,f1n'],...,[fk1,...,fkn']]
	train = [(np.array(data[0]).reshape(9,-1)[:,feature].reshape(-1).tolist(), data[1]) for data in train]
	test = [np.array(data).reshape(9,-1)[:,feature].reshape(-1).tolist() for data in test]

	# change data dimension => train = [([f11,...,f11**dim,...,f1n',...f1n'**dim],[t11,...,t1m]),...,([fk1,...,fk1**dim,...,fkn',...,fkn'***dim],[tk1,...,tkm])]
	# change data dimension => test = [[f11,...,f11**dim],...,[fkn',...fkn'**dim]]
	train = [([e**p for e in data[0] for p in xrange(1,dim+1)], data[1])for data in train]
	test = [[e**p for e in data for p in xrange(1,dim+1)] for data in test]

	return (train, test)