import numpy as np

def parse(X_train, Y_train, X_test, feature=range(106), dim=1):
	train_X = np.array([line.strip().split(',') for line in open(X_train)])[1:].astype(float)
	train_Y = np.array([line.strip().split(',') for line in open(Y_train)]).astype(float)
	train = [(train_X[i].tolist(), train_Y[i].tolist())for i in xrange(len(train_X))]

	test = np.array([line.strip().split(',') for line in open(X_test)])[1:].astype(float)
	test = test.tolist()

	# extract feature => train = [([f11,...,f1n'],[t11,...,t1m]),...,([fk1,...,fkn'],[tk1,...,tkm])]
	# extract feature => test = [[f11,...,f1n'],...,[fk1,...,fkn']]
	data_dim = len(train[0][0])
	train = [(np.array(data[0])[feature].tolist(), data[1]) for data in train]
	test = [np.array(data)[feature].tolist() for data in test]

	# change data dimension => train = [([f11,...,f11**dim,...,f1n',...f1n'**dim],[t11,...,t1m]),...,([fk1,...,fk1**dim,...,fkn',...,fkn'***dim],[tk1,...,tkm])]
	# change data dimension => test = [[f11,...,f11**dim],...,[fkn',...fkn'**dim]]
	train = [([e**p for e in data[0] for p in xrange(1,dim+1)], data[1])for data in train]
	test = [[e**p for e in data for p in xrange(1,dim+1)] for data in test]

	return (train, test)