import numpy as np

def parse(fname, dim=1):
	data = [line.strip().split(',') for line in open(fname)][1:]
	X = np.array([x_part.split() for y_part, x_part in data]).astype(float)
	X = np.reshape(X, (-1,48,48,1))
	Y = np.array([1 if idx == int(y_part) else 0 for y_part, x_part in data for idx in xrange(7)])
	Y = np.reshape(Y, (-1,7))
	return (X, Y)

def parse_feature(train, test, feature=range(106)):
	data_dim = len(train[0][0])
	m_train = [(np.array(data[0])[feature].tolist(), data[1]) for data in train]
	m_test = [np.array(data)[feature].tolist() for data in test]
	return (m_train, m_test)

def parse_power(train, test, power=1):
	m_train = [([e**p for e in data[0] for p in xrange(1,power+1)], data[1])for data in train]
	m_test = [[e**p for e in data for p in xrange(1,power+1)] for data in test]
	return (m_train, m_test)

def parse_normalize_min(train, test):
	data_np = np.vstack([data[0] for data in train])
	data_min = np.amin(data_np, axis=0)
	data_sca = np.amax(data_np, axis=0) - data_min
	data_sca = np.array([sca if sca != 0 else 1 for sca in data_sca])
	
	m_train = [( ((np.array(data[0]) - data_min)/data_sca).tolist() , data[1]) for data in train]
	m_test = [((np.array(data) - data_min)/data_sca).tolist() for data in test]
	return (m_train, m_test)

def parse_normalize_mean(train, test):
	data_np = np.vstack([data[0] for data in train])
	data_min = np.amin(data_np, axis=0)
	data_mean = np.mean(data_np, axis=0)
	data_sca = np.amax(data_np, axis=0) - data_min
	data_sca = np.array([sca if sca != 0 else 1 for sca in data_sca])
	
	m_train = [( ((np.array(data[0]) - data_mean)/data_sca).tolist() , data[1]) for data in train]
	m_test = [((np.array(data) - data_mean)/data_sca).tolist() for data in test]
	return (m_train, m_test)

def parse_feature_scaling(train, test):
	data_np = np.vstack([data[0] for data in train])
	data_mean = np.mean(data_np, axis=0)
	data_std = np.sqrt(np.sum((data_np-data_mean)*(data_np-data_mean),axis=0))
	data_std = np.array([std if std != 0 else 1 for std in data_std])
	m_train = [(((np.array(data[0]) - data_mean)/data_std).tolist() , data[1]) for data in train]
	m_test = [((np.array(data) - data_mean)/data_std).tolist() for data in test]
	return (m_train, m_test)

def pca(X,k):#k is the components you want
	#mean of each feature
	n_samples, n_features = X.shape
	mean=np.array([np.mean(X[:,i]) for i in range(n_features)])
	#normalization
	norm_X=X-mean
	#scatter matrix
	scatter_matrix=np.dot(np.transpose(norm_X),norm_X)
	#Calculate the eigenvectors and eigenvalues
	eig_val, eig_vec = np.linalg.eig(scatter_matrix)
	eig_pairs = [(np.abs(eig_val[i]), eig_vec[:,i]) for i in range(n_features)]
	# sort eig_vec based on eig_val from highest to lowest
	eig_pairs.sort(reverse=True)
	# select the top k eig_vec
	feature=np.array([ele[1] for ele in eig_pairs[:k]])
	#get new data
	data=np.dot(norm_X,np.transpose(feature))
	return data



