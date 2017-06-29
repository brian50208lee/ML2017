import numpy as np
from datetime import datetime
import sys
import collections

#private_num 9

def pre_process(data):
	ignore_features = [0, #id
	 13, #region_cod
	 14, #district_code
	 16, #ward
	 19, #recorded_by
	 21, #scheme_name
	 25, #extract_type_group
	 26, #extraction_type_class
	 30, #payment_type
	 31, #water_quality
	 34, #quantity_group
	 ]
	one_hot_features = [2, 3, 5, 8, 10, 11, 12, 15, 18, 20, 22, 24, 27, 28, 29, 32, 33, 35, 36, 37, 38, 39]
	replace_zero_to_mean = [4, 23]
	date_features = [2]
	X = []
	nb_features = data.shape[1]
	data_T = data.T
	for i in range(nb_features):
		print(i)
		if i in ignore_features:
			continue
		elif i in one_hot_features:
			feature = select_feature(data_T[i], 100)
		elif i in replace_zero_to_mean:
			feature = np.array(data_T[i], dtype='float').reshape(-1,1)
			feature[feature==0] = np.mean(feature)
		elif i in date_features:
			feature = process_date_feature(data_T[i])
		else:
			feature = np.array(data_T[i], dtype='float').reshape(-1,1)
		print(feature.shape)
		X.append(feature)
	
	X = np.hstack(X)	
	print(X.shape)
	return X

def one_hot_encoding(data, dic=None):
	if dic is None:
		dic = {}
		value = 0
		#build dic
		for i in range(data.shape[0]):
				if data[i] in dic:
					continue
				else:
					dic[data[i]] = value
					value += 1
	# do one of n encoding
	res = []
	n = max(dic.values()) + 1
	for i in range(data.shape[0]):
		x = np.zeros(n)
		x[dic[data[i]]] = 1
		res.append(x)

	return np.array(res)

def process_date_feature(data):
	res = []
	for i in range(data.shape[0]):
		d = datetime.strptime(data[i], "%Y-%m-%d")
		t = d.timestamp()
		res.append([t])
	return np.array(res)

def process_unique_feature(data):
	res = []
	dic = {}
	value = 0
	for i in range(data.shape[0]):
		if data[i] in dic:
			res.append([dic[data[i]]])
		else:
			dic[data[i]] = value
			res.append([value])
			value += 1
	return np.array(res)

def process_funder_feature(data):
	res = []
	for i in range(data.shape[0]):
		x = np.zeros(1)
		if data[i] == 'Government Of Tanzania':
			x[0] = 1
		res.append(x)
	return np.array(res)

def process_installer_feature(data):
	res = []
	for i in range(data.shape[0]):
		x = np.zeros(1)
		if data[i] == 'DWE':
			x[0] = 1

		res.append(x)
	return np.array(res)

def process_extraction_feature(data):
	res = []
	for i in range(data.shape[0]):
		x = np.zeros(1)
		if data[i] == 'gravity':
			x[0] = 1

		res.append(x)
	return np.array(res)

def process_wpt_feature(data):
	res = []
	for i in range(data.shape[0]):
		x = np.zeros(1)
		if data[i] == 'none':
			x[0] = 1

		res.append(x)
	return np.array(res)


def select_feature(data, number):
	dic = collections.OrderedDict()
	for i in range(59400,data.shape[0]):
		if data[i] == '':
			continue
		if data[i] in dic:
			dic[data[i]] += 1
		else:
			dic[data[i]] = 1
	nb_features = 0
	select_feature = collections.OrderedDict()
	for key, value in dic.items():
		if value > number:
			nb_features += 1
			select_feature[key] = nb_features
	#print(select_feature)
	res = []
	for i in range(data.shape[0]):
		x = np.zeros(nb_features + 1)
		if data[i] in select_feature:
			x[select_feature[data[i]]] = 1
		res.append(x)
	return np.array(res)


def load_data():
	train_data = []
	f = open(sys.argv[1])
	f.readline()
	for line in f:
		features = line[:-1].split(',')
		train_data.append(features)

	f.close()

	test_data = []
	f = open(sys.argv[3])
	f.readline()
	for line in f:
		features = line[:-1].split(',')
		test_data.append(features)
	f.close()

	nb_training_samples = len(train_data)
	data = train_data + test_data
	
	x_data = pre_process(np.array(data))

	x_train = x_data[:nb_training_samples]
	x_test = x_data[nb_training_samples:]

	return x_train, x_test


def load_target():
	res = []
	f = open(sys.argv[2])
	f.readline()
	for line in f:
		target = line.split(',')[1][:-1]
		t = np.zeros(3)
		if 'functional' == target:
			t = 0
		elif 'functional needs repair' == target:
			t = 1
		else:
			t = 2
		res.append(t)
	return np.array(res)

def predict(pred):
	f1 = open(sys.argv[4], 'w')
	f2 = open(sys.argv[3])
	
	f1.write('id,status_group\n')
	f2.readline()

	i = 0
	for line in f2:
		pump_id = int(line[:-1].split(',')[0])
		if pred[i] == 0:
			f1.write("{},{}\n".format(pump_id, 'functional'))
		elif pred[i] == 1:
			f1.write("{},{}\n".format(pump_id, 'functional needs repair'))
		else:
			f1.write("{},{}\n".format(pump_id, 'non functional'))
		#print(i)
		i += 1
	f1.close()
	f2.close()

def calculate_distribution():
	train_data = []
	fr = open(sys.argv[1])
	fr.readline()

	
	dic = {}
	for line in fr:
		features = line[:-1].split(',')
		feature = features[21]
		if feature in dic:
			dic[feature] += 1
		else:
			dic[feature] = 1
	fw = open('scheme_name_dis.txt', 'w')
	sorted_dic = sorted(dic.items(), key=lambda x:x[1])
	for key, value in sorted_dic:		
		fw.write('{} {}\n'.format(key, value))
	fr.close()

def save_data(x_train, y_train, test):
	np.save('x_train2.npy', x_train)
	np.save('y_train2.npy', y_train)
	np.save('test2.npy', test)
	sys.exit(1)