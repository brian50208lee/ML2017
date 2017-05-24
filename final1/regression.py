import sys, os
import numpy as np

from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout
from keras import backend as K
from keras.callbacks import EarlyStopping, ModelCheckpoint

train_data_path = 'data' + os.sep + 'train.csv'
test_data_path = sys.argv[1] if len(sys.argv) > 1 else 'data' + os.sep + 'test.csv'
predict_file_path = sys.argv[2] if len(sys.argv) > 2 else 'predict.csv'

def load_train_data(data_path=train_data_path):
	print('Load training data...')
	f = open(data_path, encoding='utf8')
	fileds = f.readline().strip().split(',')
	datas = []
	for line in f.readlines():
		datas.append(line.strip().split(','))
	fileds, datas = np.array(fileds), np.array(datas)
	return (fileds, datas)

def format_data(fields, datas, format_fields=[], field_type_dict=None):
	print('Format data...')
	new_datas = datas.copy()
	if field_type_dict is None:
		field_type_dict = {}

	new_datas[new_datas=='NA'] = 0
	new_datas[new_datas=='no'] = 0
	new_datas[new_datas=='yes'] = 1

	format_data = []
	for field, data_array in zip(fields, new_datas.T):
		if field in format_fields:
			data_array, types_dict = to_idx_array(data_array, field_type_dict.get(field))
			print('one_hot_encode:',field,'\tlen:',len(types_dict))
			data_array = to_one_hot_array(data_array, len(types_dict))
			field_type_dict[field] = types_dict
		else:
			data_array = data_array.reshape((len(data_array),-1))
		format_data.append(data_array)

	format_data = np.hstack(format_data).astype('float32')
	return format_data, field_type_dict

def feature_normalize(X):
	scale = np.max(X,axis=0) - np.min(X,axis=0)
	scale[scale==0] = 1
	return X / scale


def to_idx_array(data_array, type_dict=None):
	if type_dict is None: # create type_dict
		types = set(data_array)
		type_dict = dict(zip(types, range(len(types))))
	new_data_array = np.array([type_dict[data] for data in data_array])
	return new_data_array, type_dict

def to_one_hot_array(data_array, type_num):
	one_hot_data = np.array([[1 if idx == hot else 0 for idx in range(type_num)] for hot in data_array])
	return one_hot_data


	


def build_Regression_model(X, Y):
	print('Build RNN model...')
	model = Sequential()
	model.add(Dense(512, input_dim=X.shape[-1], activation='relu'))
	model.add(Dropout(0.3))
	model.add(Dense(256,activation='relu'))
	model.add(Dropout(0.3))
	model.add(Dense(128,activation='relu'))
	model.add(Dropout(0.3))
	model.add(Dense(64,activation='relu'))
	model.add(Dropout(0.3))
	model.add(Dense(32,activation='relu'))
	model.add(Dropout(0.3))
	model.add(Dense(16,activation='relu'))
	model.add(Dropout(0.3))
	model.add(Dense(Y.shape[-1]))
	model.compile(loss='mean_squared_logarithmic_error', optimizer='adadelta', metrics=[])
	model.summary()
	return model

def _save_model(path='./model/rnn.h5'):
	model.save(path)

def _load_model(path='./model/rnn.h5'):
	model = load_model(path, custom_objects={'f1_score':f1_score})
	return model


fileds, datas = load_train_data()
format_data, field_type_dict = format_data(fileds[2:], datas[:,2:], format_fields=['product_type','sub_area','ecology'])

X, Y = format_data[:,:-1], format_data[:,-1:]
X = feature_normalize(X)

valid_size = 1000
train_X, train_Y = X[valid_size:][:], Y[valid_size:][:]
valid_X, valid_Y = X[:valid_size], Y[:valid_size]

model = build_Regression_model(X,Y)
hist = model.fit(train_X, train_Y, batch_size=32, epochs=200, validation_data=(valid_X,valid_Y))


