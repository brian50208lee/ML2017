import sys, os, random
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

def load_test_data(data_path=test_data_path):
	print('Load testing data...')
	f = open(data_path, encoding='utf8')
	fileds = f.readline().strip().split(',')
	datas = []
	for line in f.readlines():
		datas.append(line.strip().split(','))
	fileds, datas = np.array(fileds), np.array(datas)
	return (fileds, datas)

def format_data(fields, datas, categorical_fields=[], non_categorical_fields=[], field_type_dict=None):
	print('Format data...')
	# init
	new_datas = datas.copy()
	if field_type_dict is None:
		field_type_dict = {}

	# pre-process
	new_datas[new_datas=='NA'] = 0
	new_datas[new_datas=='no'] = 0
	new_datas[new_datas=='yes'] = 1

	for field, data_array in zip(fields, new_datas.T):
		if field == 'timestamp':
			for idx in range(len(data_array)):
				data_array[idx] = data_array[idx].split('-')[0]

	# categorical one-hot-encoding
	format_data = []
	for field, data_array in zip(fields, new_datas.T):
		if field in non_categorical_fields:
			data_array = data_array.reshape((len(data_array),-1))
		elif field in categorical_fields or any(cf[1:] in field for cf in categorical_fields if cf.startswith('*')):
			data_array, types_dict = to_idx_array(data_array, field_type_dict.get(field,None))
			print('categorical_fields:',field,'\tdict_len:',len(types_dict))
			data_array = to_one_hot_array(data_array, len(types_dict))
			field_type_dict[field] = types_dict
		else:
			data_array = data_array.reshape((len(data_array),-1))
		format_data.append(data_array)

	# check format
	for field, data_array in zip(fields, format_data):
		try:
			data_array.astype('float32')
		except:
			print('Unformat Filed:',field)

	format_data = np.hstack(format_data).astype('float32')
	return format_data, field_type_dict

def to_idx_array(data_array, type_dict=None):
	if type_dict is None: # create type_dict
		types = set(data_array)
		type_dict = dict(zip(types, range(len(types))))
	new_data_array = np.array([type_dict.get(data,0) for data in data_array])
	return new_data_array, type_dict

def to_one_hot_array(data_array, type_num):
	one_hot_data = np.array([[1 if idx == hot else 0 for idx in range(type_num)] for hot in data_array])
	return one_hot_data

def feature_normalize(X,scale=None):
	print('Feature normalize by scale...')
	if scale is None:
		scale = np.max(X,axis=0) - np.min(X,axis=0)
		scale[scale==0] = 1
	return (X / scale, scale)
	


def build_Regression_model(X, Y):
	print('Build RNN model...')
	model = Sequential()
	model.add(Dense(50, input_dim=X.shape[-1], activation='relu'))
	model.add(Dropout(0.4))
	model.add(Dense(50,activation='relu'))
	model.add(Dropout(0.4))
	model.add(Dense(50,activation='relu'))
	model.add(Dropout(0.2))
	model.add(Dense(Y.shape[-1]))
	model.compile(loss='mean_squared_logarithmic_error', optimizer='adam', metrics=[])
	model.summary()
	return model

def _save_model(path='./model/rnn.h5'):
	model.save(path)

def _load_model(path='./model/regression_0.247.h5'):
	#model = load_model(path, custom_objects={'f1_score':f1_score})
	model = load_model(path)
	return model


fileds, datas = load_train_data()
categorical_fields = ['product_type','sub_area','ecology']
non_categorical_fields = []
'''
categorical_fields  = ['timestamp','floor','max_floor','material','build_year']
categorical_fields += ['num_room','kitch_sq','state','product_type','sub_area']
categorical_fields += ['*raion','*id']
categorical_fields += ['culture_objects_top_25','water_1line','big_road1_1line','railroad_1line','ecology']
non_categorical_fields = ['raion_popul']
'''
formated_data, field_type_dict = format_data(fileds[2:], datas[:,2:], categorical_fields=categorical_fields,non_categorical_fields=non_categorical_fields)

X, Y = formated_data[:,:-1], formated_data[:,-1:]
X, scale = feature_normalize(X)

pair = list(zip(X,Y))
random.shuffle(pair)
X, Y = zip(*pair)
X = np.array(X).astype('float32')
Y = np.array(Y).astype('float32')

valid_size = 1000
train_X, train_Y = X[valid_size:][:], Y[valid_size:][:]
valid_X, valid_Y = X[:valid_size], Y[:valid_size]


model = build_Regression_model(X,Y)
earlystopping = EarlyStopping(monitor='val_loss', patience = 20, verbose=0, mode='min')
checkpoint = ModelCheckpoint(
				filepath='./model/regression_{val_loss:.3f}.h5',
				verbose=1,
				save_best_only=True,
				save_weights_only=False,
				monitor='val_loss',
				mode='min'
			)
hist = model.fit(train_X, train_Y, batch_size=32, epochs=10000, validation_data=(valid_X,valid_Y), callbacks=[earlystopping,checkpoint])

#model = _load_model()
# output
fileds, datas = load_test_data()
ids = datas[:,:1]
ids.reshape(-1)
formated_data, field_type_dict = format_data(fileds[2:], datas[:,2:], categorical_fields=categorical_fields,non_categorical_fields=non_categorical_fields,field_type_dict=field_type_dict)
test_X, _ = feature_normalize(formated_data,scale)


print('Output')
preds = model.predict(test_X)
out = open(predict_file_path,'w')
out.write('id,price_doc\n')
for _id, _pred in zip(ids,preds):
	out.write('{},{}\n'.format(_id[0],_pred[0]))
out.close()

