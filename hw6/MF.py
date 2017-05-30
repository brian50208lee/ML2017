import os, sys

import pandas as pd
import numpy as np
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout
from keras.callbacks import EarlyStopping, ModelCheckpoint

data_train = 'train.csv'
data_test = 'test.csv'
data_users = 'users.csv'
data_movies = 'movies.csv'
data_directory = sys.argv[1] + os.sep if len(sys.argv) > 1 else './data/'
prediction_file = sys.argv[2] if len(sys.argv) > 2 else 'predict.csv'

# load data
df_train = pd.read_csv(data_directory + data_train)
df_test = pd.read_csv(data_directory + data_test)
df_users = pd.read_csv(data_directory + data_users, sep='::')
df_movies = pd.read_csv(data_directory + data_movies, sep='::')

# fix name mapping
df_movies.rename(columns={'movieID':'MovieID'}, inplace=True)

# concate train and test
train_Y = df_train['Rating'].values.astype('float32')
test_id = df_test['TestDataID'].values
df_train.drop(labels=['Rating','TrainDataID'], axis=1 ,inplace=True)
df_test.drop(labels=['TestDataID'], axis=1 ,inplace=True)
df_all = pd.concat([df_train, df_test])

# merge
df_all = pd.merge_ordered(left=df_all, right=df_users, on='UserID', how='left')
df_all = pd.merge_ordered(left=df_all, right=df_movies, on='MovieID', how='left')

# drop
drop_labels  = []
drop_labels += ['Zip-code', 'UserID', 'MovieID', 'Title']
df_all.drop(labels=drop_labels, axis=1 ,inplace=True)

# process categorical
categorical_labels = []
categorical_labels += [('Gender'), 'Occupation']
for cat_label in categorical_labels:
	df_cat = pd.get_dummies(data=df_all[cat_label], prefix=cat_label)
	df_all = pd.concat([df_all, df_cat], axis=1)
	df_all.drop(labels=[cat_label], axis=1, inplace=True)

# process multilabels
mult_labels = []
mult_labels += [('Genres','|')]
for mult_label, sep in mult_labels:
	df_mult = df_all[mult_label].str.get_dummies(sep=sep)
	prefix = mult_label + '_'
	df_mult.rename(columns=dict(zip(df_mult.keys(),prefix + df_mult.keys())), inplace=True)
	df_all = pd.concat([df_all, df_mult], axis=1)
	df_all.drop(labels=[mult_label], axis=1, inplace=True)


# train, test
all_X = df_all.values.astype('float32')
train_X, test_X = all_X[:len(df_train)], all_X[len(df_train):]

# validation
indices = np.arange(train_X.shape[0])  
np.random.shuffle(indices) 
train_X, train_Y = train_X[indices], train_Y[indices]
valid_size = 5000
valid_X, valid_Y = train_X[:valid_size], train_Y[:valid_size]
train_X, train_Y = train_X[valid_size:], train_Y[valid_size:]

print('train_X:', train_X.shape)
print('train_Y:', train_Y.shape)
print('test_X:', test_X.shape)

# build model
model = Sequential()
model.add(Dense(128, input_dim=train_X.shape[-1], activation='relu'))
model.add(Dropout(0.4))
model.add(Dense(128,activation='relu'))
model.add(Dropout(0.4))
model.add(Dense(128,activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mean_squared_error'])
model.summary()

# train
earlystopping = EarlyStopping(monitor='val_loss', patience=20, verbose=0, mode='min')
checkpoint = ModelCheckpoint(
				filepath='./model/nn_best.h5',
				verbose=1,
				save_best_only=True,
				save_weights_only=False,
				monitor='val_loss',
				mode='min'
			)
model.fit(train_X, train_Y,batch_size=1024, epochs=1, validation_data=(valid_X,valid_Y), callbacks=[earlystopping, checkpoint])

# predict
model = load_model('./model/nn_best.h5')
y_preds = model.predict(test_X).reshape(-1)
df_pred = pd.DataFrame({'TestDataID': test_id, 'Rating': y_preds})
df_pred.to_csv(prediction_file, columns=['TestDataID','Rating'], index=False)





