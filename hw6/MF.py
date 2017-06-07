import os, sys
import pandas as pd
import numpy as np
from keras.models import Sequential, Model, load_model
from keras.layers import Dense, Dropout, Input, Embedding, Flatten, Dot, Add, Concatenate
from keras.callbacks import EarlyStopping, ModelCheckpoint

# path
data_train = 'train.csv'
data_test = 'test.csv'
data_users = 'users.csv'
data_movies = 'movies.csv'
data_directory = sys.argv[1] + os.sep if len(sys.argv) > 1 else './data/'
prediction_file = sys.argv[2] if len(sys.argv) > 2 else 'predict.csv'

# user parameter
K_FACTORS = 300
best_model_weight_path = './model/mf_weight_best.h5'

# load data
df_train = pd.read_csv(data_directory + data_train)
df_test = pd.read_csv(data_directory + data_test)
df_users = pd.read_csv(data_directory + data_users, sep='::')
df_movies = pd.read_csv(data_directory + data_movies, sep='::')

# get numpy array for train and test
train_UserID = df_train['UserID'].values
train_MovieID = df_train['MovieID'].values
train_Rating = df_train['Rating'].values.astype('float32')

test_DataID = df_test['TestDataID'].values
test_UserID = df_test['UserID'].values
test_MovieID = df_test['MovieID'].values

# max id
max_userid = np.max(train_UserID).astype('int')
max_movieid = np.max(train_MovieID).astype('int')

# shuffle train data
indices = np.arange(train_UserID.shape[0])  
np.random.shuffle(indices) 
train_UserID = train_UserID[indices]
train_MovieID = train_MovieID[indices]
train_Rating = train_Rating[indices]

train_Rating = (train_Rating - 3.581712) / 1.1168977

# split validation
valid_size = 5000
valid_UserID = train_UserID[:valid_size]
valid_MovieID = train_MovieID[:valid_size]
valid_Rating = train_Rating[:valid_size]
train_UserID  = train_UserID[valid_size:]
train_MovieID  = train_MovieID[valid_size:]
train_Rating  = train_Rating[valid_size:]



# data info
print('-'*30)
print('valid_UserID:', valid_UserID.shape)
print('valid_MovieID:', valid_MovieID.shape)
print('valid_Rating:', valid_Rating.shape)
print('train_UserID:', train_UserID.shape)
print('train_MovieID:', train_MovieID.shape)
print('train_Rating:', train_Rating.shape)
print('test_UserID:', test_UserID.shape)
print('test_MovieID:', test_MovieID.shape)
print('-'*30)

# build model
def build_model():
	# input
	input_user = Input(shape=(1,))
	input_movie = Input(shape=(1,))

	# user, movie ,user bias, movie bias 
	emb_user = Embedding(max_userid + 1, K_FACTORS, input_length=1)(input_user)
	emb_movie = Embedding(max_movieid + 1, K_FACTORS, input_length=1)(input_movie)
	emb_user_bias = Embedding(max_userid + 1, 1, input_length=1)(input_user)
	emb_movie_bias = Embedding(max_movieid + 1, 1, input_length=1)(input_movie)

	# flatten
	U = Flatten()(emb_user)
	M = Flatten()(emb_movie)	
	U_b = Flatten()(emb_user_bias)
	M_b = Flatten()(emb_movie_bias)

	# MF
	'''
	# out = (user dot movie) + user_bias + movie_bias
	out_lay = Dot(axes=1)([U, M])
	out_lay = Add()([out_lay, U_b, M_b])
	model = Model([input_user, input_movie], out_lay)
	'''

	# DNN
	out_lay = Concatenate()([U, M])
	out_lay = Dense(100, activation='relu')(out_lay)
	out_lay = Dropout(0.5)(out_lay)
	out_lay = Dense(50, activation='relu')(out_lay)
	out_lay = Dropout(0.5)(out_lay)
	out_lay = Dense(1)(out_lay)
	out_lay = Add()([out_lay, U_b])
	model = Model([input_user, input_movie], out_lay)

	# compile
	model.compile(loss='mean_squared_error', optimizer='adam')
	model.summary()
	return model
	
model = build_model()

# train
earlystopping = EarlyStopping(monitor='val_loss', patience=3, verbose=0, mode='min')
checkpoint = ModelCheckpoint(
				filepath=best_model_weight_path,
				verbose=1,
				save_best_only=True,
				save_weights_only=True,
				monitor='val_loss',
				mode='min'
			)
model.fit(
		[train_UserID, train_MovieID], train_Rating, 
		batch_size=5000, epochs=1000, 
		validation_data=([valid_UserID,valid_MovieID],valid_Rating), 
		callbacks=[earlystopping, checkpoint]
	)

# load best model
model = build_model()
model.load_weights(best_model_weight_path)

# predict
y_preds = model.predict([test_UserID, test_MovieID]).reshape(-1)
y_preds = y_preds * 1.1168977 + 3.581712
y_preds = np.clip(y_preds, 0, 5)
df_pred = pd.DataFrame({'TestDataID': test_DataID, 'Rating': y_preds})
df_pred.to_csv(prediction_file, columns=['TestDataID','Rating'], index=False)





