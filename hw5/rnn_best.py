import sys, os
import numpy as np

from keras.preprocessing import text, sequence
from keras.models import Sequential, load_model
from keras.layers import Embedding, LSTM, GRU, Dense, Dropout
from keras.optimizers import Adam
from keras import backend as K

train_data_path = 'data' + os.sep + 'train_data.csv'
test_data_path = sys.argv[1] if len(sys.argv) > 1 else 'data' + os.sep + 'test_data.csv'
predict_file_path = sys.argv[2] if len(sys.argv) > 2 else 'predict.csv'

MAX_SEQUENCE_LENGTH = 306
TAG_INDEX_PATH = './model/049580/tag_index.npy'
WORD_INDEX_PATH = './model/049580/word_index.npy'

def load_train_data(data_path=train_data_path):
	print('Load training data...')
	ids, tags, texts = [], [], []
	for line in open(data_path, encoding='utf8').readlines()[1:]:
		_id, _tag, _text = line.strip().split(',',2)
		_tag = _tag.strip('"').split()
		ids.append(_id)
		tags.append(_tag)
		texts.append(_text)
	ids, tags, texts = np.array(ids), np.array(tags), np.array(texts)
	print('Texts shape:',texts.shape)
	print('Tags shape:',tags.shape)
	return (ids, tags, texts)

def load_test_data(data_path=test_data_path):
	print('Load testing data...')
	ids, texts = [], []
	for line in open(data_path, encoding='utf8').readlines()[1:]:
		_id, _text = line.strip().split(',',1)
		ids.append(_id)
		texts.append(_text)
	ids, texts = np.array(ids), np.array(texts)
	print('Texts shape:',texts.shape)
	return (ids, texts)

def save_dict(myDict, path):
	print('Saving dictionary to:', path)
	np.save(path, myDict)

def load_dict(path):
	print('Loading dictionary from:', path)
	myDict = np.load(path).item()
	return myDict

def tags_mapping(tags, load=False, path=TAG_INDEX_PATH):
	print ('Tags mapping...')
	if load:
		tag_index = load_dict(path) # load mapping dict
	else:
		tagset = list(set([t for tag in tags for t in tag]))
		tag_index = dict(zip(tagset, range(len(tagset)))) # create mapping dict	
	
	mTags = []
	for tag in tags:
		mTag = [0]*len(tag_index)
		for t in tag:
			mTag[tag_index[t]] = 1
		mTags.append(mTag)
	mTags = np.array(mTags)
	return mTags, tag_index

def texts_mapping(texts, load=False, path=WORD_INDEX_PATH):
	print ('Tests mapping...')
	if load:
		word_index = load_dict(path) # load mapping dict
		tokenizer = text.Tokenizer()
		tokenizer.word_index = word_index
	else:
		tokenizer = text.Tokenizer()
		tokenizer.fit_on_texts(texts)
		word_index = tokenizer.word_index # create mapping dict
	sequences = tokenizer.texts_to_sequences(texts)
	return sequences, word_index

def texts_padding(texts, maxlen=MAX_SEQUENCE_LENGTH):
	print('Texts padding...','maxlen:',maxlen)
	return sequence.pad_sequences(texts, maxlen=maxlen)

def f1_score(y_true, y_pred):
	thresh = 0.4
	y_pred = K.cast(K.greater(y_pred,thresh),dtype='float32')
	tp = K.sum(y_true * y_pred)

	precision=tp/(K.sum(y_pred))
	recall=tp/(K.sum(y_true))
	return 2*((precision*recall)/(precision+recall))

def gloveEmbedding(word_index, embedding_dim=100, vector_path=os.sep.join(['data','glove.6B','glove.6B.{}d.txt'])):
	print('Load gloveEmbedding...','embedding_dim:',embedding_dim)
	# load all vectors from glove
	vector_path = vector_path.format(embedding_dim)
	word_vector_map = dict([(line.split()[0], np.array(line.split()[1:]).astype('float32')) for line in open(vector_path, encoding='utf8')])
	
	# create embadding matrix
	embedding_matrix = np.zeros((len(word_index)+1, embedding_dim))
	for word, idx in word_index.items():
		if word in word_vector_map:
			embedding_matrix[idx] = word_vector_map[word]

	# create embadding layer
	embeddingLayer = Embedding(
		len(word_index)+1, 
		embedding_dim, 
		weights=[embedding_matrix],
		input_length=MAX_SEQUENCE_LENGTH,
		trainable=False)

	return embeddingLayer

def build_RNN_model(word_index):
	print('Build RNN model...')
	model = Sequential()
	embeddingLayer = gloveEmbedding(word_index)
	model.add(embeddingLayer)
	model.add(GRU(128,activation='tanh',dropout=0.3))
	model.add(Dense(256,activation='relu'))
	model.add(Dropout(0.3))
	model.add(Dense(128,activation='relu'))
	model.add(Dropout(0.3))
	model.add(Dense(64,activation='relu'))
	model.add(Dropout(0.3))
	model.add(Dense(38,activation='sigmoid'))
	#adam = Adam(lr=0.001,decay=1e-6,clipvalue=0.5)
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=[f1_score])
	model.summary()
	return model


def _save_model(path='./model/049580/rnn.h5'):
	model.save(path)

def _load_model(path='./model/049580/rnn.h5'):
	model = load_model(path, custom_objects={'f1_score':f1_score})
	return model

def output():
	# load test data
	ids, texts = load_test_data()

	# pre-processing
	texts, word_index = texts_mapping(texts, True, WORD_INDEX_PATH)
	texts =	texts_padding(texts)

	# load model
	model = _load_model()

	# predict
	preds = model.predict(texts)
	preds[preds < 0.4]=0
	preds[preds >= 0.4]=1

	# load tags dictionary
	tag_index = load_dict(TAG_INDEX_PATH)
	inverse_tag_index = dict(zip(tag_index.values(),tag_index))

	# output
	out = open(predict_file_path, 'w')
	out.write('"id","tags"\n')
	for _id, _pred in zip(ids, preds):
		line = '"{}","{}"\n'
		labels = ""
		for idx, val in enumerate(_pred):
			if val == 1:
				labels += inverse_tag_index[idx] + " "
		line = line.format(_id, labels.strip())
		out.write(line)
	out.close()

'''
# load train data
ids, tags, texts = load_train_data()

# pre-processing
tags, tag_index = tags_mapping(tags)
texts, word_index = texts_mapping(texts)
texts =	texts_padding(texts)

# save dictionary
save_dict(tag_index, TAG_INDEX_PATH)
save_dict(word_index, WORD_INDEX_PATH)

# validation
valid_size = 500
train_X, train_Y = texts[valid_size:][:], tags[valid_size:][:]
valid_X, valid_Y = texts[:valid_size], tags[:valid_size]

# train
model = build_RNN_model(word_index)
model.fit(train_X, train_Y,batch_size=128, epochs=70, validation_data=(valid_X,valid_Y))

# save
_save_model()
'''
# output
output()