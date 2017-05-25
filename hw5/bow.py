import sys, os
import numpy as np

from keras.preprocessing import text, sequence
from keras.models import Sequential
from keras.layers import Embedding, Dense, Dropout
from keras.optimizers import Adam
from keras import backend as K
from keras.callbacks import EarlyStopping, ModelCheckpoint


train_data_path = 'data' + os.sep + 'train_data.csv'
test_data_path = sys.argv[1] if len(sys.argv) > 1 else 'data' + os.sep + 'test_data.csv'
prediction_file = sys.argv[2] if len(sys.argv) > 2 else 'predict.csv'

TAG_INDEX_PATH = 'tag_index.npy'
WORD_INDEX_PATH = 'word_index.npy'

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

def texts_mapping(texts, map_path='word_index.npy'):
	print ('Tests mapping...')
	tokenizer = text.Tokenizer()
	tokenizer.fit_on_texts(texts)
	word_index = tokenizer.word_index # mapping dict
	return tokenizer.texts_to_sequences(texts), word_index

def texts_to_BOW_vectors(texts, word_index):
	print('Texts to BOW vector...')
	BOW_vectors = np.zeros((len(texts), (len(word_index)+1))).astype('float32')
	for idx, word_idxs in enumerate(texts):
		for word_idx in word_idxs:
			if word_idx <= len(word_index):
				BOW_vectors[idx][word_idx] += 1.0
	return BOW_vectors

def texts_TFIDF(texts):
	print('Texts TFIDF...')
	IDF = np.zeros(texts.shape)
	IDF[texts!=0] = 1.0
	IDF = IDF.sum(axis=0)
	IDF[IDF==0] = len(texts)
	IDF = np.log(len(texts)/IDF)
	TFIDF = texts*IDF
	return TFIDF

def f1_score(y_true,y_pred):
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

def build_BOW_model(word_index):
	print('Build BOW model...')
	model = Sequential()
	model.add(Dense(1024, activation='relu', input_dim=len(word_index)+1))
	model.add(Dropout(0.3))
	model.add(Dense(512, activation='relu'))
	model.add(Dropout(0.3))
	model.add(Dense(128, activation='relu'))
	model.add(Dropout(0.3))
	model.add(Dense(38, activation='sigmoid'))
	model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=[f1_score])
	model.summary()
	return model
def _save_model(path):
	model.save(path)

def _load_model(path):
	model = load_model(path, custom_objects={'f1_score':f1_score})
	return model
	
def output():
	# load test data
	ids, texts = load_test_data()

	# pre-processing
	texts, word_index = texts_mapping(texts, load=True, path=WORD_INDEX_PATH)
	texts =	texts_padding(texts)

	# load model
	model = _load_model('./model/bow_0.566.h5')

	# predict
	preds = model.predict(texts)

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
			if val > 0.4:
				labels += inverse_tag_index[idx] + " "
		line = line.format(_id, labels.strip())
		out.write(line)
	out.close()

# load data
ids, tags, texts = load_train_data()

# pre-processing
tags, tag_index = tags_mapping(tags)
texts, word_index = texts_mapping(texts)
texts =	texts_to_BOW_vectors(texts, word_index)
texts = texts_TFIDF(texts)

# validation
indices = np.arange(texts.shape[0])  
np.random.shuffle(indices) 
X, Y = texts[indices], tags[indices]
valid_size = 500
train_X, train_Y = X[valid_size:][:], Y[valid_size:][:]
valid_X, valid_Y = X[:valid_size], Y[:valid_size]

# train
model = build_BOW_model(word_index)
earlystopping = EarlyStopping(monitor='val_f1_score', patience = 20, verbose=0, mode='max')
checkpoint = ModelCheckpoint(
				filepath='./model/rnn_{val_f1_score:.3f}.h5',
				verbose=0,
				save_best_only=True,
				save_weights_only=False,
				monitor='val_f1_score',
				mode='max'
			)
model.fit(train_X, train_Y,batch_size=128, epochs=50, validation_data=(valid_X,valid_Y))

	
