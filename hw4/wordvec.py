import os, sys
import word2vec
from sklearn import decomposition
import nltk
import numpy as np
from adjustText import adjust_text
import matplotlib.pyplot as plt

# info
corpus_folder = './data/Book5TheOrderOfThePhoenix'
corpus_save_path = './experiment/2_word2vec_train.txt'
corpus_phrase_save_path = './experiment/2_word2vec_train_phrase.txt'
word2vec_model_path = corpus_save_path.replace('.txt','.bin')
word2vec_word_dim = 500
k_most_frequent = 500
word2vec_vis_save_path = './experiment/2_word2vec_vis_{}.png'.format(word2vec_word_dim)



# load all corpus
corpus_paths = os.listdir(corpus_folder)
corpus_paths = [corpus_folder + os.sep + fname for fname in corpus_paths if not fname.startswith('.')]
corpus = [open(path,'r').read() for path in corpus_paths]


# save corpus to single file
corpus_file = open(corpus_save_path,'w')
for line in corpus:
	corpus_file.write(line)
	corpus_file.write('\n')
corpus_file.close()


# train word2vec model
word2vec.word2phrase(corpus_save_path, corpus_phrase_save_path, verbose=True)
word2vec.word2vec(corpus_phrase_save_path, word2vec_model_path, size=word2vec_word_dim, verbose=True)
model = word2vec.load(word2vec_model_path)

# sklearn pca to 2d vector 
word_vectors = model.vectors
pca = decomposition.PCA(n_components=2)
pca.fit(word_vectors)
word_vectors_2d = pca.transform(word_vectors)

# filter word from top k freq
tok_and_pos = nltk.pos_tag(model.vocab[:k_most_frequent])
vectors = word_vectors_2d[:k_most_frequent]
plt_data = []
for word_idx in xrange(len(tok_and_pos)):
	tok, pos_tag = tok_and_pos[word_idx]
	if any(punc in tok for punc in [',','.',':',';',u'\u2019','!','?',u'\u201c']): continue
	if len(tok) <= 1:	continue
	if pos_tag in ['JJ','NNP','NN','NNS']:
		plt_data.append((tok,vectors[word_idx]))
		#print tok,'\t',pos_tag
plt_words, plt_vectors= zip(*plt_data)
X,Y = zip(*plt_vectors)

# plot 2d vector
plt.figure(figsize=(12,8))
plt.scatter(X, Y, marker = 'o',s=3,c='r')
texts = []
for word, x, y in zip(plt_words, X, Y):
	texts.append(plt.text(x, y, word, size=7))
adjust_text(texts, arrowprops=dict(arrowstyle="-", color='k', lw=1.0))
plt.title('Word2vec Visualization (dim={})'.format(word2vec_word_dim))
plt.savefig(word2vec_vis_save_path)



