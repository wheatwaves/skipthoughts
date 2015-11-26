import os

import theano
import theano.tensor as tensor

import cPickle as pkl
import numpy as np
import copy
import nltk

from collections import OrderedDict, defaultdict
from scipy.linalg import norm
from nltk.tokenize import word_tokenize
from gensim.models import Word2Vec as word2vec
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import SGD

path_to_tables = '/Users/wheatwaves/deeplearning/skip-thoughts/data/'
path_to_word2vec = '/Users/wheatwaves/deeplearning/skip-thoughts/data/GoogleNews-vectors-negative300.bin'
def main():
	f = open(path_to_tables + 'dictionary.txt', 'rb')
	words = []
	for line in f:
		words.append(line.decode('utf-8').strip())
	embed_map = word2vec.load_word2vec_format(path_to_word2vec, binary=True)
	X=[]
	for word in words:
		try:
			X.append(embed_map[word])
		except:
			X.append(np.array([0]*300))
	X = np.array(X)
	model = Sequential()
	model.add(Dense(620, input_dim=300, init='uniform'))
	model.add(Activation('tanh'))
	model.add(Dense(620, init='uniform'))
	model.add(Activation('softmax'))
	model.load_weights(path_to_tables+'mlp_weight');
	sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
	model.compile(loss='mean_squared_error', optimizer=sgd)
	Y = np.array(model.predict(X))
	np.save(path_to_tables+'new_btable.npy',Y)
if __name__=='__main__':
	main()