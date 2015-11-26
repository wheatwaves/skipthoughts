"""
A multi-layer-perceptron transformation from word2vec to embedding space
"""
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
#-----------------------------------------------------------------------------#
# Specify model and dictionary locations here
#-----------------------------------------------------------------------------#
path_to_word2vec = '/Users/wheatwaves/deeplearning/skip-thoughts/data/GoogleNews-vectors-negative300.bin'
path_to_models = '/Users/wheatwaves/deeplearning/skip-thoughts/data/'
path_to_tables = '/Users/wheatwaves/deeplearning/skip-thoughts/data/'
path_to_X = path_to_models + 'b_X'
path_to_Y = path_to_models + 'b_Y'
# input_dim = 300
# output_dim = 620
#-----------------------------------------------------------------------------#

def main():
	f = open(path_to_X)
	b_X = pkl.load(f)
	f.close()
	f = open(path_to_Y)
	b_Y = pkl.load(f)
	f.close()
	X_train = []
	Y_train = []
	model = Sequential()
	model.add(Dense(620, input_dim=300, init='uniform'))
	model.add(Activation('tanh'))
	model.add(Dense(620, init='uniform'))
	model.add(Activation('softmax'))
	sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
	model.compile(loss='mean_squared_error', optimizer=sgd)
	X_train = np.array(b_X)
	Y_train = np.array(b_Y)
	model.fit(X_train, Y_train, nb_epoch=5, batch_size=32)
	model.save_weights(path_to_tables+'mlp_weight',overwrite = True)
if __name__=='__main__':
	main()