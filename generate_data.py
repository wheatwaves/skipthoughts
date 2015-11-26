import os

import theano
import theano.tensor as tensor

import cPickle as pkl
import numpy
import copy
import nltk

from collections import OrderedDict, defaultdict
from scipy.linalg import norm
from nltk.tokenize import word_tokenize
from gensim.models import Word2Vec as word2vec

profile = False

#-----------------------------------------------------------------------------#
# Specify model and table locations here
#-----------------------------------------------------------------------------#
path_to_models = '/Users/wheatwaves/deeplearning/skip-thoughts/data/'
path_to_tables = '/Users/wheatwaves/deeplearning/skip-thoughts/data/'
#-----------------------------------------------------------------------------#

path_to_umodel = path_to_models + 'uni_skip.npz'
path_to_bmodel = path_to_models + 'bi_skip.npz'
path_to_word2vec = '/Users/wheatwaves/deeplearning/skip-thoughts/data/GoogleNews-vectors-negative300.bin'
f = open(path_to_tables+'book_dictionary_large.pkl')
voc = pkl.load(f)
f.close()
btable = numpy.load(path_to_tables + 'btable.npy')
words = []
f = open(path_to_tables + 'dictionary.txt', 'rb')
for line in f:
	words.append(line.decode('utf-8').strip())
f.close()
btable = OrderedDict(zip(words, btable))
embed_map = word2vec.load_word2vec_format(path_to_word2vec, binary=True)
b_X=[]
b_Y=[]
for word in voc.keys()[:20000]:
	try:
		x = embed_map[word]
		y = btable[word]
		b_X.append(x)
		b_Y.append(y)
	except:
		continue
f = open(path_to_tables + 'b_X','w')
pkl.dump(b_X,f)
f.close()
f = open(path_to_tables + 'b_Y','w')
pkl.dump(b_Y,f)
f.close()
