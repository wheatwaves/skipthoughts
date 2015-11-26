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
def cos_similarity(X,Y):
	return numpy.dot(X,Y).flatten()/(norm(X)*norm(Y))