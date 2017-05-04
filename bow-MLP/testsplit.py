import pandas as pd
import numpy as np
from sklearn import metrics
import re
from sklearn.feature_extraction.text import CountVectorizer
from bs4 import BeautifulSoup
import nltk
import cPickle as pickle
import itertools
from sklearn.metrics import confusion_matrix

prob = pickle.load( open( "prob.pickle", "rb" ) )
split = np.hsplit(prob,2)
print prob
scores = split[1]
print type(scores)