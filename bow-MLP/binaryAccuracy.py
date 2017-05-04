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


test = pd.read_csv('test.csv',header=0)
print test.columns.values
#print test['articleBody'][34]
print "\n\n"

num_lables = test['Stance'].size

labels = []
stances = []
print "Parsing the labels...\n"

for i in xrange(0, num_lables):
    stance = test["Stance"][i]
    stances.append(stance)
    if stance == "unrelated":
        label = 0
        labels.append(label)
    elif stance == "agree":
        label = 1
        labels.append(label)
    elif stance == "discuss":
        label = 2
        labels.append(label)
    else:
        label = 3
        labels.append(label)
    if ((i + 1) % 1000 == 0):
        print "Review %d of %d\n" % (i + 1, num_lables)

predictions = pickle.load( open( "twoPredictions.pickle", "rb" ) )
#print type(np.int32(predictions[0]).item())

for i in xrange(0,9987):
    np.int32(predictions[i]).item()

unrelated = 0
related = 0
a=0
b=0

for i in xrange(0,9987):
    if labels[i] == predictions[i] and labels[i] == 0:
        unrelated = unrelated+1
    elif labels[i] != predictions[i] and labels[i] == 0:
        a = a+1
    elif labels[i] == predictions[i] and labels[i] == 1:
        related = related+1
    elif labels[i] != predictions[i] and labels[i] == 1:
        b = b+1

print "Unrelated instances classified correctly:",unrelated
print "Unrelated instances classified incorrectly:",a
print "Related instances classified correctly:",related
print "Related instances classified incorrectly:",b
