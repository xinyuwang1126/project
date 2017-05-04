import pandas as pd
import numpy as np
import re
from sklearn.feature_extraction.text import CountVectorizer
from bs4 import BeautifulSoup
import nltk
import cPickle as pickle


train = pd.read_csv('train.csv',header=0)
print train.columns.values
#print test['articleBody'][34]
print "\n\n"




#clean_article = article_to_words(test['articleBody'][34])
#print clean_article
#num_articles = train['articleBody'].size
num_lables = train['Stance'].size

labels = []
stances = []
print "Parsing the labels...\n"

for i in xrange(0, num_lables):
    stance = train["Stance"][i]
#    stances.append(stance)
#    if stance == "unrelated":
 #       label = 0
  #      labels.append(label)
    if stance == "agree":
        label = 0
        labels.append(label)
        stances.append(stance)
    elif stance == "discuss":
        label = 1
        labels.append(label)
        stances.append(stance)
    elif stance == "disagree":
        label = 2
        labels.append(label)
        stances.append(stance)
    if ((i + 1) % 1000 == 0):
        print "Review %d of %d\n" % (i + 1, num_lables)

print type(labels)
print len(labels)
print len(stances)
print zip(stances,labels)

with open('relatedLabels.pickle', 'wb') as handle:
    pickle.dump(labels, handle, protocol=pickle.HIGHEST_PROTOCOL)


