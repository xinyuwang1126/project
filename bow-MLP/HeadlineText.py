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



num_articles = train['articleBody'].size

clean_train_articles = []
clean_train_headlines = []
print "Cleaning and parsing the articles...\n"


for i in xrange(0, num_articles):
    b = BeautifulSoup(train["Headline"][i],"html5lib").get_text()
    b = re.sub("[^a-zA-Z0-9]", " ", b)
 #   b = b.lower()
  #  a = BeautifulSoup(train["articleBody"][i],"html5lib").get_text()
  #  a = re.sub("[^a-zA-Z0-9]", " ", a)
  #  a = a.lower()
    #text = b+a
    clean_train_headlines.append(b)
   # clean_train_articles.append(a)
    if ((i + 1) % 1000 == 0):
        print "Review %d of %d\n" % (i + 1, num_articles)

#print len(clean_train_articles)
print len(clean_train_headlines)
#print test_data_features

with open('headlinetext.pickle', 'wb') as handle:
    pickle.dump(clean_train_headlines, handle, protocol=pickle.HIGHEST_PROTOCOL)

#with open('articletext.pickle', 'wb') as handle:
 #   pickle.dump(clean_train_articles, handle, protocol=pickle.HIGHEST_PROTOCOL)