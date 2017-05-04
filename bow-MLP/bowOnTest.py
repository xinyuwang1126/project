import pandas as pd
import numpy as np
import re
from sklearn.feature_extraction.text import CountVectorizer
from bs4 import BeautifulSoup
import nltk
import cPickle as pickle

vectorizer = CountVectorizer(max_features=5000)
headlineVectorizer = CountVectorizer(max_features=2500)
train = pd.read_csv('BoWMLPBinary.csv',header=0)
print train.columns.values
#print test['articleBody'][34]
print "\n\n"


def article_to_words(raw_article):
    article_text = BeautifulSoup(raw_article,"html5lib").get_text()
    letters_only = re.sub("[^a-zA-Z0-9]", " ", article_text)
    words = letters_only.lower()
    return(words)

#clean_article = article_to_words(test['articleBody'][34])
#print clean_article
num_articles = train['articleBody'].size

clean_train_articles = []
clean_train_headlines = []
print "Cleaning and parsing the articles...\n"
a= article_to_words(train["articleBody"][0])
b= article_to_words(train["Headline"][0])
print type(b+a)
for i in xrange(0, num_articles):
    if train["BoWMLP"][i]!=0:
        b = article_to_words(train["Headline"][i])
        a = article_to_words(train["articleBody"][i])
    #text = b+a
        clean_train_headlines.append(b)
        clean_train_articles.append(a)
    if ((i + 1) % 1000 == 0):
        print "Review %d of %d\n" % (i + 1, num_articles)

train_data_features = vectorizer.fit_transform(clean_train_articles)
print vectorizer.get_feature_names()
train_data_features = train_data_features.toarray()
train_headlines = headlineVectorizer.fit_transform(clean_train_headlines)
train_headlines = train_headlines.toarray()
print train_data_features.shape
print train_headlines.shape
headline_article = np.hstack((train_headlines,train_data_features))
print headline_article.shape
#print test_data_features

with open('furtherDetectionTest.pickle', 'wb') as handle:
    pickle.dump(headline_article, handle, protocol=pickle.HIGHEST_PROTOCOL)



