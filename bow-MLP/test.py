import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

vectorizer = CountVectorizer(min_df=1)

corpus = [
    'This is the first document.',
    'This is the second second document.',
    'And the third one.',
    'Is this the first document?',
 ]
X = vectorizer.fit_transform(corpus)
#print X
print vectorizer.get_feature_names()
Y=X.toarray()
print Y
print vectorizer.vocabulary_.get('document')
print vectorizer.vocabulary_.get('ad')
print type(Y)

print vectorizer.transform(['Something and completely new.']).toarray()

bigram_vectorizer = CountVectorizer(ngram_range=(1, 2), token_pattern=r'\b\w+\b', min_df=1)
analyze = bigram_vectorizer.build_analyzer()
#print analyze ('Bi-grams are cool!')

X_2 = bigram_vectorizer.fit_transform(corpus).toarray()
print bigram_vectorizer.get_feature_names()