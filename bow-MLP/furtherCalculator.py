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


test = pd.read_csv('BoWMLPBinary.csv',header=0)
print test.columns.values
#print test['articleBody'][34]
print "\n\n"
count=0
num_lables = test['Stance'].size
for i in xrange(0, num_lables):
    if test["BoWMLP"][i]!=1:
        count=count+1
labels = []
stances = []
print "Parsing the labels...\n"

for i in xrange(0, num_lables):
    if test["BoWMLP"][i] != 0:
        stance = test["Stance"][i]
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
        else:
            label = 99
            labels.append(label)
            stances.append(stance)
    if ((i + 1) % 1000 == 0):
        print "Review %d of %d\n" % (i + 1, num_lables)

print "number of stances",len(stances)

predictions = pickle.load( open( "predictedStances.pickle", "rb" ) )
#print type(np.int32(predictions[0]).item())


agree=0
discuss=0
disagree=0
a=0
b=0
c=0
AGREE=0
DISCUSS=0
DISAGREE=0
for i in xrange(0,len(predictions)):
    if stances[i] == predictions[i] and labels[i] == 0:
        agree=agree+1
    elif stances[i] != predictions[i] and labels[i] == 0:
        a = a+1
       # if predictions[i] == "disagree":
        #    DISAGREE=DISAGREE+1
        #elif predictions[i] == "discuss":
         #   DISCUSS = DISCUSS+1
    elif stances[i] == predictions[i] and labels[i] == 1:
        discuss=discuss+1
    elif stances[i] != predictions[i] and labels[i] == 1:
        b = b+1
        if predictions[i] == "disagree":
            DISAGREE=DISAGREE+1
        elif predictions[i] == "agree":
            AGREE=AGREE+1
    elif stances[i] == predictions[i] and labels[i] == 2:
        disagree=disagree+1
    elif stances[i] != predictions[i] and labels[i] == 2:
        c = c+1
       # if predictions[i] == "discuss":
          #  DISCUSS=DISCUSS+1
        #elif predictions[i] == "agree":
         #   AGREE=AGREE+1

print "Agree instances classified correctly:",agree
print "Agree instances classified incorrectly:",a
print "Discuss instances classified correctly:",discuss
print "Discuss instances classified incorrectly:",b
print "Disagree instances classified correctly:",disagree
print "Disagree instances classified incorrectly:",c
print agree+a+disagree+c+discuss+b
print len(predictions)
print count
print AGREE,DISAGREE,DISCUSS