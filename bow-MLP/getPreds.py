# -*- coding: utf-8 -*-
"""
Created on Mon Apr 17 20:07:13 2017

@author: xwang699
"""

import cPickle as pickle
predictions = pickle.load( open( "Predictions.pickle", "rb" ) )
print type(predictions)
print predictions.shape
#for line in predictions:
    #print line