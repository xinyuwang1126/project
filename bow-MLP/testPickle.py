import cPickle as pickle
import numpy as np
from scipy import sparse
#test1 = pickle.load( open( "headline_article1.pickle", "rb" ) )
test2 = pickle.load( open( "twoPredictions.pickle", "rb" ) )
#test = np.vstack((test1, test2))
for line in test2:
    if line!=0:
        print line
print type(test2)
print len(test2)
#print type(test2)
#print test2.shape
#sTest = sparse.csr_matrix(test2)
#print type(sTest)
#print sTest

#with open('TrainTextSparse2.pickle', 'wb') as handle:
 #   pickle.dump(sTest, handle, protocol=pickle.HIGHEST_PROTOCOL)
#print type(test1)
#print test.shape

