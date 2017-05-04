import numpy as np
import cPickle as pickle

predictions = pickle.load( open( "furtherPredictions.pickle", "rb" ) )
print len(predictions)
print type(predictions)
classes = []
for line in predictions:
    if line[0]>=line[1] and line[0]>=line[2]:
        classes.append(0)
        print line[0],line
    elif line[1]>=line[0] and line[1]>=line[2]:
        classes.append(1)
        print line[1],line
    elif line[2]>=line[0] and line[2]>=line[1]:
        classes.append(2)
        print line[2],line

print type(classes),classes
labels = []
for x in classes:
    if x == 0:
        labels.append("agree")
    elif x == 1:
        labels.append("discuss")
    elif x == 2:
        labels.append("disagree")
print labels
with open('predictedStances.pickle', 'wb') as handle:
    pickle.dump(labels, handle, protocol=pickle.HIGHEST_PROTOCOL)

