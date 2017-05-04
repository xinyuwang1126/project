from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
import numpy as np
import cPickle as pickle
from scipy import sparse

print "Start loading data..."
train1 = pickle.load( open( "TrainTextSparse1.pickle", "rb" ) )
train1 = sparse.csr_matrix.toarray(train1)
print type(train1)
train2 = pickle.load( open( "TrainTextSparse2.pickle", "rb" ) )
train2 = sparse.csr_matrix.toarray(train2)
print type(train2)
labels = pickle.load( open( "labels.pickle", "rb" ) )
test = pickle.load( open( "TestSparse.pickle", "rb" ) )
test = sparse.csr_matrix.toarray(test)
print type(test)
print "Start processing data..."

print "Start converting training data..."
scaler = StandardScaler()
scaler.fit(train1)
train1 = scaler.transform(train1)
scaler.fit(train2)
train2 = scaler.transform(train2)
train = np.vstack((train1, train2))
print type(train)
print train.shape
X = train
print X.shape
y = labels
print "Start converting testing data..."
scaler.fit(test)
test = scaler.transform(test)

clf = MLPClassifier(solver='sgd', alpha=1e-5,
                  hidden_layer_sizes=(5, 2), max_iter=100, batch_size='auto',random_state=1)

print "Start training MLP classifier..."
clf.fit(X, y)

print "Start making predictions..."
predictions = clf.predict(test)
with open('Predictions.pickle', 'wb') as handle:
    pickle.dump(predictions, handle, protocol=pickle.HIGHEST_PROTOCOL)

for x in predictions:
    print x
#print [coef.shape for coef in clf.coefs_]
#print clf.predict_proba([[2., 2.], [4.5, 4.8]])
#scaler = StandardScaler()
#scaler.fit(X)
#X_train = scaler.transform(X)
#print X_train
#X_test = scaler.transform([[0.,0.],[2., 2.], [-1., -2.],[4.5, 4.8]])
#print X_test
#clf.fit(X_train, y)
#print clf.predict(X_test)