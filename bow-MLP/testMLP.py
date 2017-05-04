from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
import numpy as np
import cPickle as pickle


train1 = pickle.load( open( "headline_article1.pickle", "rb" ) )
train2 = pickle.load( open( "headline_article2.pickle", "rb" ) )
labels = pickle.load( open( "labels.pickle", "rb" ) )
test = pickle.load( open( "testBOW.pickle", "rb" ) )
train = np.vstack((train1, train2))
X = train
y = labels

clf = MLPClassifier(solver='lbfgs', alpha=1e-5,
                  hidden_layer_sizes=(5, 2), random_state=1)

clf.fit(X, y)

MLPClassifier(activation='relu', alpha=1e-05, batch_size='auto',
       beta_1=0.9, beta_2=0.999, early_stopping=False,
       epsilon=1e-08, hidden_layer_sizes=(5, 2), learning_rate='constant',
       learning_rate_init=0.001, max_iter=200, momentum=0.9,
       nesterovs_momentum=True, power_t=0.5, random_state=1, shuffle=True,
       solver='lbfgs', tol=0.0001, validation_fraction=0.1, verbose=False,
       warm_start=False)

print clf.predict([[0.,0.],[2., 2.], [-1., -2.],[4.5, 4.8]])
#print [coef.shape for coef in clf.coefs_]
#print clf.predict_proba([[2., 2.], [4.5, 4.8]])
scaler = StandardScaler()
scaler.fit(X)
X_train = scaler.transform(X)
#print X_train
X_test = scaler.transform([[0.,0.],[2., 2.], [-1., -2.],[4.5, 4.8]])
#print X_test
clf.fit(X_train, y)
print clf.predict(X_test)