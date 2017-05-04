from keras.models import Sequential
from keras.layers import Dense, Dropout, BatchNormalization, Conv1D
import keras.utils
from keras.initializers import RandomUniform
from keras.optimizers import Adam
from keras.layers.advanced_activations import LeakyReLU
from keras.layers import Input, Dense, Dropout, Activation, BatchNormalization, Conv1D
from keras.initializers import RandomUniform
from keras.optimizers import Adam
from keras.layers.advanced_activations import LeakyReLU
from keras.models import Model
from keras import backend as K
from keras.utils import np_utils
from keras import metrics
import numpy as np
import cPickle as pickle
from scipy import sparse
from keras.optimizers import SGD

print "Start loading data..."
train1 = pickle.load( open( "furtherDetection.pickle", "rb" ) )
#train1 = sparse.csr_matrix.toarray(train1)
print type(train1)
#train2 = pickle.load( open( "TrainTextSparse2.pickle", "rb" ) )
#train2 = sparse.csr_matrix.toarray(train2)
#print type(train2)
labels = pickle.load( open( "relatedLabels.pickle", "rb" ) )
test = pickle.load( open( "furtherDetectionTest.pickle", "rb" ) )
#test = sparse.csr_matrix.toarray(test)
print type(test)
print "Start processing data..."

print "Start converting training data..."

#train = np.vstack((train1, train2))
#print type(train)
#print train.shape
X = train1
print X.shape
y = np_utils.to_categorical(labels)

print "Start converting testing data..."
test = np.asarray(test)
print type(test)

vocab_size = 7500
print "Start training MLP classifier..."
MLP_input = Input(shape=[vocab_size])
layer = Dense(vocab_size, kernel_initializer='glorot_normal')(MLP_input)
layer = BatchNormalization()(layer)
layer = LeakyReLU(0.2)(layer)
#layer = Dropout(0.5)(layer)
layer = Dense(vocab_size//16, kernel_initializer='glorot_normal')(layer)
layer = BatchNormalization()(layer)
layer = LeakyReLU(0.2)(layer)
layer = Dense(vocab_size//16, kernel_initializer='glorot_normal')(layer)
layer = BatchNormalization()(layer)
layer = LeakyReLU(0.2)(layer)
layer = Dense(vocab_size//16, kernel_initializer='glorot_normal')(layer)
layer = BatchNormalization()(layer)
layer = LeakyReLU(0.2)(layer)
#layer = Dense(vocab_size//16, kernel_initializer='glorot_normal')(layer)
#layer = BatchNormalization()(layer)
#layer = LeakyReLU(0.2)(layer)
#layer = Dropout(0.5)(layer)
layer = Dense(vocab_size//32, kernel_initializer='glorot_normal')(layer)
layer = BatchNormalization()(layer)
layer = LeakyReLU(0.2)(layer)
#layer = Dropout(0.5)(layer)
layer = Dense(vocab_size//64, kernel_initializer='glorot_normal')(layer)
layer = BatchNormalization()(layer)
layer = LeakyReLU(0.2)(layer)
layer = Dense(vocab_size//128, kernel_initializer='glorot_normal')(layer)
layer = BatchNormalization()(layer)
layer = Dense(3, kernel_initializer='glorot_normal')(layer)
out = Activation('sigmoid')(layer)

opt = Adam(lr=1e-4)
MLP = Model(MLP_input, out)
MLP.compile(loss='categorical_crossentropy', optimizer=opt, metrics=[metrics.categorical_accuracy])
print MLP.summary()




MLP.fit(X, y,
          epochs=20,
         batch_size=40,verbose=1)

print "Start making predictions..."
predictions = MLP.predict(test, batch_size=32, verbose=1)
for line in predictions:
    print line


#predictions = model.predict_classes(test,  batch_size=20)

with open('furtherPredictions.pickle', 'wb') as handle:
    pickle.dump(predictions, handle, protocol=pickle.HIGHEST_PROTOCOL)

#for x in predictions:
 #   print x
