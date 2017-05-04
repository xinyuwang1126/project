import numpy as np
import pandas as pd
import time
import datetime

from utils.dataset import DataSet
from utils.generate_test_splits import kfold_split, get_stances_for_folds
from utils.score import report_score, LABELS, score_submission
from MyVocabProcessor import MyVocabProcessor

import tensorflow as tf
from tensorflow.contrib import learn
import nltk

def loadGloVe(filename):
    vocab = []
    embd = []
    file = open(filename, 'r', encoding="utf8")
    for line in file.readlines():
        row = line.strip().split(' ')
        vocab.append(row[0])
        embd.append(row[1:])
    print("Loaded GloVe.", len(vocab), " words loaded!")
    file.close()
    return vocab,embd

def stance2num(stance):
    output = []
    for s in stance:
        output.append(LABELS.index(s))
    return output

def num2stance(input):
    output = []
    for s in input:
        output.append(LABELS[s])
    return output

def bidirect_concat(feature, target):
    # in this model, headline and body texts are first concatenated together and then encoded by common RNN cells
    # the final states of this RNN encoder are concatenated as feature for classfier.

    text_id = feature['text_id']
    target = target['stance']

    # Convert indexes of words into embeddings.
    # This creates embeddings matrix of [n_words, EMBEDDING_SIZE] and then
    # maps word indexes of the sequence into [n_of_training_samples, length_of_headLine_or_body,
    # EMBEDDING_SIZE].
    with tf.name_scope('W'):
        W = tf.get_variable('W', [400001, 50])
        text_tensor = tf.nn.embedding_lookup(W, text_id)

    # # Split into list of embedding per word
    # # text_list results to be a list of tensors [n_of_training_samples, EMBEDDING_SIZE].
    text_list = tf.unstack(text_tensor, axis=1)

    # Create forward and backward direction GRU cell
    fw_cell = tf.contrib.rnn.GRUCell(200)
    bw_cell = tf.contrib.rnn.GRUCell(200)

    # Create an unrolled Bidirectional RNN with length MAX_DOCUMENT_LENGTH and passes text_list as inputs
    _, fw_encoding, bw_encoding = tf.contrib.rnn.static_bidirectional_rnn(fw_cell, bw_cell, text_list, dtype=tf.float32)

    # Concatenate fw_encoding and bw_enconding as features for classfier
    encoding = tf.concat([fw_encoding, bw_encoding], axis=1)

    # Given encoding of RNN, take encoding of last step (e.g hidden size of the
    # neural network of last step) and pass it as features for a 1-hidden layer NN classifier with 100 units
    target = tf.one_hot(target, 4, 1, 0)
    # logits = tf.contrib.layers.fully_connected(encoding, 4, activation_fn=None)
    logits = tf.contrib.layers.stack(encoding, tf.contrib.layers.fully_connected, [100, 4])
    loss = tf.contrib.losses.softmax_cross_entropy(logits, target)

    # # Create a training op.
    train_op = tf.contrib.layers.optimize_loss(
        loss,
        tf.contrib.framework.get_global_step(),
        optimizer='Adam',
        learning_rate=0.001)

    return ({   'class': tf.argmax(logits, 1),
                'prob': tf.nn.softmax(logits)
            }, loss, train_op)


if __name__ == '__main__':

    start_time = time.time()

    print('loading data')
    d = DataSet()
    folds, hold_out = kfold_split(d, training=0.9, n_folds=1)
    fold_stances, hold_out_stances = get_stances_for_folds(d, folds, hold_out)
    body = pd.read_csv('fnc-1/train_bodies.csv', index_col=0)
    train = pd.DataFrame(fold_stances[0]).join(body, on='Body ID')
    test = pd.DataFrame(hold_out_stances).join(body, on='Body ID')
    print('data is loaded')

    print('loading pre-trained embedding')
    vocab, embd = loadGloVe('glove.6B.50d.txt')
    vocab_size = len(vocab)
    embedding_dim = len(embd[0])
    embedding = np.asarray(embd)

    with tf.name_scope('W'):
        W = tf.Variable(tf.constant(0.0, shape=[vocab_size, embedding_dim]),
                        trainable=False, name="W")
        embedding_placeholder = tf.placeholder(tf.float32, [vocab_size, embedding_dim])
        embedding_init = W.assign(embedding_placeholder)

    sess = tf.Session()
    sess.run(embedding_init, feed_dict={embedding_placeholder: embedding})

    # init vocab processor
    max_headLine_length, max_body_length = 50, 200
    vocab_processor = MyVocabProcessor(vocab)

    headLine_id = vocab_processor.transform(train['Headline'].str.lower(), max_headLine_length)
    body_id = vocab_processor.transform(train['articleBody'].str.lower(), max_body_length)

    print('get text id for training set')
    feature = {'text_id': np.concatenate((headLine_id, body_id), axis=1)}
    target = {'stance': pd.Series(stance2num(train['Stance']))}
    print('Shape of training feature is ', feature['text_id'].shape)
    print('Shape of training target is ', target['stance'].shape)

    print('building rnn model')
    classifier = learn.Estimator(model_fn=bidirect_concat)

    print('training rnn model')
    n_steps = 15000       # total training iterations
    b_size = 32        # batch size for SGD
    print('n_steps: %s, b_size: %s ' % (n_steps, b_size))
    time_train = time.time()
    tf.logging.set_verbosity(tf.logging.INFO)
    monitor = tf.contrib.learn.monitors.ValidationMonitor(feature, target, every_n_steps=1)
    classifier.partial_fit(feature, target, steps=n_steps, batch_size=b_size, monitors=[monitor])
    print('--- %s seconds for training ---' % (time.time() - time_train))

    print('predict')
    headLine_id_test = vocab_processor.transform(test['Headline'].str.lower(), max_headLine_length)
    body_id_test = vocab_processor.transform(test['articleBody'].str.lower(), max_body_length)
    feature_test = {'text_id': np.concatenate((headLine_id_test, body_id_test), axis=1)}
    target_test = {'stance': pd.Series(stance2num(test['Stance']))}

    y_pred = [ p['class'] for p in classifier.predict(feature_test, as_iterable=True) ]
    stance_pred = num2stance(y_pred)
    test['stance_pred'] = stance_pred
    test.to_csv('test_bidirec_concat_stance_pred_%s_batch_size_%s_nSteps_%s.csv' \
                                  % (datetime.datetime.now().strftime("%Y-%m-%d"), b_size, n_steps))
    train.to_csv('train_bidirec_concat_stance_pred_%s_batch_size_%s_nSteps_%s.csv' \
                                  % (datetime.datetime.now().strftime("%Y-%m-%d"), b_size, n_steps))

    report_score(list(test['Stance']), stance_pred)

    print('--- %s seconds in total ---' % (time.time() - start_time))
