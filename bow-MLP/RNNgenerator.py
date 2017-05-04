# Implemented after reading Andrej Karpathy's post about RNN
# References:
# https://www.youtube.com/watch?v=cO0a0QYmFm8&list=PLlJy-eBtNFt6EuMxFYRiNRS07MCWN5UIA&index=10
# http://karpathy.github.io/2015/05/21/rnn-effectiveness/
# http://cs231n.github.io/neural-networks-case-study/


import numpy as np
#import cPickle as pickle

def train(inputs, targets, hprev):
    # Initialization
    xhat = {} # Holds one-hot representation of x
    # Holds one-hot representation of predicted y(unnormalized)
    yhat = {}
    # Normalized probabilites of each output through time
    p = {}
    # Holds state vectors through time
    h = {}
    h[-1] = np.copy(hprev)
    # Forward pass
    loss = 0
    dWxh = np.zeros_like(Wxh)
    dWhh = np.zeros_like(Whh)
    dWhy = np.zeros_like(Why)
    dbh = np.zeros_like(bh)
    dby = np.zeros_like(by)

    # Here we iterate through the words we have taken. In this particular program,
    # there is only one iteration because we read in the words one by one.
    for t in xrange(len(inputs)):
        # 1-to-k representation
        xhat[t] = np.zeros((wordSize,1))
        xhat[t][inputs[t]] = 1
        # New hidden state
        h[t] = np.tanh(np.dot(Wxh, xhat[t]) + np.dot(Whh, h[t-1] + bh))
        yhat[t] = np.dot(Why, h[t]) + by
        p[t] = np.exp(yhat[t])/np.sum(np.exp(yhat[t]))
        # Cross-entropy loss
        loss += -np.log(p[t][targets[t],0])
    # Backword pass
    dhnext = np.zeros_like(h[0])
    for t in reversed(xrange(len(inputs))):
        # Backpropagation into y
        dy = np.copy(p[t])
        dy[targets[t]] -= 1
        # Get updates for y
        dWhy += np.dot(dy, h[t].T)
        dby += dy
        # Backprop into h
        dh = np.dot(Why.T, dy) + dhnext
        dhraw = (1-h[t]*h[t]) * dh
        # Get updates for h
        dWxh += np.dot(dhraw, xhat[t].T)
        dWhh += np.dot(dhraw, h[t-1].T)
        dbh += dhraw
        # Save dhnext for next iteration
        dhnext = np.dot(Whh.T, dhraw)
    # Clip to mitigate exploding gradients
    for dparam in [dWxh, dWhh, dWhy, dbh, dby]:
        np.clip(dparam, -5, 5, out=dparam)
    # Returns loss, gradients on model parameters and last hidden state
    return loss, dWxh, dWhh, dWhy, dbh, dby, h[len(inputs)-1]




def sample(hprev, seedIndex, n):
    indices = []
    # Encode words
    xhat = np.zeros((wordSize, 1))
    xhat[seedIndex] = 1
    # Sample n indices
    for t in range(n):
        hprev = np.tanh(np.dot(Wxh, xhat) + np.dot(Whh, hprev) + bh)
        y = np.dot(Why, hprev) +by
        p = np.exp(y)/np.sum(np.exp(y))
        index = np.random.choice(range(wordSize), p = p.ravel())
        indices.append(index)
        # Update
        xhat = np.zeros((wordSize,1))
        xhat[index] = 1
    return indices




# Read data
# Data preprocessing might be necessary for certain data input.
data = open('headlinetext.txt').read()
#data = pickle.load( open( "headlinetext.pickle", "rb" ) )
data = data.split()
words = list(set(data))
dataSize = len(data)
wordSize = len(words)
# Preparation for one-hot word encoding
wordToIndex = {word:i for i,word in enumerate(words)}
indexToWord = {i:word for i,word in enumerate(words)}

# Insize and outsize are the same as word size.
# Here, size of hidden layer of neurons here is 100. Learning rate is 0.1
learningRate = 0.00015
hiddenSize = 100
inSize = wordSize
outSize = wordSize
# When the input is big, truncation would be needed.
# Here because the data input is small, we don't actually truncate.
seqLength = 1
# Generate model parameters in the form of matrix.
# Input to hidden layer
Wxh = np.random.randn(hiddenSize,inSize)*0.01
# From hidden to hidden
Whh = np.random.randn(hiddenSize,hiddenSize)*0.01
# From hidden to output
Why = np.random.randn(outSize,hiddenSize)*0.01
# Hidden bias
bh = np.zeros((hiddenSize,1))
# Output bias
by = np.zeros((outSize,1))
# Initialize memory variables for adaptive gradient descent
mWxh = np.zeros((hiddenSize,inSize))
mWhh = np.zeros((hiddenSize,hiddenSize))
mWhy = np.zeros((outSize,hiddenSize))
mbh = np.zeros((hiddenSize,1))
mby = np.zeros((outSize,1))


# Main loop
# Initialize data pointer and iteration counter
p = 0; n = 0
# Initialize the loss
smoothLoss = -np.log(1.0/wordSize)*seqLength
while smoothLoss>0.5 or n<=100000:
    # Prepare the inputs
    if p + seqLength + 1 >= len(data) or n==0:
        # Reset/Initialize RNN memory
        # hprev stands for previous hidden state
        hprev = np.zeros((hiddenSize,1))
        # Start from the beginning
        p = 0
    # Get seqLength words
    inputs = [wordToIndex[word] for word in data[p:p+seqLength]]
    # Targets are the ones next to inputs
    targets = [wordToIndex[word] for word in data[p+1:p+seqLength+1]]
    # Sample from the model
    if n % 100 == 0:
        sampleIndex = sample(hprev,inputs[0],10)
        text = ' '.join(indexToWord[i] for i in sampleIndex)
        print text
    # Forward seqLength words to the model and fetch gradient
    loss, dWxh, dWhh, dWhy, dbh, dby, hprev = train(inputs, targets, hprev)
    smoothLoss = smoothLoss*0.999 + loss*0.001
    # Print progress
    if n % 100 == 0:
        print 'iter %d, loss: %f' % (n, smoothLoss)
    # Update parameters
    for param, dparam, mem in zip([Wxh, Whh, Why, bh, by],
                                [dWxh, dWhh, dWhy, dbh, dby],
                                [mWxh, mWhh, mWhy, mbh, mby]):
        mem += dparam * dparam
        param += -learningRate * dparam/np.sqrt(mem + 1e-8)
    # Move data pointer
    p += seqLength
    # Increase iteration counter
    n += 1
