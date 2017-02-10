#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  8 21:43:39 2017

@author: peterweber
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mxnet as mx
import xgboost as xgb
import time
import sklearn as sk

#%%
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

#%%
img_size = 28
val_size = 100
train_size = train.shape[0] - val_size
test_size = test.shape[0]

#%%
train_lbl, train_img = train.iloc[:,0].values, train.iloc[:,1:].values
train_img = train_img.reshape((train.shape[0], 28, 28)).astype(np.uint8)

#%%
def randomize(dataset, labels):
  permutation = np.random.permutation(labels.shape[0])
  shuffled_dataset = dataset[permutation,:,:]
  shuffled_labels = labels[permutation]
  return shuffled_dataset, shuffled_labels

train_img, train_lbl = randomize(train_img, train_lbl)

#%%
val_img = train_img[:val_size,:,:]
train_img = train_img[val_size:,:,:]

val_lbl = train_lbl[:val_size]
train_lbl = train_lbl[val_size:] 

test_img = test.values.reshape((test_size, img_size, img_size)).astype(np.uint8)
#%% Verify that the data looks fine
for i in range(10):
    plt.subplot(1,10,i+1)
    plt.imshow(train_img[i], cmap='Greys_r')
    plt.axis('off')
plt.show()
print('label: %s' % (train_lbl[0:10],))

#%% Implement first xgboost
nsamples, nx, ny = train_img.shape
dtrain_img = train_img.reshape((nsamples,nx*ny))

dtrain = xgb.DMatrix(data = dtrain_img, label = train_lbl)

nsamples, nx, ny = val_img.shape
dval_img = val_img.reshape((nsamples,nx*ny))
dval = xgb.DMatrix(data = dval_img)
#%%
# specify parameters via map
param = {'max_depth':2, 'eta':1, 'silent':1, 'objective':'multi:softprob', 'num_class':10}
num_round = 2
start = time.time()
bst = xgb.cv(param, dtrain, num_round, nfold = 3, verbose_eval = bool)
end = time.time()
end-start
# make prediction
#preds = bst.predict(dval)

#print("xgb accuracy on validation set", sk.metrics.accuracy_score(val_lbl, preds))


#%%
def to4d(img):
    return img.reshape(img.shape[0], 1, img_size, img_size).astype(np.float32)/255

batch_size = 100
train_iter = mx.io.NDArrayIter(to4d(train_img), train_lbl, batch_size, shuffle=True)
val_iter = mx.io.NDArrayIter(to4d(val_img), val_lbl, batch_size)

test_img = to4d(test_img)

#%% 
# Create a place holder variable for the input data
data = mx.sym.Variable('data')
# Flatten the data from 4-D shape (batch_size, num_channel, width, height) 
# into 2-D (batch_size, num_channel*width*height)
data = mx.sym.Flatten(data=data)

# The first fully-connected layer
fc1  = mx.sym.FullyConnected(data=data, name='fc1', num_hidden=128)
# Apply relu to the output of the first fully-connnected layer
act1 = mx.sym.Activation(data=fc1, name='relu1', act_type="relu")

# The second fully-connected layer and the according activation function
fc2  = mx.sym.FullyConnected(data=act1, name='fc2', num_hidden = 64)
act2 = mx.sym.Activation(data=fc2, name='relu2', act_type="relu")

# The thrid fully-connected layer, note that the hidden size should be 10, which is the number of unique digits
fc3  = mx.sym.FullyConnected(data=act2, name='fc3', num_hidden=10)
# The softmax and loss layer
mlp  = mx.sym.SoftmaxOutput(data=fc3, name='softmax')

# We visualize the network structure with output size (the batch_size is ignored.)
shape = {"data" : (batch_size, 1, 28, 28)}
mx.viz.plot_network(symbol=mlp, shape=shape)


#%%
# @@@ AUTOTEST_OUTPUT_IGNORED_CELL
import logging
logging.getLogger().setLevel(logging.DEBUG)

model = mx.model.FeedForward(
    symbol = mlp,       # network structure
    num_epoch = 10,     # number of data passes for training 
    learning_rate = 0.1 # learning rate of SGD 
)
model.fit(
    X=train_iter,       # training data
    eval_data=val_iter, # validation data
    batch_end_callback = mx.callback.Speedometer(batch_size, 200) # output progress for each 200 data batches
)

#%%Prediction of a sinle image
# @@@ AUTOTEST_OUTPUT_IGNORED_CELL
plt.imshow(val_img[0], cmap='Greys_r')
plt.axis('off')
plt.show()
prob = model.predict(val_img[0:1].astype(np.float32)/255)[0]
assert max(prob) > 0.95, "Low prediction accuracy."
print 'Classified as %d with probability %f' % (prob.argmax(), max(prob))

#%% Validation accuracy
# @@@ AUTOTEST_OUTPUT_IGNORED_CELL
valid_acc = model.score(val_iter)
print 'Validation accuracy: %f%%' % (valid_acc *100,)
assert valid_acc > 0.9, "Low validation accuracy."


#%%
data = mx.symbol.Variable('data')
# first conv layer
conv1 = mx.sym.Convolution(data=data, kernel=(5,5), num_filter=20)
tanh1 = mx.sym.Activation(data=conv1, act_type="tanh")
pool1 = mx.sym.Pooling(data=tanh1, pool_type="max", kernel=(2,2), stride=(2,2))
# second conv layer
conv2 = mx.sym.Convolution(data=pool1, kernel=(5,5), num_filter=50)
tanh2 = mx.sym.Activation(data=conv2, act_type="tanh")
pool2 = mx.sym.Pooling(data=tanh2, pool_type="max", kernel=(2,2), stride=(2,2))
# first fullc layer
flatten = mx.sym.Flatten(data=pool2)
fc1 = mx.symbol.FullyConnected(data=flatten, num_hidden=500)
tanh3 = mx.sym.Activation(data=fc1, act_type="tanh")
# second fullc
fc2 = mx.sym.FullyConnected(data=tanh3, num_hidden=10)
# softmax loss
lenet = mx.sym.SoftmaxOutput(data=fc2, name='softmax')
mx.viz.plot_network(symbol=lenet, shape=shape)


#%%
# @@@ AUTOTEST_OUTPUT_IGNORED_CELL
model = mx.model.FeedForward(
    ctx = mx.cpu(0),     # use GPU 0 for training, others are same as before
    symbol = lenet,       
    num_epoch = 5,     
    learning_rate = 0.1)
model.fit(
    X=train_iter,  
    eval_data=val_iter, 
    batch_end_callback = mx.callback.Speedometer(batch_size, 200)
) 
# assert model.score(val_iter) > 0.98, "Low validation accuracy."

#%% Predict on test set
test_lbl = model.predict(test_img)
test_lbl = np.argmax(test_lbl, axis = 1)

#%%
predictions = np.vstack((range(1,test_size+1), test_lbl)).transpose()

#%%
submission = pd.DataFrame(data = predictions, columns = ["ImageId", "Label"])
submission.to_csv("submission.csv", index = False)

