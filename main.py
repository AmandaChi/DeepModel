import os, sys
from keras.optimizers import *
from keras.callbacks import *

Xtrain, Ytrain =
Xvalid, Yvalid = 

print('train shapes:', Xtrain.shape, Ytrain.shape)
print('valid shapes:', Xvalid.shape, Yvalid.shape)

from transformer import WPTransformer, LRSchedulerPerStep, LRSchedulerPerEpoch

d_model = 512
d_inner_hid = 512
n_head = 8
d_k = 64
d_v = 64
n_days = 100
n_alpha = 300
layers = 1
dropout = 0.1
M = WPTransformer(d_model, d_inner_hid, n_head, d_k, d_v, n_days, n_alpha, layers, dropout)

mfile = 'models/model.h5'

lr_scheduler = LRSchedulerPerStep(d_model, 4000)   # there is a warning that it is slow, however, it's ok.
#lr_scheduler = LRSchedulerPerEpoch(d_model, 4000, Xtrain.shape[0]/64)  # this scheduler only update lr per epoch
model_saver = ModelCheckpoint(mfile, save_best_only=True, save_weights_only=True)

s2s.compile(Adam(0.001, 0.9, 0.98, epsilon=1e-9))
s2s.model.summary()
try: s2s.model.load_weights(mfile)
except: print('\n\nnew model')

m.model.fit([Xtrain, Ytrain], None, batch_size=64, epochs=30, \
	    validation_data=([Xvalid, Yvalid], None), \
			    callbacks=[lr_scheduler, model_saver])
