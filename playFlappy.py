#!/usr/bin/env python
from __future__ import print_function

from collections import deque

import cPickle
import pickle
import json

from keras import backend as K
from keras import initializers
from keras.initializers import normal, identity
from keras.models import model_from_json
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import SGD , Adam

import numpy as np
import tensorflow as tf
import skimage as skimage
from skimage import transform, color, exposure
from skimage.transform import rotate
from skimage.viewer import ImageViewer

import time
import argparse
import random

import sys
sys.path.append("game/")
import flappy_bird_gameplay as game

ACTIONS = 2 #VALID ACTIONS; FLAP, NO_FLAP
EPOCH_SIZE = 5000 #THE TIMESTAMP SIZE OF A EPOCH
GAMMA = 0.99 #DECAY_RATE FOR Q_FUNCTION
EXPLORE = 300000.
FINAL_EPSILON = 0.0001
INITIAL_EPSILON = 0.1
REPLAY_MEMORY = 50000 #PREVIOUS OBSERVATIONS IN MEMORT
BATCH = 32 # BATCH SIZE
FRAME_PER_ACTION = 1
LEARNING_RATE = 1e-4
tss = {}
state = "observe/test"


def buildmodel():
    model = Sequential()
    model.add(Convolution2D(32, 8, 8, subsample=(4, 4), border_mode='same',input_shape=(80,80,4)))  
    model.add(Activation('relu'))
    model.add(Convolution2D(64, 4, 4, subsample=(2, 2), border_mode='same'))
    model.add(Activation('relu'))
    model.add(Convolution2D(64, 3, 3, subsample=(1, 1), border_mode='same'))
    model.add(Activation('relu'))
    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dense(2))
    adam = Adam(lr=LEARNING_RATE)
    model.compile(loss='mse',optimizer=adam)
    return model

def plotGraph():
    import matplotlib.pyplot as plt
    import numpy as np
    import json
    import os
    with open('model/graph_p.pickle', 'rb') as f:
        g = pickle.load(f)
    ts = pickle.load(open('model/ts.pickle', 'rb'))
    plt.xlabel('Timestamp')
    plt.ylabel('Q_max')
    plt.plot(g['Q_max'])
    plt.savefig('graph/Fig'+str(ts['t'])+'.png')
    l = {'avgQ': float(np.mean(g['Q_max']))}
    with open('model/avgQMax','a') as f:
        json.dump(l, f)
        f.write(os.linesep)

def saveProgress(t, model, eps):
    tss = {}
    tss['t'] = t
    tss['eps'] = eps
    print("Now we save model")
    model.save_weights("model.h5", overwrite=True)
    with open("model/ts.pickle", "wb") as fp:
        pickle.dump(tss,fp)
    with open("model/model.json", "w") as outfile:
        json.dump(model.to_json(), outfile)

def trainNetwork(model,args):
    
    game_state = game.GameState()
    prev_obs = deque() #PREVIOUS OBSERVATIONS STORED IN REPLAY MEMORY
    init_state = np.zeros(ACTIONS)
    init_state[0] = 1
    xt, r, terminal = game_state.frame_step(init_state)
    xt = skimage.color.rgb2gray(xt)
    xt = skimage.transform.resize(xt,(80,80))
    xt = skimage.exposure.rescale_intensity(xt,out_range=(0,255))
    s_t = np.stack((xt, xt, xt, xt), axis=2)
    s_t = s_t.reshape(1, s_t.shape[0], s_t.shape[1], s_t.shape[2]) 
    tss = {"t": 0}
    if args != "Train":
        t = 0
        temp = OBSERVE = 2147483648   
        epsilon = FINAL_EPSILON
        model.load_weights("model.h5")
        adam = Adam(lr=LEARNING_RATE)
        model.compile(loss='mse',optimizer=adam)  
    else:                      
        OBSERVE = 3200 
        ts = pickle.load(open('model/ts.pickle', 'rb'))
        data = cPickle.load(open('model/xxx.dmp', 'r'))  
        model.load_weights("model.h5")
        temp = t = ts['t']
        epsilon = ts['eps']
        prev_obs = data['D']

    graph_p = {'reward':[],'Q_max':[],'loss':[]}
    while (t < temp + EPOCH_SIZE):
        loss, Q_sa, action_index, r_t = 0, 0, 0, 0
        a_t = np.zeros([ACTIONS])
        if t % FRAME_PER_ACTION == 0:
            if random.random() <= epsilon:
                action_index = random.randrange(ACTIONS)
                a_t[action_index] = 1
            else:
                q = model.predict(s_t)    
                max_Q = np.argmax(q)
                action_index = max_Q
                a_t[max_Q] = 1
        if epsilon > FINAL_EPSILON and t > OBSERVE:
            epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / EXPLORE
        x_t1_colored, r_t, terminal = game_state.frame_step(a_t)
        x_t1 = skimage.color.rgb2gray(x_t1_colored)
        x_t1 = skimage.transform.resize(x_t1,(80,80))
        x_t1 = skimage.exposure.rescale_intensity(x_t1, out_range=(0, 255))
        x_t1 = x_t1.reshape(1, x_t1.shape[0], x_t1.shape[1], 1)
        s_t1 = np.append(x_t1, s_t[:, :, :, :3], axis=3)
        prev_obs.append((s_t, action_index, r_t, s_t1, terminal))
        if len(prev_obs) > REPLAY_MEMORY:
            prev_obs.popleft()
        if t > OBSERVE:
            minibatch = random.sample(prev_obs, BATCH)
            inputs = np.zeros((BATCH, s_t.shape[1], s_t.shape[2], s_t.shape[3]))  
            print (inputs.shape)
            targets = np.zeros((inputs.shape[0], ACTIONS))                         
            for i in range(0, len(minibatch)):
                st, at, rt, state_t1, terminal  = minibatch[i][0], minibatch[i][1], minibatch[i][2], minibatch[i][3], minibatch[i][4]
                inputs[i:i + 1] = st
                targets[i] = model.predict(st)  
                Q_sa = model.predict(st)
                if terminal:
                    targets[i, at] = rt
                else:
                    targets[i, at] = rt + GAMMA * np.max(Q_sa)
            loss += model.train_on_batch(inputs, targets)
        s_t = s_t1
        t = t + 1
        if t > OBSERVE and t <= OBSERVE + EXPLORE:
            state = "explore"
        else:
            state = "train"

        print("TIMESTEP", t, "/ STATE", state, \
            "/ EPSILON", epsilon, "/ ACTION", action_index, "/ REWARD", r_t, \
            "/ Q_MAX " , np.max(Q_sa), "/ Loss ", loss)
        graph_p['reward'].append(r_t)
        graph_p['Q_max'].append(np.max(Q_sa))
        graph_p['loss'].append(loss)
        with open('model/graph_p.pickle','wb') as f:
            pickle.dump(graph_p,f)
            print("Dumped -----> graph_p.pickle")
        
    print("Episode finished!")
    if args == "Train":
        plotGraph()
        saveProgress(t, model, epsilon)
                       
if __name__ == "__main__":
    start = time.time()
    args = sys.argv[1]
    model = buildmodel()
    trainNetwork(model,args)
    end = time.time()
    print('Training time for each Epcoh: ', (end - start))
