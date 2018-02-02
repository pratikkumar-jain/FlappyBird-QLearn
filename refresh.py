#RUN THIS FILE TO REFRESH THE ENTIRE PROJECT
def refresh():
    import cPickle
    from collections import deque
    tss = {}
    tss['t'] = 0
    tss['D'] = deque()
    out = open('model/xxx.dmp', 'w')
    cPickle.dump(tss, out)
    out.close()

def buildmodel():
    model = Sequential()
    model.add(Convolution2D(32, 8, 8, subsample=(4, 4), border_mode='same',input_shape=(img_rows,img_cols,img_channels)))  #80*80*4
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

refresh()
