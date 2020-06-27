#!/usr/bin/env python
# coding: utf-8

import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
import numpy as np
import tensorflow as tf # NOTE: This code runs with tensorflow version 2.0.0 # also may need to run conda update mkl
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Input, Dense, LSTM, GRU, Embedding, Lambda, concatenate, multiply, add
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tqdm import tqdm,trange


# # Set parameters

max_train_length = 5
alphabetSize = 5
num_train_examples = int(np.power(alphabetSize,max_train_length))
losstype='mse'
matDim = 5


# # Testing Approximation of Multiset Automaton

def gen_complex_diag(matDim,numMats=1,rad=1):
    '''Generates a set of diagonal matrices with complex entries.  Each entry is drawn randomly
    from -1, 1 for real and imaginary parts independently.  If rad is set to 1, then the spectral radius
    is set to 1 by renormalizing the vectors at the end of the generation process.  If rad is set to 2,
    then all entries are normalized to have absolute value of 1.  If rad is set to 3, each entry is 
    capped at absolute value of 1.'''
    diagMats = []
    complexDim = matDim // 2
    realDim = matDim % 2
    for x in range(numMats):
        real = np.random.uniform(-1,1,complexDim+realDim)
        imag = np.random.uniform(-1,1,complexDim)
        if realDim > 0:
            imag = np.append(imag, [0])
        diagMat = real + 1j * imag
        if rad==1:
            # normalize by the largest element
            diagMat = diagMat / np.abs(diagMat).max()
        elif rad==2:
            # normalize all entries
            diagMat = diagMat / np.abs(diagMat)
        elif rad==3:
            normed = diagMat / np.abs(diagMat)
            diagMat = np.where(np.abs(diagMat)>np.abs(normed), normed, diagMat)
        diagMats.append(diagMat)
    return diagMats

def gen_complex_lambdarho(matDim):
    '''Generates a complex valued vector corresponding to lambdarho. We push all weights onto lambda
    and call it lambdarho which allows us to treat rho as all 1s.'''
    complexDim = matDim // 2
    realDim = matDim % 2
    # Note that in the complex conjugate case the end result is a doubling, so we do that here
    real = np.random.uniform(-2,2,complexDim)
    imag = np.random.uniform(-2,2,complexDim)
    if realDim > 0:
        real = np.append(real, np.random.uniform(-1,1,1))
        imag = np.append(imag, [0])
    vec = real + 1j * imag
    return vec

def gen_strings_complex(lambdarho, tMats, num_train_examples, max_train_length):
    alphabetSize = len(tMats)
    X = np.zeros((num_train_examples,100))
    stringWeights = np.zeros((num_train_examples))
    for i in tqdm(range(num_train_examples), desc='Generating train examples: '):
        n = np.random.randint(1,max_train_length)
        curWeight = lambdarho
        for j in range(1,n+1):
            X[i,-j] = np.random.randint(1,alphabetSize+1)
            curWeight = np.multiply(curWeight, tMats[int(X[i,-j]-1)])
        stringWeights[i] = np.sum(curWeight).real
    return X, stringWeights

def gen_strings_complex_length(lambdarho, tMats, num_train_examples, train_length):
    alphabetSize = len(tMats)
    X = np.zeros((num_train_examples,100))
    stringWeights = np.zeros((num_train_examples))
    for i in tqdm(range(num_train_examples), desc='Generating train examples: '):
        n = train_length
        curWeight = lambdarho
        for j in range(1,n+1):
            X[i,-j] = np.random.randint(1,alphabetSize+1)
            curWeight = np.multiply(curWeight, tMats[int(X[i,-j]-1)])
        stringWeights[i] = np.sum(curWeight).real
    return X, stringWeights

def gen_all_complex_length(lambdarho, tMats, train_length):
    alphabetSize = len(tMats)
    num_train_examples = np.power(alphabetSize, train_length)
    X = np.zeros((num_train_examples,100))
    stringWeights = np.zeros((num_train_examples))
    for i in tqdm(range(num_train_examples), desc='Generating train examples: '):
        n = train_length
        curWeight = lambdarho
        k = i
        for j in range(1,n+1):
            X[i,-j] = np.int((k % alphabetSize) + 1)
            curWeight = np.multiply(curWeight, tMats[int(X[i,-j]-1)])
            k = k // alphabetSize
        stringWeights[i] = np.sum(curWeight).real
    return X, stringWeights


# # Helper function

def fun(x, mask):
    if K.is_keras_tensor(mask):
        mask_cast = K.cast(mask, 'float32')
        expanded = K.expand_dims(mask_cast)
        return K.sum(expanded * x, axis=1)
    return K.sum(x, axis=1)


def trainModel(model, numEpochs, trainData, trainWeights, lrtype='val',stop=0.00001,verbose=1):
    # train
    if lrtype=='val':
        earlyStop = EarlyStopping(monitor='val_loss',patience=10,verbose=verbose,min_delta=stop)
        lrscheduler = ReduceLROnPlateau(monitor='val_loss',patience=2,factor=0.5,verbose=verbose,min_delta=stop,cooldown=3)
        history = model.fit(trainData, trainWeights, epochs=numEpochs, batch_size=128, verbose=verbose,
            shuffle=True, validation_split=0.0123456789,
            callbacks=[earlyStop,lrscheduler])
    else:
        earlyStop = EarlyStopping(monitor='loss',patience=10,verbose=verbose,min_delta=stop)
        lrscheduler = ReduceLROnPlateau(monitor='loss',patience=2,factor=0.5,verbose=verbose,min_delta=stop,cooldown=3)
        history = model.fit(trainData, trainWeights, epochs=numEpochs, batch_size=128, verbose=verbose,
            shuffle=True,
            callbacks=[earlyStop,lrscheduler])

    temp_we = {}
    for idx, layer in enumerate(model.layers):
        w = layer.get_weights()
        temp_we[idx] = w
    return temp_we, history

def loadModel(model, modelWeights):
    # load weights
    for idx, layer in enumerate(model.layers):
        w = modelWeights[idx]
        layer.set_weights(w)
    return model

def evaluateModel(model, testData, testWeights):
    # prediction
    preds = model.predict(testData, batch_size=128, verbose=0)
    return np.dot(np.squeeze(preds)-testWeights, np.squeeze(preds)-testWeights)/len(testWeights)


# # Complex Normalized Multiset Model

def complexNormedMultiply(q, mask):
    x = q[0]
    y = q[1]
    r = q[2]
    initX = q[3]
    initY = q[4]
    initR = q[5]
    # Here x is the real part and y is the imaginary part
    if tf.is_tensor(mask):
        # this sets masked values to 1+0i
        mask_cast = K.cast(mask, 'float32')
        expanded = K.expand_dims(mask_cast)
        zeroX = expanded * x
        newY = expanded * y
        newR = expanded * r
        # here I flip the mask (essentially XOR)
        antiMask = tf.ones(expanded.shape)-expanded
        newX = zeroX+antiMask
    else:
        newX = x
        newY = y
        newR = r
    sumVecs = tf.math.sqrt(tf.multiply(newX,newX)+tf.multiply(newY,newY))
    normedX = newX/sumVecs
    normedY = newY/sumVecs
    normedR = newR# Leave this value alone
    initSum = tf.math.sqrt(tf.multiply(initX,initX)+tf.multiply(initY,initY))
    inX = initX/initSum
    inY = initY/initSum
    # Using builtin complex numbers
    complexVec = tf.complex(normedX,normedY)
    initVec = tf.complex(inX,inY)
    complexOut = K.prod(complexVec,axis=1)
    newCOut = tf.multiply(complexOut,tf.expand_dims(initVec,0))

    rOut = K.sum(normedR,axis=1)
    newROut = tf.add(rOut,tf.expand_dims(initR,0))

    # This part is new to account for rho
    expR = K.exp(newROut)
    mainReal = tf.math.real(newCOut)
    vecReal = tf.multiply(expR,mainReal)
    singleReal = K.sum(vecReal, axis=1)
    return singleReal


def get_normedcartset_model(max_length, alphabetSize=10, edim=50):
    input_txt = Input(shape=(max_length,))
    # We want x to be the real part and y to be the imaginary part and r is the magnitude in a sense
    # e^r(x+yi)
    if False:
        # This embedding setup is simpler but does not work as well
        x = Embedding(alphabetSize+1, edim, mask_zero=True)(input_txt)
        y = Embedding(alphabetSize+1, edim, mask_zero=True)(input_txt)
        r = Embedding(alphabetSize+1, edim, mask_zero=True)(input_txt)
    else:
        # Note that VarianceScaling is Glorot uniform if scale=1.0, mode='fan_avg' and distribution='uniform'
        # Glorot uniform gives NaN's since that is way too large, so I set scale to be lower
        scaleVar = 0.005
        x = Embedding(alphabetSize+1, edim, mask_zero=True, embeddings_initializer=tf.keras.initializers.VarianceScaling(scale=scaleVar, mode='fan_avg', distribution='uniform'))(input_txt)
        y = Embedding(alphabetSize+1, edim, mask_zero=True, embeddings_initializer=tf.keras.initializers.VarianceScaling(scale=scaleVar, mode='fan_avg', distribution='uniform'))(input_txt)
        r = Embedding(alphabetSize+1, edim, mask_zero=True, embeddings_initializer=tf.keras.initializers.VarianceScaling(scale=scaleVar, mode='fan_avg', distribution='uniform'))(input_txt)
    # the init variables account for lambda*rho
    # thus lambda*rho = e^initR(initX+initY)
    initX = K.variable(value=np.ones(edim),dtype='float32')
    initY = K.variable(value=np.zeros(edim),dtype='float32')
    initR = K.variable(value=np.zeros(edim),dtype='float32')
    CM = Lambda(complexNormedMultiply, output_shape=lambda s: (s[0][0], s[0][2]*3), name="NormedComplexMultiply")
    z = CM([x,y,r,initX,initY,initR])
    model = Model(input_txt, z)
    adam = Adam(lr=1e-4, epsilon=1e-3)
    model.compile(optimizer=adam, loss='mse')
    return model


# # Run 10 tests per matrix size

for matDim in range(2,21):
    runMSE = {}
    for dataNum in range(10):
        # Generate a new complex diagonal automaton
        diagMats = gen_complex_diag(matDim, alphabetSize, rad=1)
        lambdarho = gen_complex_lambdarho(matDim)
        trainData, trainWeights =  gen_all_complex_length(lambdarho, diagMats, max_train_length)
        # Store it for future reference
        baseFilename = 'data/complexDiag-%d-%d-' % (matDim, dataNum)
        np.save(baseFilename+"diagMats",diagMats)
        np.save(baseFilename+"lambdarho",lambdarho)
        np.save(baseFilename+"trainData",trainData)
        np.save(baseFilename+"trainWeights",trainWeights)
    
        for runNum in range(10):
            # Train our model on the data
            model = get_normedcartset_model(100, alphabetSize, matDim)
            normedcartset_mse_we, history = trainModel(model, numEpochs=500, trainData=trainData, trainWeights=trainWeights, lrtype='loss')
            runMSE[(matDim,dataNum,runNum)] = history.history['mse']
            baseFilename = 'data/complexDiag-%d-%d-%d-' % (matDim, dataNum, runNum)
            # Store the model for future reference
            np.save(baseFilename+"model",normedcartset_mse_we)
    # Store the results
    baseFilename = 'data/complexDiag-%d-' % matDim
    np.save(baseFilename+"runMSE",runMSE)

