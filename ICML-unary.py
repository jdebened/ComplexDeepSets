#!/usr/bin/env python
# coding: utf-8
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
import numpy as np
import tensorflow as tf # NOTE: This code runs with tensorflow version 2.0.0 # also may need to run conda update mkl
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Input, Embedding, Lambda, concatenate, multiply, add
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tqdm import tqdm,trange
import sys

# # Set parameters

matDim = 2
max_train_length = 20
losstype='mse'
alphabetSize = 1


# # Testing Approximation of Multiset Automaton

def gen_init(matDim):
    initVec = [np.eye(matDim)[0]]
    return initVec

def gen_rho(matDim):
    rho = np.random.rand(matDim,1)
    return rho

from scipy.stats import ortho_group
def gen_unary(matDim):
    # In the unary case we generate an orthogonal matrix to make sure the norm does not explode
    return [ortho_group.rvs(matDim)]


def gen_unary_strings(initVec, tMat, rho, max_train_length):
    '''Generates strings and corresponding weights from initial weights (initVec), transition matrix (tMat),
    and final weights (rho).  All strings up to max_train_length are generated.'''
    X = np.zeros((max_train_length+1,100))
    stringWeights = np.zeros((max_train_length+1))
    for i in tqdm(range(max_train_length+1), desc='Generating train examples: '):
        curWeight = initVec
        for j in range(1,i+1):
            X[i,-j] = np.int(1)
            curWeight = np.matmul(curWeight, tMat[0])
        stringWeights[i] = np.matmul(curWeight, rho)[0][0]
    return X, stringWeights


# # Get Data
def get_data(matDim=matDim, gen=True, saveData=True, num=0):
    if gen:
        unaryMat = gen_unary(matDim)
        initVec = gen_init(matDim)
        rhoVec = gen_rho(matDim)
        trainData, trainWeights = gen_unary_strings(initVec, unaryMat, rhoVec, max_train_length)
        # If we want to generate and save the training and test sets
        if saveData:
            baseFilename = "data/unary-%d-%d-" % (matDim, num)
            np.save(baseFilename+"trainData",trainData)
            np.save(baseFilename+"trainWeights",trainWeights)
            np.save(baseFilename+"initVec",initVec)
            np.save(baseFilename+"rhoVec",rhoVec)
            np.save(baseFilename+"unaryMat",unaryMat)
    else:
        baseFilename = "data/unary-%d-%d-" % (matDim, num)
        trainData = np.load(baseFilename+"trainData.npy")
        trainWeights = np.load(baseFilename+"trainWeights")
        if False:
            initVec = np.load(baseFilename+"initVec")
            rhoVec = np.load(baseFilename+"rhoVec")
            unaryMat = np.load(baseFilename+"unaryMat")
    return trainData, trainWeights


# # Helper function

def fun(x, mask):
    if K.is_keras_tensor(mask):
        mask_cast = K.cast(mask, 'float32')
        expanded = K.expand_dims(mask_cast)
        return K.sum(expanded * x, axis=1)
    return K.sum(x, axis=1)


def trainModel(model, numEpochs, trainData, trainWeights, stop=0.00001,verbose=1):
    # train
    # for unary I changed patience on early stopping since epochs are small
    earlyStop = EarlyStopping(monitor='loss',patience=500,verbose=verbose,min_delta=stop)
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
    normedR = newR
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
        # Used to do this, but didn't seem to scale well
        x = Embedding(alphabetSize+1, edim, mask_zero=True)(input_txt)
        y = Embedding(alphabetSize+1, edim, mask_zero=True)(input_txt)
        r = Embedding(alphabetSize+1, edim, mask_zero=True)(input_txt)
    else:
        # Note that VarianceScaling is Glorot uniform if scale=1.0, mode='fan_avg' and distribution='uniform'
        # Glorot uniform gives NaN's since that is way too large, so I set scale lower
        scaleVar = 1e-5
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


# # Train Normed Complex Cartesian model
for matDim in range(2,21):
    runMSE = {}
    for dataNum in range(10):
        trainData, trainWeights = get_data(matDim=matDim, gen=True, saveData=True,num=dataNum)

        for runNum in range(10):
            model = get_normedcartset_model(100, alphabetSize, matDim)
            normedcartset_mse_we, history = trainModel(model,numEpochs=30000,trainData=trainData,trainWeights=trainWeights)
            runMSE[(dataNum,runNum)] = history.history['mse']
            baseFilename = "data/unary-%d-%d-%d-" % (matDim, dataNum, runNum)
            np.save(baseFilename+"model",normedcartset_mse_we)
    baseFilename = "data/unary-%d-%d-" % matDim
    np.save(baseFilename+"runMSE",runMSE)
