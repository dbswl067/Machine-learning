'''
Created on 15.12.2014

@author: Peter U. Diehl
'''

import scipy.ndimage as sp
import numpy as np
import pylab
import os
import sys
import getopt


def randomDelay(minDelay, maxDelay):
    return np.random.rand() * (maxDelay - minDelay) + minDelay


def computePopVector(popArray):
    size = len(popArray)
    complex_unit_roots = np.array([np.exp(1j * (2 * np.pi / size) * cur_pos) for cur_pos in range(size)])
    cur_pos = (np.angle(np.sum(popArray * complex_unit_roots)) % (2 * np.pi)) / (2 * np.pi)
    return cur_pos


def sparsenMatrix(baseMatrix, pConn):
    weightMatrix = np.zeros(baseMatrix.shape)
    numWeights = 0
    numTargetWeights = baseMatrix.shape[0] * baseMatrix.shape[1] * pConn
    weightList = [0] * int(numTargetWeights)
    while numWeights < int(numTargetWeights):
        idx = (np.int32(np.random.rand() * baseMatrix.shape[0]), np.int32(np.random.rand() * baseMatrix.shape[1]))
        if not (weightMatrix[idx]):
            weightMatrix[idx] = baseMatrix[idx]
            weightList[numWeights] = (idx[0], idx[1], baseMatrix[idx])
            numWeights += 1
    return weightMatrix, weightList


def create_weights():
    nInput = 784
    nE = 400
    nI = nE

    weight = {}
    weight['ee_input'] = 1E-6
    weight['ei_input'] = 0.
    weight['ee'] = 0.1
    weight['ei'] = 10.4
    weight['ie'] = 4.0
    weight['ii'] = 0.4

    pConn = {}
    pConn['ee_input'] = 0.02
    pConn['ei_input'] = 0.
    pConn['ee'] = 1.0
    pConn['ei'] = 0.0025
    pConn['ie'] = 0.9
    pConn['ii'] = 0.1

    # Initialize seed for the random number generators
    np.random.seed(0)

    if (stoc_enable == 0):
        dataPath = './random/'
    else:
        dataPath = './random_stoc/'

    if (stoc_enable == 0):
        print('Create random connection matrix between input(Xe) and excitatory(Ae) neurons')
        connNameList = ['XeAe']
        for name in connNameList:
            weightMatrix = np.random.random((nInput, nE)) + 0.01
            weightMatrix *= weight['ee_input']
            if pConn['ee_input'] < 1.0:
                weightMatrix, weightList = sparsenMatrix(weightMatrix, pConn['ee_input'])
            else:
                weightList = [(i, j, weightMatrix[i, j]) for j in range(nE) for i in range(nInput)]
            # np.set_printoptions(threshold='nan')
            # print weightMatrix.transpose()
            print('Saving connection matrix', dataPath + name, '\n')
            np.save(dataPath + name, weightList)
    else:
        print('Create random connection matrix between input(Xe) and excitatory(Ae) neurons')
        connNameList = ['XeAe']
        for name in connNameList:
            weightMatrix = np.random.random((nInput, nE))
            weightMatrix = (weightMatrix < pConn['ee_input']) * 1.0
            weightMatrix += ((1.0 - weightMatrix) * weight['ee_input'])
            # np.set_printoptions(threshold='nan')
            # print weightMatrix
            weightList = [(i, j, weightMatrix[i, j]) for j in range(nE) for i in range(nInput)]
            print('Saving connection matrix', dataPath + name, '\n')
            np.save(dataPath + name, weightList)

    print('Create connection matrix from excitatory(Ae) to inhibitory(Ai) neurons')
    connNameList = ['AeAi']
    for name in connNameList:
        if nE == nI:
            weightList = [(i, i, weight['ei']) for i in range(nE)]
        else:
            weightMatrix = np.random.random((nE, nI))
            weightMatrix *= weight['ei']
            weightMatrix, weightList = sparsenMatrix(weightMatrix, pConn['ei'])
        print('Saving connection matrix', dataPath + name, '\n')
        np.save(dataPath + name, weightList)

    print('Create connection matrix from inhibitory(Ai) to excitatory(Ae) neurons')
    connNameList = ['AiAe']
    for name in connNameList:
        if nE == nI:
            weightMatrix = np.ones((nI, nE))
            weightMatrix *= weight['ie']
            for i in range(nI):
                weightMatrix[i, i] = 0
            weightList = [(i, j, weightMatrix[i, j]) for i in range(nI) for j in range(nE)]
        else:
            weightMatrix = np.random.random((nI, nE))
            weightMatrix *= weight['ie']
            weightMatrix, weightList = sparsenMatrix(weightMatrix, pConn['ie'])
        print('Saving connection matrix', dataPath + name, '\n')
        np.save(dataPath + name, weightList)


if __name__ == "__main__":
    # Parse the command line arguments
    stoc_enable = 1
    opts, args = getopt.getopt(sys.argv[1:], "hs", ["help", "stoc_enable"])

    for opt, arg in opts:
        if opt in ("-h", "--help"):
            print('---------------')
            print('Usage Example:')
            print('---------------')
            print(os.path.basename(__file__) + ' --help        -> Print script usage example')
            print(os.path.basename(__file__) + ' --stoc_enable -> Enable stochasticity')
            sys.exit(1)
        elif opt in ("-s", "--stoc_enable"):
            stoc_enable = 1

    if (stoc_enable):
        print('--------------------------------------------------------------------')
        print('Synapses connecting the input and excitatory neurons are stochastic!')
        print('--------------------------------------------------------------------')
    else:
        print('------------------------------------------------------------------------')
        print('Synapses connecting the input and excitatory neurons are NOT stochastic!')
        print('------------------------------------------------------------------------')

    # Initialize the synaptic weight matrices
    create_weights()

