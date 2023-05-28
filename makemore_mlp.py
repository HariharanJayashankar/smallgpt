import torch
import numpy as np
import random
import torch.nn.functional as F
import matplotlib.pyplot as plt
from makemore import *

d = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
g = torch.Generator(device=d).manual_seed(2147483647)

def make_data(words, block_size=3, device=d):
    '''
    block_size == context size whcih is the 
    size of blocks we feed into the neural net
    '''

    stoi, itos = bigram_lookup(words)

    X, Y = [], []

    for w in words:

        context = [0] * block_size

        for ch in w + '.':

            ix = stoi[ch]
            X.append(context)
            Y.append(ix)
            context = context[1:] + [ix]

    X = torch.tensor(X, device=d)
    Y = torch.tensor(Y, device=d)


    return X, Y

def getEmbeddings(X, C):

    emb = C[X]

    # now we need to cat the last two dimension of emb
    emb = emb.view(emb.shape[0], emb.shape[1]*emb.shape[2]).T

    return emb

def initHiddenLayer1(emb, nweights=100):

    dim1, dim2 = emb.shape
    W1 = torch.randn((nweights, dim1 ),device=d, generator=g)
    b1 = torch.randn((nweights, 1), device=d, generator=g)

    return W1, b1

def initOutputLayer(nweights=100):
    
    W2 = torch.randn((27, nweights), device=d, generator=g)
    b2 = torch.randn((27, 1), device=d, generator=g)
    
    return W2, b2

def forwardPass(X, Y, parameters, batchidx=None):
    
    C, W1, b1, W2, b2 = parameters

    if batchidx is not None:
        Xidx = X[batchidx]
        Yidx = Y[batchidx]
    else:
        Xidx = X
        Yidx = Y

    emb = getEmbeddings(Xidx, C)
    h = torch.tanh(W1 @ emb + b1)
    logits = W2 @ h + b2
    loss = F.cross_entropy(logits.T, Yidx)

    return loss

def backwardPass(loss, parameters, learnrate=0.1):

    for p in parameters:
        p.grad = None

    loss.backward()

    for p in parameters:
        p.data += -learnrate * p.grad

def gradDescentInner(
    X, Y,
    parameters,
    batchsize=32,
    learnrate=0.1,
    niter=1000
):
    
    for p in parameters:
        p.requires_grad = True

    for _ in range(niter):

        # mini batch
        ix = torch.randint(0, X.shape[0], (batchsize,))

        loss = forwardPass(X, Y, parameters, batchidx=ix)
        backwardPass(loss, parameters, learnrate)


    print(loss)
    return parameters

def gradDescent(X, Y, 
                nembed=10,
                nweights=300,
                batchsize=32,
                learnrate=0.1, 
                niter=1000,
                init=True
                ):

    C = torch.randn((27, nembed), device=d, generator=g)
    emb = getEmbeddings(X, C)
    W1, b1 = initHiddenLayer1(emb, nweights)
    W2, b2= initOutputLayer(nweights)

    parameters = [C, W1, b1, W2, b2]

    parameters = gradDescentInner(
        X, Y, parameters,
        batchsize, learnrate,
        niter
    )

    return parameters


if __name__ == '__main__':
    
    words = readdata('names.txt')

    # training, validation, testing split
    # 80 - 10 - 10
    random.seed(42)
    random.shuffle(words)
    n1 = int(0.8 * len(words))
    n2 = int(0.9 * len(words))

    stoi, itos = bigram_lookup(words)
    Xtr, Ytr = make_data(words[:n1], block_size=3)
    Xdev, Ydev = make_data(words[n1:n2], block_size=3)
    Xtest, Ytest = make_data(words[n2:], block_size=3)

    parameters = gradDescent(Xtr, Ytr, niter=30000, learnrate=0.1)
    # learning decay
    parameters = gradDescentInner(Xtr, Ytr, parameters, niter=30000, learnrate=0.01)


    # for tuning hyper parameters (manually tweak hypers atm)
    lossdev = forwardPass(Xdev, Ydev, parameters)
    print(lossdev)

    lossfinal = forwardPass(Xtest, Ytest, parameters)
    print(lossfinal)

