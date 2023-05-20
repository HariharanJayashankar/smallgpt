import torch
import torch.nn.functional as F
from makemorebigram import *

def createTrainingSet(filepath, device):

    with open(filepath, 'r') as f:
        words = f.read().splitlines()

    stoi, itos = bigram_lookup(words)

    xs, ys = [], []

    for w in words:
        chs = ['.'] + list(w) + ['.']
        for ch1, ch2 in zip(chs, chs[1:]):
            ix1 = stoi[ch1]
            ix2 = stoi[ch2]
            xs.append(ix1)
            ys.append(ix2)

    xs = torch.tensor(xs, device=device)
    ys = torch.tensor(ys, device=device)

    xenc = F.one_hot(xs, num_classes=27).float()
    yenc = F.one_hot(ys, num_classes=27).float()

    return xenc, yenc, xs, ys

def feedForward(xenc, W):
    # W is 27 * 27 matrix

    # single layer NN basically a logit func
    # with softmax normalization

    out = (xenc @ W).exp()
    probs = out / out.sum(1, keepdims=True)

    return probs

def lossFunc(pred, ys):

    # negative loglik
    loss = -pred[torch.arange(pred.shape[0]), ys].log().mean()
    return loss

def backProp(W, pred, ys):

    W.grad = None #reset grad to 0
    loss = lossFunc(pred, ys)
    loss.backward()

    W.data += -0.1 * W.grad

def gradDescent(
        xs, xenc,
        ys, yenc,
        g, eta=50,
        tol=1e-6, niter=1000
):

    W = torch.randn((27, 27), device=device, generator=g, requires_grad=True)

    loss0 = 1
    error = 1
    i = 1

    while i < niter and error > tol:
        p = feedForward(xenc, W)
        backProp(W, p, ys)

        i += 1
        loss1 = lossFunc(p, ys)
        error = torch.absolute(loss0 - loss1)
        loss0 = loss1

        if i % 100 == 0:
            print(lossFunc(p, ys))

    return p ,W

if __name__ == '__main__':

    device = 'cuda'
    xenc, yenc, xs, ys = createTrainingSet('names.txt', device)

    g = torch.Generator(device=device).manual_seed(2147483647)
    P, W = gradDescent(xs, xenc, ys, yenc, g)
    words = sampleNames(P, 50)



