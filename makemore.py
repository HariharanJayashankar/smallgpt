import numpy as np
import torch


def bigram_lookup(words):

    # get lookup table for each letter/characters
    # and map them to unique indices

    chars = sorted(list(set(''.join(words))))
    stoi = {s:i+1 for i,s in enumerate(chars)}

    # dot is a char signifying either begining or
    # end of a word
    stoi['.'] = 0

    itos = {i:s for s, i in stoi.items()}

    return stoi, itos

def bigramcount(words, stoi):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    N = torch.zeros((28, 28), dtype=torch.int32, device=device)

    for w in words:
        chs = ['.'] + list(w) + ['.']
        for ch1, ch2 in zip(chs, chs[1:]):

            ix1 = stoi[ch1]
            ix2 = stoi[ch2]
            N[ix1, ix2] += 1

    return N


def probFromN(N):
    # torch broadcasting semantics : https://pytorch.org/docs/stable/notes/broadcasting.html

    P = N.float()

    # P is 27 x 27
    # P.sum is 27 x 1
    # This broadcasts how we want

    P /=  P.sum(1, keepdim=True)

    return P


def sampleN(P, idx, g):

    ix = torch.multinomial(P[idx], num_samples=1, replacement=True, generator=g).item()


    return ix

def sampleWord(P, itos, g):

    ix= 0
    c = ''
    while True:
        ix = sampleN(P, ix, g)
        c += itos[ix]

        if ix == 0:
            break

    c = c[:-1]

    return c


if __name__ == '__main__':


    with open('names.txt', 'r') as f:
        words = f.read().splitlines()

    # ==  bigram model (garbage but useful exercise) == #
    stoi, itos = bigram_lookup(words)
    N = bigramcount(words, stoi)
    P = probFromN(N)

    # generate some names
    g = torch.Generator(device='cuda').manual_seed(2147483647)
    words = []
    for i in range(50):
        word = sampleWord(P, itos, g)
        words.append(word)



