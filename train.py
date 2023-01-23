import torch
import torch.nn as nn
from torch.nn import functional as F

def read_text(path):

    '''
    Read text input
    '''

    with open(path, 'r', encoding = 'utf-8') as f:
        text = f.read()

    chars = sorted(list(set(text)))
    vocab_size = len(chars)

    return text, chars, vocab_size


def get_enc_func(chars):
    '''
    Return functions which encode and decode strings
    '''
    stoi = { ch:i for i,ch in enumerate(chars) }
    itos = { i:ch for i,ch in enumerate(chars) }
    encode = lambda s: [stoi[c] for c in s]
    decode = lambda l: ''.join([itos[i] for i in l])

    return encode, decode


def get_batch(split):
    '''
    generate a small batch of data of inputs x and targets y
    returns a tensor of size batch_size * block_size
    
    '''
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    return x, y

class BigramLanguageModel(nn.Module):

    def __init__(self, vocab_size):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        # assigns a score for each of the cahracters which are possible for us
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)


    def forward(self, idx, targets=None):

        # idx and targets are both (B,T) tensor of integers
        logits = self.token_embedding_table(idx) # (B,T,C)


        if targets is None:
            loss = None
        else:
            # pytorch needs C to be second dimension so we strech out
            # B and T dimensions
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)

        loss = F.cross_entropy(logits, targets)

        return logits, loss

    

if __name__ == '__main__':

    torch.manual_seed(123)

    text, chars, vocab_size = read_text('tinyshakespeare.txt')
    encode, decode = get_enc_func(chars)
    
    # process into training and test data
    data = torch.tensor(encode(text), dtype = torch.long)
    n = int(0.9 * len(data))
    train_data = data[:n]
    val_data = data[n:]

    # chunking
    block_size = 8
    batch_size = 4
    xb, yb = get_batch('train')

    m = BigramLanguageModel(vocab_size)
    out = m(xb, yb)



    
    
    