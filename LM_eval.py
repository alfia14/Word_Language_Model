''' This file is created using the evaluate function and other functions called by the evaluate function of the LM_train.py file.
The evaluate function calculates the loss by feeding the input dataset i.e, the train set, the test set and the valid set 
on the saved and already trained model.pt. And finally the perplexity is calculated using the loss'''

#The dataset to be given for evaluation is specified by the dataset variable which is passed as an argument

import argparse
import torch
import torch.nn as nn
import data
import math

parser = argparse.ArgumentParser(description='PyTorch Language Model')

parser.add_argument('--data', type=str, default='./data/wikitext-2',
                    help='location of the data corpus')
parser.add_argument('--dataset', type=str, default='test',
                    help='type of dataset')
parser.add_argument('--checkpoint', type=str, default='./model.pt',
                    help='model checkpoint to use')
parser.add_argument('--bptt', type=int, default=35,
                    help='sequence length')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')
parser.add_argument('--cuda', action='store_true',
                    help='use CUDA')
parser.add_argument('--temperature', type=float, default=1.0,
                    help='temperature - higher will increase diversity')
parser.add_argument('--log-interval', type=int, default=200,
                    help='reporting interval')
args = parser.parse_args()

torch.manual_seed(args.seed)
device = torch.device("cuda" if args.cuda else "cpu")

corpus = data.Corpus(args.data)

def batchify(data, bsz):
    # Work out how cleanly we can divide the dataset into bsz parts.
    nbatch = data.size(0) // bsz
    # Trim off any extra elements that wouldn't cleanly fit (remainders).
    data = data.narrow(0, 0, nbatch * bsz)
    # Evenly divide the data across the bsz batches.
    data = data.view(bsz, -1).t().contiguous()
    return data.to(device)

eval_batch_size = 10

if(args.dataset == 'test'):
    data = batchify(corpus.test, eval_batch_size)
elif(args.dataset == 'train'):
    data = batchify(corpus.train, eval_batch_size)
else:
    data = batchify(corpus.valid, eval_batch_size)

# Set the random seed manually for reproducibility.

with open(args.checkpoint, 'rb') as f:
    model = torch.load(f, map_location=device)

criterion = nn.CrossEntropyLoss()

def repackage_hidden(h):
    """Wraps hidden states in new Tensors, to detach them from their history."""

    if isinstance(h, torch.Tensor):
        return h.detach()
    else:
        return tuple(repackage_hidden(v) for v in h)

def get_batch(source, i):
    seq_len = min(args.bptt, len(source) - 1 - i)
    data = source[i:i+seq_len]
    target = source[i+1:i+1+seq_len].view(-1)
    return data, target

def evaluate(data_source):
    
    model.eval()
    total_loss = 0.
    ntokens = len(corpus.dictionary)
    
    hidden = model.init_hidden(eval_batch_size)
    with torch.no_grad():
        for i in range(0, data_source.size(0) - 1, args.bptt):
            data, targets = get_batch(data_source, i)
            output, hidden = model(data, hidden)
            hidden = repackage_hidden(hidden)
            total_loss += len(data) * criterion(output, targets).item()
    return total_loss / (len(data_source) - 1)

'''Calculating the loss and perplexity on the dataset given'''

loss = evaluate(data)
if(args.dataset == 'test'):
    print('=' * 89)
    print(' test_loss {:5.2f} | ppl {:8.2f}'.format(
    loss, math.exp(loss)))
elif(args.dataset == 'train'):
    print('=' * 89)
    print(' train_loss {:5.2f} | ppl {:8.2f}'.format(
    loss, math.exp(loss)))
else:
    print('=' * 89)
    print(' valid_loss {:5.2f} | ppl {:8.2f}'.format(
    loss, math.exp(loss)))

print('=' * 89)


