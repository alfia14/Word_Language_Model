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


''' Analysis Questions and Answers 

1.Describe the language model that gave you the best results on the validation data, and that you used on the test data.  
What was the structure of that language model, and what hyperparameters did you set (and what values did you use to set them)? 
What sources gave you "inspiration" for this model?

-> The language model is implemented by using the LSTM model. The best results were obtained by using a single LSTM layer with 200 hidden dimensions.
The following hyperparameters were set at the following values:

number of hidden units per layer = 200
size of word embeddings = 100
number of layers = 1
dropout = 0
epochs = 6 
gradient clipping = 0.25

Sources: https://www.programcreek.com/python/?code=pytorch%2Fexamples%2Fexamples-master%2Fword_language_model%2Fmain.py

2.Explain what Perplexity is, and what it tells us about a language model (in general).  Compare the Perplexity 
on your validation data, test data, and training data. How did they differ? What conclusions can we draw from this result?

The standard evaluation metric for language models is perplexity. It is defined as the inverse
probability of the corpus according to the language model. In general, perplexity is a measurement 
of how well a probability model predicts a sample. The smaller the perplexity better is the model.

Perplexity of the model on the : 

validation data = 95.31
test data = 217.38
training data = 44.93

The perplexity of the training data is lowest and for the test data is the highest. The perplexity of train data is the best because the model is optimized using the training data and the weights are adjusted using backpropagation
based on the optimization of the loss function. The validation set is used to optimize model parameters. The trained model with saved weights is then
used to evaluate the test data which is used to get an unbiased estimate of the final model performance, hence the perplexity is a bit higher than train and validation data,
 but still is lower enough to say that the model is working well on our test data in this case. The conclusion that can be drawn is that the model is 
 trained and tuned well with low perplexities and gives reasonable perplexity on test data.

(Word Count: 208)

3. How does the Perplexity of your final model on your test data compare to published Neural Language Model perplexity results? 
Find one paper that reports the results of a NLM using perplexity and compare your results to those.  Provide a citation to that paper 
in your response. How does your Perplexity compare to the published result? What kind of model and data did the published result use? 
What do you think accounts for the difference in the perplexity you see?

Paper: RECURRENT NEURAL NETWORK REGULARIZATION- Wojciech Zaremba, Ilya Sutskever, Oriol Vinyals,2015
Link: https://arxiv.org/pdf/1409.2329.pdf

The perplexity of my model on test data : 217.38
The perplexity of the published model on test data : 78.4

The published model uses the LSTM model with two layers that have 1500 units per layer and its parameters are initialized uniformly in
[−0.04, 0.04]. Applying 65% dropout on the non-recurrent connections. Trained the model for 55 epochs with a learning rate of 1;
after 14 epochs they start to reduce the learning rate by a factor of 1.15 after each epoch. Also clipping the norm of the gradients at 10. 
Training this network takes an entire day on an NVIDIA K20 GPU as reported in the paper.

The difference in perplexity in my model and the published model according to me is mainly due to the increased number of hidden dimensions i.e, 1500 per layer, and
applying the dropout accordingly. But increasing the number of hidden dimensions increases the time for training the model drastically. My model took less time and gives the perplexity of 217.38
on test data which is reasonable for 200 hidden units per layer. I think increasing the hidden units and adjusting the dropout accordingly can lower the perplexity of my 
model as well but eventually, it will increase the time taken for training the model. 

(Word Count : 223)


'''

