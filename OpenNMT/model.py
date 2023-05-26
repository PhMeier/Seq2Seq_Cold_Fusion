"""
https://github.com/ceshine/examples/blob/master/word_language_model/main.py
https://github.com/deeplearningathome/pytorch-language-model"""

import torch.nn as nn
from torch import Tensor
import torch
from torch.autograd import Variable

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#device = torch.device("cpu" if torch.cuda.is_available() else "cpu")

class LM(nn.Module):

    def __init__(self, layers, type, num_steps,hidden_size, embed_size, vocab_size, dropout, batch_size):
        super(LM, self).__init__()
        self.layers = layers # three for gru
        self.type = type #either lstm or gru
        self.num_steps = num_steps
        self.hidden_size = hidden_size #either 1024 or 512
        self.embed_size = embed_size #256 for lstm
        self.dropout = nn.Dropout(dropout) #???
        self.embedding = nn.Embedding(vocab_size, self.embed_size)
        self.batch_size = batch_size
        self.vocab_size = vocab_size
        if self.type == "LSTM":
            self.rnn = nn.LSTM(input_size=self.embed_size, hidden_size=self.hidden_size, num_layers=self.layers, dropout=dropout)
        elif self.type == "GRU":
            self.rnn = nn.GRU(input_size = self.embed_size, hidden_size= self.hidden_size, num_layers=self.layers, dropout=dropout)

        self.fc = nn.Linear(in_features= hidden_size, out_features = vocab_size)
        self.init_weights()


    def forward(self, input: Tensor, hidden):
        #input size: num_seps, batch_size
        emb = self.dropout(self.embedding(input))# num_steps, batch_size, embedding size

        out, hidden = self.rnn(emb, hidden)# num_steps, batch_size, hidden_size

        out = self.dropout(out)# num_steps, batch_size, hidden_size

        output = self.fc(out.view(-1, out.size(2))) # num_steps, batch_size, hidden_size #dimensions
        # num_steps, batch_size, vocab_size
        return output.view(self.num_steps, self.batch_size, self.vocab_size), hidden

    def init_weights(self):
        init_range = 0.1
        self.embedding.weight.data.uniform_(-init_range, init_range)
        self.fc.bias.data.fill_(0.0)
        self.fc.weight.data.uniform_(-init_range, init_range)

    def init_hidden(self):
        weight = next(self.parameters()).data
        if self.type == "LSTM":
            return (Variable(weight.new(self.layers, self.batch_size, self.hidden_size).zero_()).to(device),
                    Variable(weight.new(self.layers, self.batch_size, self.hidden_size).zero_()).to(device))
        else:
            return Variable(weight.new(self.layers, self.batch_size, self.hidden_size).zero_().to(device))



def repackage_hidden(h):
    """Wraps hidden states in new Tensors to detach them from their history."""
    if isinstance(h, torch.Tensor):
        return h.detach()
    else:
        return tuple(repackage_hidden(v) for v in h)


