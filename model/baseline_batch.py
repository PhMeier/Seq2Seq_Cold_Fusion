from __future__ import unicode_literals, print_function, division
from io import open
import unicodedata
import string
import re
import random

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import pickle
from test_query import MyData
import numpy as np
from torch.utils.data import DataLoader
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import random


class Encoder_LSTM(nn.Module):
    def __init__(self, input_size, emb_size, enc_units, batch_size, n_layers=1, drop_prob=0):
        super(Encoder_LSTM, self).__init__()
        self.batch_size = batch_size
        self.input_size = input_size
        self.emb_size = emb_size  
        self.enc_units = enc_units
        self.emb_size = emb_size
        self.n_layers = n_layers
        self.embedding = nn.Embedding(self.input_size, self.emb_size)
        self.lstm = nn.GRU(self.emb_size, self.enc_units, n_layers, dropout=drop_prob, batch_first=True)

    def forward(self, inputs, lens, device):
        # Embed input words
        embedded = self.embedding(inputs)#.view(1,1,-1)
        # Pass the embedded word vectors into LSTM and return all outputs
        # Unpad
        self.hidden = self.init_hidden(device)
        embedded = pack_padded_sequence(embedded, lens)
        output, self.hidden = self.lstm(embedded, self.hidden)
        # pad the sequence to the max length in the batch
        output, _ = pad_packed_sequence(output)
        return output, self.hidden

    def init_hidden(self, device):
        return torch.zeros((1, self.batch_size, self.enc_units)).to(device)


class DecoderRNN(nn.Module):
    def __init__(self, vocab_size, emb_dim, hidden_size, attention, batch_size,
                 enc_uni, n_layers = 1, dropout = 0.1):
        super(DecoderRNN, self).__init__()
        self.batch_size = batch_size
        self.hidden_size = hidden_size
        self.enc_units = enc_uni
        self.dec_units = hidden_size
        self.vocab_size = vocab_size
        self.emb_size = emb_dim
        self.embedding = nn.Embedding(self.vocab_size, self.emb_size)
        self.n_layers = n_layers
        self.drop_prob = dropout

        # for the sake of readiblity and understanding the attention mechanism is separated from the decoder
        # Weight matrices and a vector
        self.W1 = nn.Linear(self.enc_units, self.dec_units)
        self.W2 = nn.Linear(self.enc_units, self.dec_units)
        self.V = nn.Linear(self.enc_units, 1)
        self.dropout = nn.Dropout(self.drop_prob)
        self.fc = nn.Linear(self.enc_units, self.vocab_size) # fully connected layer
        self.gru = nn.GRU(self.emb_size + self.enc_units, self.hidden_size, batch_first = True)
        #self.classifier = nn.Linear(self.enc_units, self.vocab_size)

    def forward(self, x, hidden, enc_output):
        # https://colab.research.google.com/drive/1uFJBO1pgsiFwCGIJwZlhUzaJ2srDbtw-#scrollTo=QiMRxHQFGPtt&forceEdit=true&sandboxMode=true

        # enc_output original: (max_length, batch_size, enc_units)
        # enc_output converted == (batch_size, max_length, hidden_size)
        enc_output = enc_output.permute(1, 0, 2)
        # hidden shape == (batch_size, hidden size)
        # hidden_with_time_axis shape == (batch_size, 1, hidden size)
        # we are doing this to perform addition to calculate the score

        # hidden shape == (batch_size, hidden size)
        # hidden_with_time_axis shape == (batch_size, 1, hidden size)
        hidden_with_time_axis = hidden.permute(1, 0, 2)

        # score: (batch_size, max_length, hidden_size) # Bahdanaus's
        # we get 1 at the last axis because we are applying tanh(FC(EO) + FC(H)) to self.V
        # It doesn't matter which FC we pick for each of the inputs
        # Following Bahdanau attention
        score = torch.tanh(self.W1(enc_output) + self.W2(hidden_with_time_axis))

        # score = torch.tanh(self.W2(hidden_with_time_axis) + self.W1(enc_output))

        # attention_weights shape == (batch_size, max_length, 1)
        # we get 1 at the last axis because we are applying score to self.V
        # apply to self.V following the formula
        attention_weights = torch.softmax(self.V(score), dim=1)

        # context_vector shape after sum == (batch_size, hidden_size)
        context_vector = attention_weights * enc_output
        context_vector = torch.sum(context_vector, dim=1)

        # x shape after passing through embedding == (batch_size, 1, embedding_dim)
        # takes case of the right portion of the model above (illustrated in red)
        x = self.embedding(x)

        # x shape after concatenation == (batch_size, 1, embedding_dim + hidden_size)
        # x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)
        # ? Looks like attention vector in diagram of source
        x = torch.cat((context_vector.unsqueeze(1), x), -1)

        # passing the concatenated vector to the GRU
        # output: (batch_size, 1, hidden_size)
        output, state = self.gru(x)

        # output shape == (batch_size * 1, hidden_size)
        output = output.view(-1, output.size(2))

        # output shape == (batch_size * 1, vocab)
        x = self.fc(output)

        return x, state, attention_weights


class Attention(nn.Module):
    """
    https://blog.floydhub.com/attention-mechanism/
    https://github.com/AotY/Pytorch-NMT/blob/master/src/attention.py

    About batch multiplication:
    https://discuss.pytorch.org/t/understanding-batch-multiplication-using-torch-matmul/16882
    (1000, 500, 100, 10) and (500, 10, 50)
    The matrix multiplication is always done with using the last two dimensions. All the ones before are considered as batch.
    In your case the matrix multiplications will be of size 100x10 and 10x50. The batch dimensions are 1000x500 and 500 and so will be broadcasted to 1000x500. The final output will thus be of size 1000x500x100x50.
    https://stackoverflow.com/questions/50826644/why-do-we-do-batch-matrix-matrix-product
    Legend for math from Luong Paper:
    h^-_t: attentional hidden state aka attentional hidden vector
    h_t: Current target state (decoder)
    h_s: Source hidden state (encoder)

    """
    def __init__(self, hidden_size, method = "dot"):
        super(Attention, self).__init__()
        self.method = method
        self.hidden_size = hidden_size
        # for the dot method, a matrix like W_a is not intended
        if self.method == "general":
            # h^T * W_a*h_s
            self.weight_matrix = nn.Linear(hidden_size, hidden_size, bias =False) # initialise the weight matrix W_a in the paper

        if self.method == "concat":
            # v_a * tan(W_a[h_t;h_s])
            self.weight_matrix = nn.Linear(hidden_size, hidden_size, bias=False)
            self.model_vector = nn.Parameter(torch.FloatTensor(1, hidden_size)) # v_a in the paper

    def forward(self, h_t, h_s): # h_t: decoder_hidden, h_s: encoder_outputs
        """
        Contains the scoring functions, following Luong et al.
        :param h_t:
        :param h_s:
        :return:
        """
        if self.method == "dot":
            return h_s.bmm(h_t) # dot product between decoder hidden state and encoder outputs

        if self.method == "general":
            # in the general method, we pass the decoder hidden values through a linear layer to receive a weight matrix
            weight_and_hidden = self.weight_matrix(h_t)
            return h_s.bmm(weight_and_hidden)

        if self.method == "concat":
            # Concat decoder hidden_states and encoder outputs together
            out = torch.tanh(self.weight_matrix(h_t + h_s))
            return out.bmm(self.model_vector.unsqueeze(-1)).squeeze(-1)


def sort_batch(X, y, lengths):
    lengths, indx = lengths.sort(dim=0, descending=True)
    X = X[indx]
    y = y[indx]
    return X.transpose(0,1), y, lengths # transpose (batch x seq) to (seq x batch)


def loss_function(real, pred):
    """ Only consider non-zero inputs in the loss; mask needed """
    # mask = 1 - np.equal(real, 0) # assign 0 to all above 0 and 1 to all 0s
    # print(mask)
    mask = real.ge(1).type(torch.FloatTensor) #cuda.FloatTensor)

    loss_ = criterion(pred, real) * mask
    return torch.mean(loss_)

if __name__ == "__main__":
    import time
    BATCH_SIZE = 32
    SEED = 42
    #N_BATCH = BUFFER_SIZE // BATCH_SIZE
    embedding_dim = 256
    units = 1024

    with open("german_sm.pickle", "rb") as f:
        ger = pickle.load(f)
    with open("english_sm.pickle", "rb") as f:
        eng = pickle.load(f)
    with open("index2word_en_sm.pickle", "rb") as f:
        id2word_en = pickle.load(f)
    with open("index2word_de_sm.pickle", "rb") as f:
        id2word_de = pickle.load(f)
    ger = np.asarray(ger, dtype=np.int64)
    eng = np.asarray(eng, dtype=np.int64)
    ger = ger[:int(len(ger) * 0.01)]
    eng = eng[:int(len(eng) * 0.01)]

    x = eng[:100] #int(len(eng) * 0.7)]
    y = ger[:100]#int(len(ger) * 0.7)]
    BUFFER_SIZE = len(eng)
    N_BATCH = BUFFER_SIZE // BATCH_SIZE
    train = MyData(eng, ger)
    #BATCH_SIZE = 64
    dataset = DataLoader(train, batch_size=BATCH_SIZE,
                         drop_last=True,
                         shuffle=True)
    #it = iter(dataset)
    #x, y, x_len = next(it)
    """
    print("Input: ", x.shape)
    print("Output: ", y.shape)
    xs, ys, lens = sort_batch(x, y, x_len)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    encoder = Encoder_LSTM(len(id2word_en), embedding_dim, units, BATCH_SIZE)
    enc_output, enc_hidden = encoder(xs.to(device), lens, device)
    #print(enc_output.size())
    #def __init__(self, vocab_size, emb_dim, hidden_size, attention, batch_size, enc_uni, n_layers = 1, dropout = 0.1):

    decoder = DecoderRNN(len(id2word_de), embedding_dim, units, Attention(512), BATCH_SIZE, units)
    decoder = decoder.to(device)
    dec_hidden = enc_hidden
    
    dec_input = torch.tensor([[1]] * BATCH_SIZE) # starting tag
    print("Decoder Input: ", dec_input.shape)
    print("--------")

    for t in range(1, y.size(1)):
        # enc_hidden: 1, batch_size, enc_units
        # output: max_length, batch_size, enc_units
        predictions, dec_hidden, _ = decoder(dec_input.to(device),
                                         dec_hidden.to(device),
                                         enc_output.to(device))
    print("Prediction: ", predictions.shape)
    print("Decoder Hidden: ", dec_hidden.shape)
    """
    criterion = nn.CrossEntropyLoss()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    ## TODO: Combine the encoder and decoder into one class
    encoder = Encoder_LSTM(len(id2word_en), embedding_dim, units, BATCH_SIZE)
    decoder = DecoderRNN(len(id2word_de), embedding_dim, units, Attention(512), BATCH_SIZE, units)

    encoder.to(device)
    decoder.to(device)

    optimizer = optim.Adam(list(encoder.parameters()) + list(decoder.parameters()),
                       lr=0.001)
    EPOCHS = 2


    for epoch in range(EPOCHS):
        print("Epoche: ", epoch)
        start = time.time()

        encoder.train()
        decoder.train()

        total_loss = 0

        for (batch, (inp, targ, inp_len)) in enumerate(dataset):
            loss = 0

            xs, ys, lens = sort_batch(inp, targ, inp_len)
            enc_output, enc_hidden = encoder(xs.to(device), lens, device)
            dec_hidden = enc_hidden

            # use teacher forcing - feeding the target as the next input (via dec_input)
            dec_input = torch.tensor([[1]]* BATCH_SIZE)

            # run code below for every timestep in the ys batch
            for t in range(1, ys.size(1)):
                predictions, dec_hidden, _ = decoder(dec_input.to(device),
                                                 dec_hidden.to(device),
                                                 enc_output.to(device))
                loss += loss_function(ys[:, t].to(device), predictions.to(device))
                # loss += loss_
                dec_input = ys[:, t].unsqueeze(1)
                #del predictions
                #del dec_hidden
                #del _
            
            batch_loss = (loss / int(ys.size(1)))
            total_loss += batch_loss

            optimizer.zero_grad()

           
            loss.backward()
            ### UPDATE MODEL PARAMETERS
            optimizer.step()
            del predictions
            del dec_hidden
            del loss
            del dec_input 
            del enc_output
            del enc_hidden

            #if batch % 100 == 0:
                #print('Epoch {} Batch {} Loss {:.4f}'.format(epoch + 1,
                                                            #batch,
                                                             #batch_loss.detach().item()))

        ### TODO: Save checkpoint for model
        print('Epoch {} Loss {:.4f}'.format(epoch + 1, total_loss / N_BATCH))
        print('Time taken for 1 epoch {} sec\n'.format(time.time() - start))

    torch.save({"encoder":encoder.state_dict(),
                "decoder":decoder.state_dict(),
                "optimizer":optimizer.state_dict(),}
                ,"./model_2_sm.pt")
