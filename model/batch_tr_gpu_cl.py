from baseline_batch import Encoder_LSTM, DecoderRNN, Attention
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import pickle
from dataset_wrapper import ParallelData
import numpy as np
from torch.utils.data import DataLoader
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import time

def sort_batch(X, y, lengths):
    lengths, indx = lengths.sort(dim=0, descending=True)
    X = X[indx]
    y = y[indx]
    return X.transpose(0,1), y, lengths # transpose (batch x seq) to (seq x batch)


def loss_function(real, pred):
    """ Only consider non-zero inputs in the loss; mask needed """
    # mask = 1 - np.equal(real, 0) # assign 0 to all above 0 and 1 to all 0s
    # print(mask)
    mask = real.ge(1).type(torch.cuda.FloatTensor)

    loss_ = criterion(pred, real) * mask
    return torch.mean(loss_)



if __name__ == "__main__":
    #import time
    BATCH_SIZE = 16
    SEED = 42
    #N_BATCH = BUFFER_SIZE // BATCH_SIZE
    embedding_dim = 512
    units = 1024

    with open("german_sm.pickle", "rb") as f:
        ger = pickle.load(f)
    with open("english_sm.pickle", "rb") as f:
        eng = pickle.load(f)
    with open("index2word_en_sm.pickle", "rb") as f:
        id2word_en = pickle.load(f)
    with open("index2word_de_sm.pickle", "rb") as f:
        id2word_de = pickle.load(f)
    print("len de ", len(id2word_de))
    print("len en ", len(id2word_en))

    
    abc = sorted(list(id2word_en.keys()), reverse=True)

    ger = np.asarray(ger, dtype=np.int64)
    eng = np.asarray(eng, dtype=np.int64)
    ger = ger[:int(len(ger) * 0.7)]
    eng = eng[:int(len(eng) * 0.7)]
    print("len ger", len(ger))
    #print(id2word_en[0])

    x = eng[:int(len(eng) * 0.7)]
    y = ger[:int(len(ger) * 0.7)]
    BUFFER_SIZE = len(eng)
    N_BATCH = BUFFER_SIZE//BATCH_SIZE
    train = ParallelData(eng, ger)
    #BATCH_SIZE = 64
    #dataset = DataLoader(train, batch_size=BATCH_SIZE,
    #                     drop_last=True,
    #                     shuffle=True)

    criterion = nn.CrossEntropyLoss()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    ## TODO: Combine the encoder and decoder into one class
    encoder = Encoder_LSTM(len(id2word_en), embedding_dim, units, BATCH_SIZE)
    decoder = DecoderRNN(len(id2word_de), embedding_dim, units, Attention(512), BATCH_SIZE, units)

    encoder.to(device)
    decoder.to(device)

    optimizer = optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=0.0001)
    EPOCHS = 20
    params = {'batch_size': BATCH_SIZE,
              'shuffle': True,
              'num_workers': 2,
              'drop_last' : True}
    training_generator = DataLoader(train, **params)
    print(training_generator)
    for epoch in range(EPOCHS):
        start = time.time()
        #print("Epoch: ", epoch)
        encoder.train()
        decoder.train()
        total_loss = 0
        for batch, targ, len_ in training_generator:
            #print(batch, targ, len_)
            loss = 0
            xs, ys, lens = sort_batch(batch, targ, len_)
            enc_output, enc_hidden = encoder(xs.to(device), lens, device)
            dec_hidden = enc_hidden
            dec_input = torch.tensor([[1]] * BATCH_SIZE)
            #print(ys.size(1))
            for t in range(1, ys.size(1)):
                #print("t ", t)
                predictions, dec_hidden, _ = decoder(dec_input.to(device),
                                                     dec_hidden.to(device),
                                                     enc_output.to(device))
                loss += loss_function(ys[:, t].to(device), predictions.to(device))
                # loss += loss_
                dec_input = ys[:, t].unsqueeze(1)

            batch_loss = (loss / int(ys.size(1)))
            total_loss += batch_loss
            #print("tot_loss: ", total_loss, "loss: ", loss) #, "batch loss: ", batch_loss)

            optimizer.zero_grad()

            loss.backward()
            optimizer.step()
            nn.utils.clip_grad_norm_(encoder.parameters(), 0.25) # with _ inplace operation
            nn.utils.clip_grad_norm_(decoder.parameters(), 0.25)
            del predictions
            del dec_hidden
            del loss
            del dec_input
            del enc_output
            del enc_hidden

            #print(batch, targ, len_)
            #local_batch, local_target = local_batch.to(device), local_target.to(device)
        print('Epoch {} Loss {:.4f}'.format(epoch + 1, total_loss/N_BATCH))
        print('Time taken for 1 epoch {} sec\n'.format(time.time()-start))

torch.save({"encoder":encoder.state_dict(), "decoder":decoder.state_dict(), "optimizer":optimizer.state_dict(),},"./model_20_lr_and_025cl.pt")
