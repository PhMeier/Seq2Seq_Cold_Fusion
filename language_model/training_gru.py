"""
https://github.com/ceshine/examples/blob/master/word_language_model/main.py
https://github.com/deeplearningathome/pytorch-language-model"""

import argparse
import torch
import torch.nn
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
from model import LM, repackage_hidden
from dataset import LM_Dataset, raw_data
import numpy as np
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#device = torch.device("cpu" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser(description="Language Model")
parser.add_argument("--data", type=str, default="./output_monolingual", #preprocessing
                    help="data folder")
parser.add_argument("--embed_size", type=int, default=256,
                    help="size of word embeddings")
parser.add_argument("--hidden_size", type=int, default=512,
                    help="hidden size")
parser.add_argument('--num_steps', type=int, default=20,
                    help='number of RNN steps')
parser.add_argument("--num_layers", type=int, default=3,
                    help="number of RNN layers")
parser.add_argument("--batch_size", type=int, default=32,
                    help="batch size")
parser.add_argument("--num_epochs", type=int, default=15,
                    help="number of epochs")
parser.add_argument("--dp_prob", type=float, default=0.1,
                    help="dropout probability")
parser.add_argument('--optimizer', type=str, default="SGD",
                    help="optimizer")
parser.add_argument("--rnn_type", type=str, default="LSTM",
                    help="type of RNN")
parser.add_argument('--initial_lr', type=float, default=0.5,
                    help='initial learning rate')
parser.add_argument("--test", type=bool, default=False,
                    help="test with smaller file")
parser.add_argument('--save', type=str,  default='lm_model_gru.pt',
                    help='path to save the final model')
args = parser.parse_args()


def run_epoch(model, optimizer, data, criterion, is_train=False, lr = 1.0):

    if is_train:
        model.train()
    else:
        model.eval()

    hidden = model.init_hidden()
    costs = 0.0
    iters = 0

    for batch in data:

        inputs = Variable(batch["source"].transpose(0, 1).contiguous()).to(device)
        optimizer.zero_grad()

        hidden = repackage_hidden(hidden)
        outputs, hidden = model(inputs, hidden)

        targets = Variable(batch["target"].transpose(0, 1).contiguous()).to(device)

        tt = torch.squeeze(targets.view(-1, model.batch_size * model.num_steps))

        loss = criterion(outputs.view(-1, model.vocab_size), tt)
        costs += loss.data.item() * model.num_steps
        iters += model.num_steps


        if is_train:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.25)
            optimizer.step()
            for g in optimizer.param_groups:
                g['lr'] = lr

    return np.exp(costs / iters)



if __name__ == "__main__":
    train_data, valid_data, test_data, word_to_id, id_2_word = raw_data(args.data)
    train = LM_Dataset(train_data, args.num_steps)
    train_dataloader = DataLoader(train, batch_size=args.batch_size, drop_last=True)
    valid = LM_Dataset(valid_data, args.num_steps)
    valid_dataloader = DataLoader(valid, batch_size=args.batch_size, drop_last=True)

    vocab_size = len(word_to_id)

    criterion = nn.CrossEntropyLoss()

    model = LM(embed_size=args.embed_size, num_steps=args.num_steps, batch_size=args.batch_size,
               vocab_size=vocab_size, layers=args.num_layers, dropout=1 - args.dp_prob, type=args.rnn_type,
               hidden_size=args.hidden_size).to(device)

    if args.optimizer == "SGD":
        optimizer = optim.SGD(model.parameters(), lr=args.initial_lr)
    else:
        optimizer = optim.Adam(model.parameters(), lr=args.initial_lr)
    model.to(device)

    scheduler = StepLR(optimizer, step_size=30, gamma=0.1)

    lr = args.initial_lr

    best_perplexity = 10e100

    print("########## Training ##########################")
    for epoch in range(args.num_epochs):
        train_p = run_epoch(model, optimizer, train_dataloader, criterion, True, lr=lr)
        valid_p = run_epoch(model, optimizer, valid_dataloader, criterion)

        if valid_p < best_perplexity:
            best_perplexity = valid_p
            checkpoint = {
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'accuracy': valid_p,
                'optimizer': optimizer.state_dict()}
            with open("gru_best_perplexity_checkpoint" + args.rnn_type + ".pth", "wb") as f:
                torch.save(checkpoint, f)
            with open("gru_best_perplexity_model" + args.rnn_type + ".pth", "wb") as f:
                torch.save(model, f)
        scheduler.step()
        print('Train perplexity at epoch {}: {:8.2f}'.format(epoch, train_p))
        print('Validation perplexity at epoch {}: {:8.2f}'.format(epoch, valid_p))
    print("########## Testing ##########################")
    test = LM_Dataset(test_data, args.num_steps)
    test_dataloader = DataLoader(test, batch_size=args.batch_size, drop_last=True)
    # model.batch_size = 1  # to make sure we process all the data
    print('Test Perplexity: {:8.2f}'.format(run_epoch(model, optimizer, test_dataloader, criterion)))
    with open(args.save, 'wb') as f:
        torch.save(model, f)
    print("########## Done! ##########################")
