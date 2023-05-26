import torch
from torch.autograd import Variable
import numpy as np
import torch.nn as nn
from dataset import LM_Dataset, raw_data
from torch.utils.data import DataLoader
import argparse




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
parser.add_argument("--batch_size", type=int, default=1,
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
parser.add_argument('--model', type=str,  default='gru_best_perplexity_modelGRU.pth',
                    help='path to save the final model')
args = parser.parse_args()


#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")

def test(model, data, criterion):

    model.eval()

    costs = 0.0
    iters = 0
    input = []
    output = []

    for batch in data:

        inputs = Variable(batch["source"].transpose(0, 1).contiguous()).to(device)
        input.append(inputs)
        hidden = model.init_hidden()

        outputs, hidden = model(inputs, hidden)
        #print(outputs.size())
        targets = Variable(batch["target"].transpose(0, 1).contiguous()).to(device)
        #print(targets.size())
        pred = torch.argmax(outputs, dim=-1)
        output.append(pred)
        #print(torch.argmax(outputs, dim=-1).size())
        tt = torch.squeeze(targets.view(-1, model.batch_size * model.num_steps))
        #print(tt.size())
        #print(outputs.view(-1, model.vocab_size).size())

        loss = criterion(outputs.view(-1, model.vocab_size), tt)
        costs += loss.data.item() * model.num_steps
        iters += model.num_steps



    return np.exp(costs / iters), input, output


def translate_back(sentences, id2word):
    sentence = sentences.reshape(-1).tolist()
    final_sentence = []
    for index in sentence:
        final_sentence.append(id2word[index])
    return  bpe_postprocess( " ".join(final_sentence))


def bpe_postprocess(string) -> str:
    """
    Post-processor for BPE output. Recombines BPE-split tokens.

    :param string:
    :return: post-processed string
    """
    return string.replace("@@ ", "").replace("@@", "")



if __name__ == "__main__":


    model = torch.load(args.model).to(device)
    train_data, valid_data, test_data, word_to_id, id_2_word = raw_data(args.data)
    model.batch_size=1
    criterion = nn.CrossEntropyLoss()
    test_d = LM_Dataset(test_data, args.num_steps)
    test_dataloader = DataLoader(test_d, batch_size=args.batch_size, drop_last=True)
    ppl, input, output = test(model, test_dataloader, criterion)
    for (s, t) in zip(input, output):
        print(translate_back(s, id_2_word))
        print(translate_back(t, id_2_word))

    print(ppl)

