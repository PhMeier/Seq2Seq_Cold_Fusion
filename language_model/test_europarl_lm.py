import torch
from torch.autograd import Variable
import numpy as np
import torch.nn as nn
from dataset import LM_Dataset, raw_data, europarl_translation_dataset, europarl_dataset
from torch.utils.data import DataLoader
import argparse




parser = argparse.ArgumentParser(description="Language Model")
parser.add_argument("--data", type=str, default="./", #preprocessing
                    help="data folder")
parser.add_argument("--batch_size", type=int, default=1,
                    help="batch size")
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
parser.add_argument('--model', type=str,  default='lm_model_europarl_bptt_amsgradbest_perplexity_model_GRU.pth',
                    help='path to save the final model')
parser.add_argument("--bptt", type=bool, default=False)
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


        if batch.batch_size != data.batch_size:
            continue

        if hasattr(batch, "text"):
            inputs = Variable(batch.text.transpose(0, 1).contiguous()).to(device)
            targets = Variable(batch.target.transpose(0, 1).contiguous()).to(device)
        else:
            inputs = Variable(batch.src[0].transpose(0, 1).contiguous()).to(device)
            targets = Variable(batch.trg[0].transpose(0, 1).contiguous()).to(device)
        input.append(inputs)
        hidden = model.init_hidden(data.batch_size)

        outputs, hidden = model(inputs, hidden)
        pred = torch.argmax(outputs, dim=-1)
        output.append(pred)
        tt = torch.squeeze(targets.view(-1, data.batch_size * inputs.size(0)))

        loss = criterion(outputs.view(-1, model.vocab_size), tt)
        costs += loss.data.item() * inputs.size(0)
        iters += inputs.size(0)



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
    if args.bptt:
        test_data_it = europarl_dataset(args.data + "europarl_monolingual_test_bptt", batch_size=1)
        vocab_size = len(test_data_it.dataset.fields["text"].vocab.itos)
        stoi = test_data_it.dataset.fields["text"].vocab.stoi
    else:
        test_data_it = europarl_translation_dataset(args.data + "europarl_monolingual_test", train=False, batch_size=1)
        vocab_size = len(test_data_it.dataset.fields["src"].vocab.itos)
        stoi = test_data_it.dataset.fields["src"].vocab.stoi
    criterion = nn.CrossEntropyLoss()

    id_2_word = {stoi[key]:key for key in stoi.keys()}


    #dataset with batch size
    ppl, input, output = test(model, test_data_it, criterion)
    for (s, t) in zip(input, output):
        print(translate_back(s, id_2_word))
        print(translate_back(t, id_2_word))

    print(ppl)
