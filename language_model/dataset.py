import os
import pickle
import numpy as np
from torch.utils.data import Dataset
import torchtext
import torch

def raw_data(data_path):
    with open(os.path.join(data_path, "ltw_sentences_final_2m.pickle"), "rb")as f:
        data = pickle.load(f)
    with open(os.path.join(data_path, "index2word_ltw.pickle"), "rb") as f:
        id2word = pickle.load(f)
    with open(os.path.join(data_path, "token2index_ltw.pickle"), "rb") as f:
        token2id = pickle.load(f)
    len_valid = 3000
    train = data[:int(len(data) -len_valid*2)]
    valid = data[int(len(data) -len_valid*2):int(len(data)-len_valid)]
    test  = data[int(len(data)-len_valid):]
    return train, valid, test, token2id, id2word


class LM_Dataset(Dataset):
    def __init__(self, data, len_sequence):

        self.sources = []
        self.targets = []

        raw_data = [item for sublist in data for item in sublist]
        count_sequences = len(raw_data) // len_sequence

        for i in range(count_sequences):
            self.sources.append(raw_data[i*len_sequence:(i+1)*len_sequence])
            self.targets.append(raw_data[i*len_sequence+1:(i+1)*len_sequence+1])

    def __len__(self):
        return len(self.sources)

    def __getitem__(self, item):
        datapoint = {"source": np.array(self.sources[item], dtype=np.int64),
                     "target": np.array(self.targets[item], dtype=np.int64)}
        return datapoint

class Vocabulary:
    def __init__(self, itos, stoi, specials):

        self.specials = specials
        self.stoi = stoi
        self.itos = itos

def europarl_dataset(path, vocab_path = "../OpenNMT/data/final/europarl.vocab.pt", train= True, batch_size=64):
    vocab_opennmt= torch.load(vocab_path)
    itos = vocab_opennmt["tgt"].base_field.vocab.itos
    stoi = vocab_opennmt["tgt"].base_field.vocab.stoi
    specials = itos[0:4]
    vocab = Vocabulary(itos, stoi, specials)
    tok_fun = lambda s:  s.split()
    text_field = torchtext.data.Field(init_token=specials[2], eos_token=specials[3],
                           pad_token=specials[1], tokenize=tok_fun,
                           batch_first=True,
                           unk_token=specials[0])
    text_field.vocab = vocab
    train_data = torchtext.datasets.LanguageModelingDataset(path=path +".en",
                                    text_field=text_field, newline_eos=False, encoding="utf-8")

    iterator = torchtext.data.BPTTIterator(train_data, batch_size, 20)
    return iterator


def europarl_translation_dataset(path, vocab_path = "../OpenNMT/data/final/europarl.vocab.pt", train= True, batch_size=64):
    vocab_opennmt = torch.load(vocab_path)
    itos = vocab_opennmt["tgt"].base_field.vocab.itos
    stoi = vocab_opennmt["tgt"].base_field.vocab.stoi
    specials = itos[0:4]
    vocab = Vocabulary(itos, stoi, specials)
    tok_fun = lambda s: s.split()

    src_field = torchtext.data.Field(init_token=specials[2], eos_token=None,
                                     pad_token=specials[1], tokenize=tok_fun,
                                     batch_first=True,
                                     unk_token=specials[0], include_lengths=True)
    src_field.vocab = vocab
    trg_field = torchtext.data.Field(init_token=None, eos_token=specials[3],
                                     pad_token=specials[1], tokenize=tok_fun,
                                     batch_first=True,
                                     unk_token=specials[0], include_lengths=True)
    trg_field.vocab = vocab

    train_data = torchtext.datasets.TranslationDataset(path=path,
                                                       exts=(".en", ".en" ),
                                                       fields=(src_field, trg_field))
    train_data_iterator = torchtext.data.BucketIterator(train_data, batch_size=batch_size, sort_key=lambda x: len(x.src),
                                                        repeat=False,
                                                        sort=False,
                                                        train=train, sort_within_batch=train, shuffle=train)
    return train_data_iterator


def create_eng_file(path):
    data = torch.load(path)
    sentences = []
    for index in range(len(data)):
        sentences.append(data.examples[index].tgt[0])
    with open("europarl_monolingual_test.en", "w", encoding="utf-8") as f:
        for line in sentences:
            f.write(" ".join(line) + "\n")

def translate_back(sentences, id2word):
    sentences = sentences.tolist()
    final_sentences = []
    for sentence in sentences:
        final_sentence = []
        for index in sentence:
            final_sentence.append(id2word[index])
        final_sentences.append(bpe_postprocess( " ".join(final_sentence)))
    return  final_sentences


def bpe_postprocess(string) -> str:
    """
    Post-processor for BPE output. Recombines BPE-split tokens.

    :param string:
    :return: post-processed string
    """
    return string.replace("@@ ", "").replace("@@", "")


def create_eng_file_bptt(path):
    data = torch.load(path)
    sentences = []
    for index in range(len(data)):
        sentences.append(data.examples[index].tgt[0])
    with open("europarl_monolingual_test_bptt.en", "w", encoding="utf-8") as f:
        for line in sentences:
            f.write(" ".join(line) + " </s>\n")

if __name__ == "__main__":
    create_eng_file("../OpenNMT/data/final/europarl.test.0.pt")
    create_eng_file_bptt("../OpenNMT/data/final/europarl.test.0.pt")



