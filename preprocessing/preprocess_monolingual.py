"""
Usage: preprocess_monolingual_s.py path_to_folder
"""

from sacremoses import MosesTokenizer
import re
import string
import os
import gzip
from spacy.lang.en import English
import argparse


parser = argparse.ArgumentParser(description="Monolingual Preprocessing")
parser.add_argument('--path', type=str,  default='./ltw_eng',
                    help='path to the gigaword folder to preprocess')
args = parser.parse_args()

def filter_unwanted_characters(data):
    """
    Filter out unwanted characters via regex and lowercase
    :param data: list of sentences
    :return en: list of cleaned sentences
    """
    en = []
    unprintable_reg = re.compile('[^%s]' % re.escape(string.printable))# inverse regex match to delete unprintable tokens

    for e in data:
        # lower the strings
        e = e.lower()
        # replace superscript and subscript numbers
        e = re.sub('[¹²³⁴⁵⁶⁷⁸⁹₁₂₃₄₅₆₇₈₉¼¾]', '', e)
        # remove digits
        e = re.sub('\d+', '', e)
        # remove punctuation
        e = e.translate(str.maketrans("", "", string.punctuation))
        unprintable_reg.sub(e, "")
        en.append(e)
    return en


def filter_sentences(data):
    """
    Filters Sentences from raw data
    :param data:
    :return: sentences: headlines and texts of news articles
    """
    sentences = []
    pattern_start = re.compile("(<P>)|(<HEADLINE>)|(<TEXT>)")
    pattern_end = re.compile("(</P>)|(</HEADLINE>)|(</TEXT)")
    index = 0
    while index < len(data):
        if re.search(pattern_start, data[index]):
            index += 1
            if data[index].startswith("<"):
                index +=1
            if not data[index].startswith("<"):
                sentence = []
                while not re.search(pattern_end, data[index]):
                    sentence.append(data[index].rstrip())
                    index+=1
                sentences.append(" ".join(sentence))
        index+= 1
    return sentences

def split_sentences(data):
    """
    :param data:
    :return:
    """
    nlp = English()
    sentencizer = nlp.create_pipe("sentencizer")
    nlp.add_pipe(sentencizer)
    content = []
    for sentence in data:
        remove_abbreviations(sentence)
        doc = nlp(sentence)
        for sen in doc.sents:
            content.append(sen.text.strip())
    return content


def remove_abbreviations(line):
    """
    :param line:
    :return:
    """

    line = re.sub("(Dr\.)", "", line)
    line = re.sub("(Mr\.)", "", line)
    line = re.sub("(Mrs\.)", "", line)
    line = re.sub("(Jr\.)", "", line)
    line = line.strip()

    return line



def tokenize(sentence_list, language_tag):
    """
    Tokenizes given sentences in a list.
    :param sentence_list:
    :param language_tag:
    :return: list
    """
    tokenized_sents = []
    mt = MosesTokenizer(lang = language_tag)

    for sent in sentence_list:
        tok_text = mt.tokenize(sent, return_str=False)
        if len(tok_text) < 60:
            tokenized_sents.append(tok_text)

    return tokenized_sents


def filter_vocab(data, english_vocab):
    """
    :param data:
    :return:
    """
    filtered_data = []
    for sentence in data:
        tokens = [word in english_vocab for word in sentence]
        if all(tokens):
            filtered_data.append(sentence)
    return filtered_data

def add_to_vocab(tokens, vocab):
    """
    :param data:
    :param vocab:
    :return:
    """
    for token in tokens:
        if token not in vocab:
            vocab.append(token)
    return vocab


def open_folder(subfold):
    data = []
    for file in os.listdir(subfold):
        with gzip.open(os.path.join(subfold, file), "rt") as f:
            data.extend(f.readlines())
    return data

def open_vocab(file):

    with open(file) as f:
        data = f.readlines()
    english_vocab = [line.split()[0] for line in data]
    return english_vocab

def convert_sentences_to_id(sentences, id):
    """
    Converts replaces the words of sentences with their id. Without SOS
    :param sentences:
    :param id:
    :return: list
    """
    result = []
    for sent in sentences:
        sent.append("<EOS>")
        x = []
        for token in sent:
            x.append(id[token])
        result.append(x)
    return result



if __name__ == "__main__":

    subfold = args.path
    data = open_folder(subfold)
    subfold_name = subfold.split("/")[-1]


    en = filter_sentences(data)
    en = split_sentences(en)
    en = filter_unwanted_characters(en)

    en_sentences = tokenize(en, "en")
    with open( subfold_name + "_sentences_preprocessed.txt", "w") as f:
        for line in en_sentences:
            if line:
                f.write(" ".join(line) + "\n")



