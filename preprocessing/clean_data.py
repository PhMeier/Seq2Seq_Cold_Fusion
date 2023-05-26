import re
import string
from sklearn.utils import shuffle

def filter_unwanted_characters(parallel_data):
    """
    Filter out unwanted charachters via regex and make everything lowercase
    :param parallel_data:
    :return en, de:
    """
    de = []
    en = []
    unprintable_reg = re.compile('[^%s]' % re.escape(string.printable))# inverse regex match to delete unprintable tokens

    for e, d in parallel_data:
        # lower the strings
        d = d.lower()
        e = e.lower()
        # replace superscript and subscript numbers
        d = re.sub('[¹²³⁴⁵⁶⁷⁸⁹₁₂₃₄₅₆₇₈₉]', '', d)
        e = re.sub('[¹²³⁴⁵⁶⁷⁸⁹₁₂₃₄₅₆₇₈₉]', '', e)
        # remove digits
        #d = re.sub('\d+', '', d)
        #e = re.sub('\d+', '', e)
        # Remove brackets
        d = re.sub('[()\[\]{}]', '', d)
        e = re.sub('[()\[\]{}]', '', e)
        # remove punctuation
        #d = d.translate(str.maketrans("", "", string.punctuation))
        #e = e.translate(str.maketrans("", "", string.punctuation))
        #unprintable_reg.sub(d, "")
        #unprintable_reg.sub(e, "")
        de.append(d.lstrip())
        en.append(e.lstrip())
    return en, de


def get_index_of_empty_lines(data):
    indexes = []
    for i in range(len(data)-1):
        if data[i] == "\n" or len(data[i].split()) <= 1:
            indexes.append(i)
    return indexes


def remove_empty_lines(data, index):
    new_content = []
    for i in range(len(data)-1):
            if i not in index:
                new_content.append(data[i])
    return new_content

def filter_length(parallel_data, max_length):
    """
    Filters out instances which are longer than a defined length and have a large mismatch between their lengths.
    :param parallel_data:
    :param max_length:
    :return: list (parallel_data)
    """
    final = []
    x = []
    y = []
    for i,j in parallel_data:
        len_i = len(i.split())
        len_j = len(j.split())
        if len_i <= max_length and len_j <= max_length:
            longer_sen = max([len_i,len_j])
            smaller_sen = min([len_i,len_j])
            if (longer_sen - smaller_sen) < 15:
                #print(len(i), len(j))
                final.append((i,j))
                x.append(i)
                y.append(j)
    return final, x, y


if __name__ == "__main__":
    de = []
    en = []
    max_len = 60
    with open("europarl-v7.de-en.de", encoding = "utf-8") as f:
        de = f.readlines()
    with open("europarl-v7.de-en.en", encoding = "utf-8") as f:
        en = f.readlines()
    #print(en[21] == "\n")
    print(len(en))
    print(len(de))

    en, de = filter_unwanted_characters(list(zip(en, de)))
    _, en ,de = filter_length(list(zip(en, de)), max_len)
    eng_index = get_index_of_empty_lines(en)
    ger_index = get_index_of_empty_lines(de)
    #print(ger_index)
    indexes = ger_index + eng_index
    indexes.sort()
    de = remove_empty_lines(de, indexes)
    en = remove_empty_lines(en, indexes)
    print(len(de), len(en))
    #print(len(de), len(en))
    en, de = shuffle(en, de)
    #"""
    with open("europarl_60.de", "w+", encoding = "utf-8") as f:
        for line in de:
            f.write(line)
    with open("europarl_60.en", "w+", encoding = "utf-8") as f:
        for line in en:
            f.write(line)
    #"""
