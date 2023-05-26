"""

"""

import pickle
import json


def convert_sentences_to_id(sentences, id):
    """
    Replaces the words of sentences with their id. Without SOS
    :param sentences:
    :param id:
    :return: list
    """
    result = []
    for sent in sentences:
        sent = sent.rstrip().split()
        sent.append("</s>")
        x = []
        for token in sent:
            try: #just incase ¾ and ¼
                x.append(id[token])
            except:
                print(token)

        result.append(x)
    return result



if __name__ == "__main__":


    with open('ito.json', encoding="utf-8") as json_file:
        english_vocab = json.load(json_file)


    with open("ltw_eng_sentences.BPE.L1", encoding="utf-8") as f:
        sentences = f.readlines()

    sentences = convert_sentences_to_id(sentences, english_vocab)

    id2word = {index: token for token, index in english_vocab.items()}

    with open("ltw_sentences_final.pickle", "wb") as f:
        pickle.dump(sentences, f)

    with open("../language model/output_monolingual/index2word_ltw.pickle", "wb") as f:
        pickle.dump(id2word, f)

    with open("../language model/output_monolingual/token2index_ltw.pickle", "wb") as f:
        pickle.dump(english_vocab, f)
