import torch

vocab = torch.load("europarl.vocab.pt")
itos_english = vocab["tgt"].base_field.vocab.itos

itos_german = vocab["src"].base_field.vocab.itos

"""with open("english_final_vocab.txt", "w", encoding="utf-8") as f:
    for word in itos_english:
        f.write(word + "\n")"""

"""with open("german_final_vocab.txt", "w", encoding="utf-8") as f:
    for word in itos_german:
        f.write(word + "\n")"""

vocab = torch.load("europarl_test.vocab.pt")
itos_test = vocab["tgt"].base_field.vocab.itos

assert itos_english == itos_test


sentences = torch.load("europarl.test.0.pt")
a = 1