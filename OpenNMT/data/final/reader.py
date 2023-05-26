import torch
import onmt
from onmt.inputters.text_dataset import TextMultiField


x = torch.load("europarl.vocab.pt")
print(x)
data = x["tgt"]
print(data.fields[0][1].vocab_cls.stoi)
