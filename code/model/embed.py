import torch
import torch.nn as nn
import math
from torch.autograd import Variable

class Embedder(nn.Module):
    def __init__(self, vocab_size, d_model):
        super().__init__()
        self.d_model = d_model
        #Internally, nn.Embedding is – like a linear layer – a M x N matrix, with M being the number of words and N being the size of each word vector. There’s nothing more to it. It just matches a word (specified by an index) to the corresponding word vector, i.e., the corresponding row in the matrix.
        self.embed = nn.Embedding(vocab_size, d_model)
    def forward(self, x):
        return self.embed(x)