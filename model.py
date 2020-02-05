import torch as t
from torch import nn
from torch.autograd import Variable
import os 
import glob

class PoetryModel(nn.Module):

    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super(PoetryModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, 2, batch_first=True)
        self.linear = nn.Linear(self.hidden_dim, vocab_size)

    def forward(self, input_, hidden=None):
        batch_size, seq_len = input_.size()
        if hidden is None:
            h_0 = input_.data.new(2, batch_size, self.hidden_dim).fill_(0).float()
            c_0 = input_.data.new(2, batch_size, self.hidden_dim).fill_(0).float()
            h_0, c_0 = Variable(h_0), Variable(c_0)

        else:
            h_0, c_0 = hidden

        embeds = self.embeddings(input_)
        output, hidden = self.lstm(embeds, (h_0, c_0))
        output = self.linear(output.reshape(seq_len * batch_size, -1))
        return output, hidden

    

        