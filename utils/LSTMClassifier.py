import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class LSTM_Classifier(nn.Module):
    def __init__(self, vocab_size, label_size, embedding_dim, hidden_dim, batch_size):
        super(LSTM_Classifier, self).__init__()
        self.batch_size = batch_size
        self.hidden_dim = hidden_dim
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim)
        self.hidden2label = nn.Linear(hidden_dim, label_size)
        self.hidden = self.init_hidden()

    def init_hidden(self):
        return (Variable(torch.zeros(1, self.batch_size, self.hidden_dim)),
                Variable(torch.zeros(1, self.batch_size, self.hidden_dim)))

    def forward(self, sentence):
        embeds = self.word_embeddings(sentence)
        x = embeds.view(len(sentence), self.batch_size - 1)
        lstm_out, self.hidden = self.lstm(x, self.hidden)
        y = self.hidden2label(lstm_out[-1])
        return y