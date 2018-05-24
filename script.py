import re
import unicodedata
import torch
import torch.nn as nn
import torch.autograd as autograd
import torch.nn.functional as F
import torch.optim as optim
import pickle


class WordIndex():
    def __init__(self):
        self.count = 0
        self.word_to_idx = {}
        self.word_count = {}

    def add_word(self, word):
        if word not in self.word_to_idx:
            self.word_to_idx[word] = self.count
            self.word_count[word] = 1
            self.count += 1
        else:
            self.word_count[word] += 1

    def add_text(self, text):
        for word in text.split(' '):
            self.add_word(word)


def normalize_string(s):
    s = s.lower().strip()
    s = re.sub(r'(\W)(?=1)', '', s)
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    return s


def limit_dict(limit, class_obj):
    dict = sorted(class_obj.word_count.items(), key=lambda t: t[1], reverse=True)
    count = 0
    for x, y in dict:
        if count >= limit - 1:
            class_obj.word_to_idx[x] = limit
        else:
            class_obj.word_to_idx[x] = count

        count += 1


VOCAB_LIMIT = 50000
MAX_SEQUENCE_LEN = 500
ob = WordIndex()

f = open('data/labeledTrainData.tsv').readlines()

print('Reading the lines ....')

for idx, lines in enumerate(f):
    if idx != 0:
        data = lines.split('\t')[2]
        data = normalize_string(data).strip()
        ob.add_text(data)

print('Read Done')

limit_dict(VOCAB_LIMIT, ob)


class Model(torch.nn.Module):
    def __init__(self, embedding_dim, hidden_dim):
        super(Model, self).__init__()
        self.hidden_dim = hidden_dim
        self.embeddings = nn.Embedding(VOCAB_LIMIT + 1, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim)
        self.linearOut = nn.Linear(hidden_dim, 2)

    def forward(self, inputs, hidden):
        x = self.embeddings(inputs).view(len(inputs), 1, -1)
        lstm_out, lstm_h = self.lstm(x, hidden)
        x = lstm_out[-1]
        x = self.linearOut(x)
        x = F.log_softmax(x)
        return x, lstm_h

    def init_hidden(self):
        return (autograd.Variable(torch.zeros(1, 1, self.hidden_dim)),
                autograd.Variable(torch.zeros(1, 1, self.hidden_dim)))


model = Model(50, 100)

loss_function = nn.NLLLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

epochs = 4

torch.save(model.state_dict(), 'model' + str(0) + '.pth')
print('Training...')

for i in range(epochs):
    avg_loss = 0.0
    for idx, lines in enumerate(f):
        if idx != 0:
            data = lines.split('\t')[2]
            data = normalize_string(data).strip()
            input_data = [ob.word_to_idx[word] for word in data.split(' ')]
            if len(input_data) > MAX_SEQUENCE_LEN:
                nput_data = input_data[0:MAX_SEQUENCE_LEN]

            input_data = autograd.Variable(torch.LongTensor(input_data))
            target = int(lines.split('\t')[1])
            target_data = autograd.Variable(torch.LongTensor([target]))
            hidden = model.init_hidden()
            y_pred, _ = model(input_data, hidden)
            model.zero_grad()
            loss = loss_function(y_pred, target_data)
            avg_loss += loss.data[0]

            if idx%500 == 0 or idx == 1:
                print('epoch :%d iterations :%d loss :%g' % (i, idx, tensor.item(loss.data[0])))

            loss.backward()
            optimizer.step()
    torch.save(model.state_dict(), 'model' + str(i + 1) + '.pth')
    print('the average loss after completion of %d epochs is %g' % ((i + 1), (avg_loss / len(f))))

with open('dict.pkl', 'wb') as f:
    pickle.dump(ob.word_to_idx, f)