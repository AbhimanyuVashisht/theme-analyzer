import numpy as np
import pandas as pd


doc = pd.read_csv("/Users/jessicasethi/Desktop/trainingandtestdata/training.csv"
                  , encoding='latin-1')


goog = pd.read_csv("http://s3.amazonaws.com/assets.datacamp.com/production/course_935/datasets/500_goog.csv")
arr = np.array(goog)
doc = np.concatenate((np.ones((501,1)), arr[:,2].reshape(501,1)), axis = 1)
noc = np.concatenate((np.zeros((501,1)), arr[:,3].reshape(501,1)), axis = 1)
doc = np.concatenate((doc, noc), axis = 0)
np.random.shuffle(doc)

raw_corpus = doc[:,1].copy()
for i in range(raw_corpus.shape[0]):
    if (pd.isnull(raw_corpus[i]) != True):
        raw_corpus[i] = tknzr.tokenize(raw_corpus[i])



df = pd.read_csv('/Users/jessicasethi/Documents/amazon_review_full_csv/train.csv')
arr = np.array(df)
arr = arr[0:1500,np.array([0,2])]

temp = arr[:,0].copy()
arr[:,0] = arr[:,1]
arr[:,1] = temp
df = arr
df[df[:,1]<4,1] = 0
df[df[:,1]>3,1] = 1

# preprocessing
raw_corpus = df[:,0].copy()
raw_corpus = list(raw_corpus)
from gensim.parsing.preprocessing import remove_stopwords
raw_corpus = [remove_stopwords(str(sublist)) for sublist in raw_corpus]
from gensim.parsing.preprocessing import strip_punctuation
raw_corpus = [strip_punctuation(str(sublist)) for sublist in raw_corpus]
from nltk.stem import PorterStemmer
ps = PorterStemmer()
raw_corpus = [ps.stem(str(sublist)) for sublist in raw_corpus]

from nltk.tokenize import TweetTokenizer
tknzr = TweetTokenizer()
texts = [tknzr.tokenize(word) for word in raw_corpus]

#split
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(np.array(texts), df[:,1])

#build word2vec model
import gensim
from gensim.models.word2vec import Word2Vec
model = Word2Vec(x_train, size=150, window=10, min_count=1, workers=10)
model.train(x_train, total_examples=len(x_train), epochs=10)

#sentiment classifier
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(analyzer=lambda x: x, min_df=1)
matrix = vectorizer.fit_transform(x_train)
tfidf = dict(zip(vectorizer.get_feature_names(), vectorizer.idf_))
text_list = [item for sublist in x_train for item in sublist]

size = 150
def buildvector(tokens, size):
    vec = np.zeros((1, size))
    count=0
    for i in range(len(tokens)):
        if (tokens[i] in model.wv.vocab):
            vec += model.wv[tokens[i]].reshape((1,size)) * tfidf[tokens[i]]
            count += 1
            if(count!=0):
                vec /= count
    return vec

from sklearn.preprocessing import scale
train_vec = np.concatenate([buildvector(sublist, size) for sublist in x_train])
train_vec = scale(train_vec)
test_vec = np.concatenate([buildvector(sublist, size) for sublist in x_test])
test_vec = scale(test_vec)

import torch
from torch.autograd import Variable
import torch.nn.functional as F

class Net(torch.nn.Module):
    def __init__(self, n_feature, n_hidden1, n_hidden2, n_output):
        super(Net, self).__init__()
        self.hidden1 = torch.nn.Linear(n_feature, n_hidden1)   # hidden layer
        self.hidden2 = torch.nn.Linear(n_hidden1, n_hidden2)
        self.out = torch.nn.Linear(n_hidden2, n_output)   # output layer

    def forward(self, x):
        x = F.relu(self.hidden1(x))
        x = F.relu(self.hidden2(x))
        x = self.out(x)
        return x

out_train = np.zeros((y_train.shape[0],2))
out_test = np.zeros((y_test.shape[0],2))
out_train[y_train==0, :] = np.array([1,0])
out_train[y_train==1, :] = np.array([0,1])
out_test[y_test==0, :] = np.array([1,0])
out_test[y_test==1, :] = np.array([0,1])

net = Net(n_feature = size, n_hidden1 = 200, n_hidden2=10, n_output = 2)
optimizer = torch.optim.SGD(net.parameters(), lr=0.02)
loss_func = torch.nn.CrossEntropyLoss()

x = torch.FloatTensor(train_vec)
y = torch.LongTensor(y_train.astype(float))
from torch.autograd import Variable
x, y = Variable(x), Variable(y)

for t in range(100):
    out = net(x)                 # input x and predict based on x
    loss = loss_func(out, y)     # must be (1. nn output, 2. target), the target label is NOT one-hotted

    optimizer.zero_grad()   # clear gradients for next train
    loss.backward()         # backpropagation, compute gradients
    optimizer.step()

pred = torch.max(out, 1)[1]
pred = pred.data.numpy().squeeze()
train_accuracy = sum(pred == y_train)/y_train.shape[0]

tes = Variable(torch.FloatTensor(test_vec))
test_out = net(tes)
tes_pred = torch.max(test_out, 1)[1]
tes_pred = tes_pred.data.numpy().squeeze()
test_accuracy = sum(tes_pred == y_test)/y_test.shape[0]
