import os
import torch
from torch.utils.data.dataset import Dataset
import numpy as np


class Dictionary(object):
    def __init__(self):
        self.word_to_ix = {}
        self.id_to_word = []

    def add_word(self, word):
        if word not in self.word_to_ix:
            # filtering out the unique word list
            self.id_to_word.append(word)
            self.word_to_ix[word] = len(self.id_to_word) - 1

    def __len__(self):
        return len(self.id_to_word)


class Corpus(object):
    def __init__(self, DATA_DIR, filenames):
        self.dictionary = Dictionary()
        self.data = self.tokenize(DATA_DIR, filenames)

    def tokenize(self, DATA_DIR, filenames):
        for filename in filenames:
            path = os.path.join(DATA_DIR, filename)
            with open(path, 'r') as f:
                tokens = 0
                for line in f:
                    words = line.split() + ['<eos>']
                    tokens += len(words)
                    for word in words:
                        self.dictionary.add_word(word)

            with open(path, 'r') as f:
                ids = torch.LongTensor(tokens)
                token = 0
                for line in f:
                    words = line.split() + ['<eos>']
                    for word in words:
                        ids[token] = self.dictionary.word_to_ix[word]
                        token += 1
        return ids


class DatasetPreprocessing(Dataset):
    def __init__(self, data_path, txt_path, txt_filename, label_filename, sen_len, corpus):
        self.txt_path = os.path.join(data_path, txt_path)
        # reading txt file from file
        txt_filepath = os.path.join(data_path, txt_filename)
        fp = open(txt_filepath, 'r')
        self.txt_filename = [x.strip() for x in fp]
        fp.close()
        # reading labels from file
        label_filepath = os.path.join(data_path, label_filename)
        fp_label = open(label_filepath, 'r')
        labels = [int(x.strip()) for x in fp_label]
        fp_label.close()
        self.label = labels
        self.corpus = corpus
        self.sen_len = sen_len

    def __getitem__(self, index):
        filename = os.path.join(self.txt_path, self.txt_filename[index])
        fp = open(filename, 'r')
        txt = torch.LongTensor(np.zeros(self.sen_len, dtype=np.int64))
        count = 0
        clip = False
        for words in fp:
            for word in words.split():
                if word.strip() in self.corpus.dictionary.word_to_ix:
                    if count > self.sen_len - 1:
                        clip = True
                        break
                    txt[count] = self.corpus.dictionary.word_to_ix[word.strip()]
                    count += 1
            if clip: break
        label = torch.LongTensor([self.label[index]])
        return txt, label

    def __len__(self):
        return len(self.txt_filename)
