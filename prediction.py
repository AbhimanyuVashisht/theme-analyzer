import os
import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
import numpy as np

def load_word_corpus():
    word_to_ix = np.load('word_corpus.npy').item()
    return word_to_ix

def DataProcessing(sentence, sen_len):
    word_to_ix = load_word_corpus()
    txt = torch.LongTensor(np.zeros(sen_len, dtype=np.int64))
    count = 0
    clip = False
    for word in sentence.split():
        if word.strip() in word_to_ix:
            if count > sen_len - 1:
                clip = True
                break
            txt[count] = word_to_ix[word.strip()]
            count += 1
    return txt


if __name__ == "__main__":
    model = torch.load('./model.pth')
    model.eval()
    print(model.batch_size)
    print(model.hidden)
    # model = model.load_state_dict(torch.load('model.pt'))
    # print(model.eval())
    input = "what the fuck was that"
    text_input = DataProcessing(input, 32)
    # text_loader = DataLoader(text_input,
    #                         batch_size=1,
    #                         shuffle=False)
    # text_to_predict = text_loader
    # model.batch_size = 1
    model.hidden = model.init_hidden()
    # print(model.batch_size)
    # print(model.hidden)
    # print(text_input[0])
    text_input = torch.Tensor.numpy(text_input)
    text_input = torch.LongTensor([text_input,text_input,text_input,text_input,text_input])
    # text_input = Variable(text_input)
    # print(text_input)
    # print(text_input.t())
    output = model(text_input.t())

    _, predicted = torch.max(output.data, 1)
    print(predicted)
