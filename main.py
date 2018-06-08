import os
import copy
import pickle
from datetime import datetime
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
import utils.DataProcessing as DP
import utils.LSTMClassifier as LSTMC

DATA_DIR = 'data'
TRAIN_DIR = 'train_txt'
TEST_DIR = 'test_txt'
TRAIN_FILE = 'train_txt.txt'
TEST_FILE = 'test_txt.txt'
TRAIN_LABEL = 'train_label.txt'
TEST_LABEL = 'test_label.txt'

# hyperParameters setting
EPOCHS = 2
BATCH_SIZE = 5
LEARNING_RATE = 0.01


def adjust_learning_rate(optimizer, epoch):
    lr = LEARNING_RATE * (0.1 ** (epoch // 10))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return optimizer


if __name__ == "__main__":
    train_file = os.path.join(DATA_DIR, TRAIN_FILE)
    test_file = os.path.join(DATA_DIR, TEST_FILE)
    fp_train = open(train_file, 'r')
    train_filenames = [os.path.join(TRAIN_DIR, f_name.strip()) for f_name in fp_train]
    file_names = train_filenames
    fp_train.close()
    fp_test = open(test_file, 'r')
    test_filenames = [os.path.join(TEST_DIR, f_name.strip()) for f_name in fp_test]
    file_names.extend(test_filenames)
    fp_test.close()

    corpus = DP.Corpus(DATA_DIR, file_names)
    n_label = 8

    # NN parameters
    embedding_dim = 100
    hidden_dim = 50
    sentence_len = 32
    # model
    model = LSTMC.LSTM_Classifier(embedding_dim=embedding_dim, hidden_dim=hidden_dim,
                                  vocab_size=len(corpus.dictionary), label_size=n_label, batch_size=BATCH_SIZE)
    print(model)
    # data processing
    dtrain_set = DP.DatasetPreprocessing(DATA_DIR, TRAIN_DIR, TRAIN_FILE, TRAIN_LABEL, sentence_len, corpus)

    train_loader = DataLoader(dtrain_set,
                              batch_size=BATCH_SIZE,
                              shuffle=True,
                              num_workers=4)

    dtest_set = DP.DatasetPreprocessing(DATA_DIR, TEST_DIR, TEST_FILE, TEST_LABEL, sentence_len, corpus)

    test_loader = DataLoader(dtest_set,
                             batch_size=BATCH_SIZE,
                             shuffle=False,
                             num_workers=4)

    optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE)
    loss_function = nn.CrossEntropyLoss()
    train_loss_ = []
    test_loss_ = []
    train_acc_ = []
    test_acc_ = []
    # training procedure
    for epoch in range(EPOCHS):
        optimizer = adjust_learning_rate(optimizer, epoch)

        # training epoch
        total_acc = 0.0
        total_loss = 0.0
        total = 0.0
        for iter, traindata in enumerate(train_loader):
            train_inputs, train_labels = traindata
            train_labels = torch.squeeze(train_labels)
            train_inputs = Variable(train_inputs)
            model.zero_grad()
            model.batch_size = len(train_labels)
            model.hidden = model.init_hidden()
            output = model(train_inputs.t())

            loss = loss_function(output, Variable(train_labels))
            loss.backward()
            optimizer.step()

            # calc training acc
            _, predicted = torch.max(output.data, 1)

            total_acc += (predicted == train_labels).sum().item()
            total += len(train_labels)
            total_loss += loss.item()

        train_loss_.append(total_loss / total)
        train_acc_.append(total_acc / total)
        # testing epoch
        total_acc = 0.0
        total_loss = 0.0
        total = 0.0
        for iter, testdata in enumerate(test_loader):
            test_inputs, test_labels = testdata
            test_labels = torch.squeeze(test_labels)
            test_inputs = Variable(test_inputs)
            model.batch_size = len(test_labels)
            model.hidden = model.init_hidden()
            output = model(test_inputs.t())

            loss = loss_function(output, Variable(test_labels))

            # calc testing acc
            _, predicted = torch.max(output.data, 1)
            total_acc += (predicted == test_labels).sum().item()
            total += len(test_labels)
            total_loss += loss.item()
        test_loss_.append(total_loss / total)
        test_acc_.append(total_acc / total)

        print('[Epoch: %3d/%3d] Training Loss: %.3f, Testing Loss: %.3f, Training Acc: %.3f, Testing Acc: %.3f'
              % (epoch, EPOCHS, train_loss_[epoch], test_loss_[epoch], train_acc_[epoch], test_acc_[epoch]))

    param = {'lr': LEARNING_RATE, "batch size": BATCH_SIZE, 'embedding dim': embedding_dim, 'hidden dim': hidden_dim,
             'sentence len': sentence_len}

    result = {'train loss': train_loss_, 'test loss': test_loss_, 'train acc': train_acc_, 'test acc': test_acc_,
              'param': param}

    filename = 'log/LSTM_classifier_' + datetime.now().strftime("%d-%h-%m-%s") + '.pkl'
    result['filename'] = filename
    # saving the modal architecture and weights
    # torch.save(model, 'model.pb')
    # saving the modal
    torch.save(model, './model.pth')
    # saving logs
    fp = open(filename, 'wb')
    pickle.dump(result, fp)
    fp.close()
    print('File %s is saved.' % filename)
