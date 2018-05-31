import re

import torch
import hashlib
import torch.nn.functional as F
from torch.autograd import Variable

from classifier.classifier import Classifier

__author__ = 'Andrea Esuli'


class LSTMTextClassificationNet(torch.nn.Module):
    def __init__(self, vocabulary_size, embedding_size, classes, class_lstm_hidden_size, class_lstm_layers,
                 class_lin_layers_sizes, dropout):
        super().__init__()

        self.class_lstm_layers = class_lstm_layers
        self.class_lstm_hidden_size = class_lstm_hidden_size

        self.dropout = dropout

        self.embedding = torch.nn.Embedding(vocabulary_size, embedding_size)
        self.class_lstm = torch.nn.LSTM(embedding_size, class_lstm_hidden_size, class_lstm_layers, dropout=self.dropout, bidirectional=False)
        prev_size = class_lstm_hidden_size
        self.class_lins = torch.nn.ModuleList()
        for lin_size in class_lin_layers_sizes:
            self.class_lins.append(torch.nn.Linear(prev_size, lin_size))
            prev_size = lin_size
        self.class_output = torch.nn.Linear(prev_size, classes)

    def init_class_hidden(self, set_size):
        var_hidden = torch.autograd.Variable(torch.zeros(self.class_lstm_layers, set_size, self.class_lstm_hidden_size))
        var_cell = torch.autograd.Variable(torch.zeros(self.class_lstm_layers, set_size, self.class_lstm_hidden_size))
        if next(self.class_lstm.parameters()).is_cuda:
            return (var_hidden.cuda(), var_cell.cuda())
        else:
            return (var_hidden, var_cell)

    def forward(self, x):
        embedded = self.embedding(x)
        rnn_output, rnn_hidden = self.class_lstm(embedded, self.init_class_hidden(x.size()[1]))
        abstracted = F.dropout(rnn_hidden[0][-1], self.dropout, training=self.training)
        for linear in self.class_lins:
            abstracted = F.dropout(F.relu(linear(abstracted)), training=self.training)
        output = self.class_output(abstracted)
        class_output = F.log_softmax(output, dim=1)
        return class_output


class LSTMClassifier(Classifier):
    def __init__(self, name, classes, n_features=(2 ** 16), embedding_size=100, class_lstm_hidden_size=100,
                 class_lstm_layers=1, class_lin_layers_sizes=[256,128], dropout=0.1):
        lr = 0.01
        self.max_steps = 100
        self.no_improvement_reset = 10
        self._name = name
        self.n_features = n_features
        self.min_accuracy = 0.9
        self._classes = list(set(classes))
        if class_lin_layers_sizes is None:
            class_lin_layers_sizes = [class_lstm_hidden_size]
        self._tokenizer = re.compile('\W+')
        self._net = LSTMTextClassificationNet(n_features, embedding_size, len(classes),
                                              class_lstm_hidden_size=class_lstm_hidden_size,
                                              class_lstm_layers=class_lstm_layers,
                                              class_lin_layers_sizes=class_lin_layers_sizes, dropout=dropout)
        #self._optimizer = torch.optim.SGD(self._net.parameters(), lr=lr)
        self._optimizer = torch.optim.Adam(self._net.parameters(), lr=lr)#, weight_decay=1e-5)
        self._loss = torch.nn.NLLLoss()

    def partial_fit(self, X, y):
        X = self.index(X)
        y = Variable(torch.LongTensor([self._classes.index(label) for label in y]))
        step = 0
        last_loss = float('inf')
        no_improvement = self.no_improvement_reset
        while step < self.max_steps:
            __batch_size=250
            __steps = X.shape[1]//__batch_size
            __epochs = 1
            print('steps',__steps)
            for __e in range(__epochs):
                for __i in range(__steps):
                    __X = X[:,__i*__batch_size:(__i+1)*__batch_size]
                    __y = y[__i * __batch_size:(__i + 1) * __batch_size]
                    self._net.train()
                    self._optimizer.zero_grad()
                    yhat = self._net(__X)
                    loss = self._loss(yhat, __y)
                    print('i',__i,'/',__steps,'loss',loss.data[0])
                    loss.backward()
                    self._optimizer.step()

            # self._net.train()
            # self._optimizer.zero_grad()
            # yhat = self._net(X)
            # loss = self._loss(yhat, y)
            # print('loss',loss.data[0])
            # loss.backward()
            # self._optimizer.step()
            self._net.eval()
            yhat = [scoring.index(max(scoring)) for scoring in self._net(X).exp().data.tolist()]
            print(yhat)
            accuracy = len([1 for pred, real in zip(yhat, y) if pred == real.data[0]]) / len(y)
            print('accuracy', accuracy)
            if accuracy > self.min_accuracy:
                print('accuracy', accuracy)
                break
            else:
                if loss.data[0] < last_loss:
                    last_loss = loss.data[0]
                    no_improvement = self.no_improvement_reset
                else:
                    no_improvement -= 1
                    if no_improvement == 0:
                        print('break', step)
                        break
            step += 1
        print('step', step)
        return loss.data.tolist()[0]

    def predict(self, X):
        scores = self.decision_function(X)
        return [self._classes[scoring.index(max(scoring))] for scoring in scores]

    def _hash(self,token):
        return int(hashlib.md5(token.encode('utf8')).hexdigest(),16)

    def index(self, X):
        X = [re.sub(r'https?://(?:[-\w.]|(?:%[\da-fA-F]{2}))+(/[\w\d]+)', r'URL', x) for x in X]
        X = [self._tokenizer.split(x) for x in X]
        X = [[self._hash(t) % self.n_features for t in x] for x in X]
        lengths = [len(x) for x in X]
        maxlen = max(lengths)
        X = [[0] * (maxlen - lengths[i]) + x for i, x in enumerate(X)]
        X = Variable(torch.LongTensor(X).transpose(0, 1))
        return X

    def decision_function(self, X):
        X = self.index(X)
        self._net.eval()
        return self._net.forward(X).exp().data.tolist()

    def name(self):
        return self._name

    def rename(self, new_name):
        self._name = new_name

    def rename_label(self, label_name, new_name):
        self._classes = [new_name if name == label_name else name for name in self._classes]

    def labels(self):
        return list(self._classes)


if __name__ == '__main__':
    texts = ['one two three', 'one four', 'two one three', 'two one three', 'two three'] * 100
    labels = ['yes', 'no', 'no', 'no', 'dd'] * 100
    clf = LSTMClassifier('test', ['yes', 'no', 'dd', 'ww', 'dwws', 'ss'])
    for _ in range(1000):
        print(clf.predict(texts))
        print(clf.decision_function(texts))
        print(clf.partial_fit(texts, labels))
