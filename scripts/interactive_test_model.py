import argparse
import csv
import pickle

from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import normalize

__author__ = 'Andrea Esuli'

MY_LARGE_FIELD = 1024 * 1024
BATCH_SIZE = 1024 * 10

if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-b', '--batch_size', help='Size of data batch', type=int, default=BATCH_SIZE)
    parser.add_argument('model', help='model file', type=str)
    args = parser.parse_args()

    with open(args.model, 'rb') as inputfile:
        model = pickle.load(inputfile)

    classifier_name = model.name()
    labels = model.labels()

    print('classifer:', classifier_name)
    print('labels:', labels)

    while True:
        line = input('>').strip()
        if line=='x':
            break
        X = [line]
        yhat = model.predict(X)
        print(yhat[0])
