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
    parser.add_argument('classifier_name', help='name of the classifier to be read from input file', type=str)
    parser.add_argument('test', help='test csv file', type=str)
    parser.add_argument('model', help='model file', type=str)
    args = parser.parse_args()

    with open(args.model, 'rb') as inputfile:
        model = pickle.load(inputfile)

    classifier_name = model.name()
    labels = model.labels()

    if csv.field_size_limit() < MY_LARGE_FIELD:
        csv.field_size_limit(MY_LARGE_FIELD)

    print('classifer:', classifier_name)
    print('labels:', labels)

    all_y = list()
    all_yhat = list()

    with open(args.test, mode='r', encoding='utf-8') as inputfile:
        reader = csv.reader(inputfile)
        next(reader)
        X = []
        y = []
        for row in reader:
            if len(row) < 3:
                continue
            text = row[1]
            classifiers_labels = row[2:]
            for classifier_label in classifiers_labels:
                try:
                    example_classifier_name, label = classifier_label.split(':')
                except:
                    continue
                example_classifier_name = example_classifier_name.strip()
                label = label.strip()
                if example_classifier_name is not None and example_classifier_name == classifier_name and \
                                label is not None and len(label) > 0:
                    X.append(text)
                    y.append(label)
                    break
            if len(X) >= args.batch_size:
                print('batch:', len(X))
                all_y.extend(y)
                all_yhat.extend(model.predict(X))
                X = []
                y = []

        if len(X) > 0:
            print('batch:', len(X))
            all_y.extend(y)
            all_yhat.extend(model.predict(X))
    print('Classification report:')
    print(classification_report(all_y, all_yhat, target_names=labels))
    print('Confusion matrix:')
    cm = confusion_matrix(all_y, all_yhat, labels=labels)
    tab_len = max([len(label) for label in labels]) + 3
    print('\t'.join([f'{label:{tab_len}}' for label in ['LABELS'] + labels]))
    for label, row in zip(labels, cm):
        print('\t'.join([f'{label:{tab_len}}' for label in [label] + [f'{value:{tab_len}}' for value in row]]))
    print('Row normalized confusion matrix:')
    ncm = normalize(cm, 'l1', 1)
    print('\t'.join([f'{label:{tab_len}}' for label in ['LABELS'] + labels]))
    for label, row in zip(labels, ncm):
        print('\t'.join([f'{label:{tab_len}}' for label in [label] + [f'{value:{tab_len}.3f}' for value in row]]))
