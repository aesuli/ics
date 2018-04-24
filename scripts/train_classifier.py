import argparse
import csv
import pickle

from classifier.classifier import get_classifier_model

__author__ = 'Andrea Esuli'

MY_LARGE_FIELD = 1024 * 1024
BATCH_SIZE = 1024 * 10

if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-t', '--type', help='type of classifier to learn', type=str,
                        default='Statistical')
    parser.add_argument('-b', '--batch_size', help='Size of data batch', type=int, default=BATCH_SIZE)
    parser.add_argument('classifier_name', help='name of the classifier to be read from input file', type=str)
    parser.add_argument('input', help='input csv file', type=str)
    parser.add_argument('output', help='output model file', type=str)
    args = parser.parse_args()

    classifier_name = args.classifier_name

    if csv.field_size_limit() < MY_LARGE_FIELD:
        csv.field_size_limit(MY_LARGE_FIELD)

    labels = set()

    with open(args.input, mode='r', encoding='utf-8') as inputfile:
        reader = csv.reader(inputfile)
        next(reader)
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
                    labels.add(label)
                    break

    labels = list(labels)

    print('classifer:', classifier_name)
    print('labels:', labels)

    model = get_classifier_model(args.type, classifier_name, labels)

    with open(args.input, mode='r', encoding='utf-8') as inputfile:
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
                print('batch:',len(X))
                model.partial_fit(X, y)
                X = []
                y = []
        if len(X) > 0:
            print('batch:', len(X))
            model.partial_fit(X, y)

    with open(args.output, 'wb') as outputfile:
        pickle.dump(model, outputfile)
