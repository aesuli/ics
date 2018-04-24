import argparse
import csv

from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import normalize

from classifier.classifier import get_classifier_model

__author__ = 'Andrea Esuli'

MY_LARGE_FIELD = 1024 * 1024
BATCH_SIZE = 1024 * 10

if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-t', '--type', help='type of classifier to learn', type=str,
                        default='Statistical')
    parser.add_argument('-k', '--folds', help='number of folds', type=int, default=10)
    parser.add_argument('-b', '--batch_size', help='Size of data batch', type=int, default=BATCH_SIZE)
    parser.add_argument('classifier_name', help='name of the classifier to be read from input file', type=str)
    parser.add_argument('input', help='input csv file', type=str)
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

    all_y = list()
    all_yhat = list()

    for fold in range(args.folds):
        print('fold', fold + 1, '/', args.folds)

        model = get_classifier_model(args.type, classifier_name, labels)
        labels = model.labels()
        with open(args.input, mode='r', encoding='utf-8') as inputfile:
            reader = csv.reader(inputfile)
            next(reader)
            X_train = []
            y_train = []
            X_test = []
            y_test = []

            for count, row in enumerate(reader):
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
                        if count % args.folds == fold:
                            X_train.append(text)
                            y_train.append(label)
                        else:
                            X_test.append(text)
                            y_test.append(label)
                        break
                if len(X_train) >= args.batch_size:
                    print('batch:', len(X_train))
                    all_y.extend(y_test)
                    model.partial_fit(X_train, y_train)
                    all_yhat.extend(model.predict(X_test))
                    X = []
                    y = []

            if len(X_train) > 0:
                print('batch:', len(X_train))
                all_y.extend(y_test)
                model.partial_fit(X_train, y_train)
                all_yhat.extend(model.predict(X_test))

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
