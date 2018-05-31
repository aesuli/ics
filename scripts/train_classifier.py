import argparse
import csv
import pickle
from collections import defaultdict

from classifier.classifier import get_classifier_model

__author__ = 'Andrea Esuli'

MY_LARGE_FIELD = 1024 * 1024
BATCH_SIZE = 1024 * 10

if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-t', '--type', help='type of classifier to learn', type=str,
                        default='Statistical')
    parser.add_argument('-b', '--batch_size', help='Size of data batch', type=int, default=BATCH_SIZE)
    parser.add_argument('-c', '--classifier_name',
                        help='name of the classifier to be read from input file (default:all)', type=str, default=None)
    parser.add_argument('-o', '--output_model_prefix', help='output model file name prefix', type=str, default='')
    parser.add_argument('input', help='input csv file', type=str)
    args = parser.parse_args()

    if csv.field_size_limit() < MY_LARGE_FIELD:
        csv.field_size_limit(MY_LARGE_FIELD)

    labels = defaultdict(set)

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
                if example_classifier_name is not None and (
                                example_classifier_name == args.classifier_name or args.classifier_name is None) and \
                                label is not None and len(label) > 0:
                    labels[example_classifier_name].add(label)

    for classifier_name in labels:
        print('classifer:', classifier_name)
        print('labels:', sorted(list(labels[example_classifier_name])))

        model = get_classifier_model(args.type, classifier_name, list(labels[classifier_name]))

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
                    print('batch:', len(X))
                    model.partial_fit(X, y)
                    X = []
                    y = []
            if len(X) > 0:
                print('batch:', len(X))
                model.partial_fit(X, y)

        with open(args.output_model_prefix + classifier_name + '.model', 'wb') as outputfile:
            pickle.dump(model, outputfile)
