import argparse
import csv

__author__ = 'Andrea Esuli'

MY_LARGE_FIELD = 1024 * 1024
BATCH_SIZE = 1024 * 10

if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-b', '--batch_size', help='size of data batch', type=int, default=BATCH_SIZE)
    parser.add_argument('classifier_name', help='name of the classifier to be read from input file', type=str)
    parser.add_argument('input', help='input csv file', type=str)
    parser.add_argument('output', help='output csv file', type=str)
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

    first_batch = True

    with open(args.input, mode='r', encoding='utf-8') as inputfile:
        reader = csv.reader(inputfile)
        next(reader)
        ids = []
        X = []
        y = []
        for row in reader:
            if len(row) < 3:
                continue
            id_ = row[0]
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
                    ids.append(id_)
                    X.append(text)
                    y.append(label)
                    break
            if len(X) >= args.batch_size:
                print('batch:', len(X))
                if first_batch:
                    mode = 'w'
                else:
                    mode = 'a'

                with open(args.output, mode=mode, encoding='utf-8') as outputfile:
                    if first_batch:
                        header_start = '"# {""classifiers"": ['
                        header_end = ']}"\n'
                        header_list = ','.join(
                            ['{""name"": ""' + label + '"", ""labels"": [""yes"", ""no""]}' for label in labels])
                        outputfile.write(header_start+header_list+header_end)
                        first_batch = False
                    writer = csv.writer(outputfile, lineterminator='\n')
                    for id_, text, example_label in zip(ids, X, y):
                        row = [id_, text]
                        for label in labels:
                            if example_label == label:
                                row.append(label + ':yes')
                            else:
                                row.append(label + ':no')
                        writer.writerow(row)

                ids = []
                X = []
                y = []

        if len(X) > 0:
            print('batch:', len(X))
            if first_batch:
                mode = 'w'
            else:
                mode = 'a'

            with open(args.output, mode=mode, encoding='utf-8') as outputfile:
                if first_batch:
                    header_start = '"# {""classifiers"": ['
                    header_end = ']}"\n'
                    header_list = ','.join(
                        ['{""name"": ""' + label + '"", ""labels"": [""yes"", ""no""]}' for label in labels])
                    outputfile.write(header_start + header_list + header_end)
                    first_batch = False
                writer = csv.writer(outputfile, lineterminator='\n')
                for id_, text, example_label in zip(ids, X, y):
                    row = [id_, text]
                    for label in labels:
                        if example_label == label:
                            row.append(label + ':yes')
                        else:
                            row.append(label + ':no')
                    writer.writerow(row)
