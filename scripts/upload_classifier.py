import argparse
import csv
import getpass
import tempfile

from webservice.service_client_session import ServiceClientSession

__author__ = 'Andrea Esuli'

MY_LARGE_FIELD = 1024 * 1024
BATCH_SIZE = 1024 * 1024 * 10

if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--protocol', help='host protocol', type=str, default='http')
    parser.add_argument('--host', help='host server address', type=str, default='127.0.0.1')
    parser.add_argument('--port', help='host server port', type=int, default=8080)
    parser.add_argument('--user_auth_path', help='server path of the auth web service', type=str,
                        default='/service/userauth')
    parser.add_argument('--classifier_path', help='server path of the classifier web service', type=str,
                        default='/service/classifiers')
    parser.add_argument('--dataset_path', help='server path of the dataset web service', type=str,
                        default='/service/datasets')
    parser.add_argument('--jobs_path', help='server path of the jobs web service', type=str,
                        default='/service/jobs')
    parser.add_argument('-t', '--type', help='type of classifier to learn', type=str,
                        default='Statistical')
    parser.add_argument('-b', '--batch_size', help='Size of data batch', type=int, default=BATCH_SIZE)
    parser.add_argument('input', help='input csv file', type=str)
    args = parser.parse_args()

    sc = ServiceClientSession(args.protocol, args.host, args.port, args.classifier_path, args.dataset_path,
                              args.jobs_path, args.user_auth_path, '', '')

    if csv.field_size_limit() < MY_LARGE_FIELD:
        csv.field_size_limit(MY_LARGE_FIELD)

    username = input('Username: ').strip()
    password = getpass.getpass('Password: ')
    sc.login(username, password)

    with open(args.input, mode='r', encoding='utf-8') as inputfile:
        reader = csv.reader(inputfile)

        row = next(reader, None)
        count = 0
        batch_count = 0
        while row:
            with tempfile.TemporaryFile(mode='w+', encoding='utf-8')  as outputfile:
                print('Temp file ' + str(count))
                while row:
                    writer = csv.writer(outputfile)
                    writer.writerow(row)
                    count += 1
                    batch_count += 1
                    row = next(reader, None)

                    if outputfile.tell() > args.batch_size:
                        outputfile.seek(0)
                        print('Sending', batch_count, 'rows.')
                        sc.classifier_upload_training_data(outputfile, args.type)
                        batch_count = 0
                        break
                if batch_count > 0:
                    outputfile.seek(0)
                    sc.classifier_upload_training_data(outputfile, args.type)

    print('Sent', count,'rows.')
