import sys
from pathlib import Path

from configargparse import ArgParser
from twiget_cli import TwiGetCLIBase

from ics.client import ClientSession


def create_uploader(client):

    def uploader(data):
        print(client)

    return uploader


if __name__ == '__main__':

    protocol = 'http'
    host = 'label.esuli.it'
    port = 80
    dataset_path = 'service/datasets'
    user_auth_path = 'service/userauth'

    default_bearer_filename = '.twiget.conf'

    parser = ArgParser()
    parser.add_argument('-c', '--config', help='configuration file', default='twitter_uploader.conf',
                        is_config_file=True)
    parser.add_argument('-b', '--bearer_filename', type=str, default=Path.home() / default_bearer_filename)
    args = parser.parse_args(sys.argv[1:])

    try:
        with open(args.bearer_filename, mode='rt', encoding='utf-8') as input_file:
            bearer = input_file.readline().strip()
    except:
        print(f'Cannot load the bearer token from {args.bearer_filename}')
        exit(-1)

    client = ClientSession(protocol, host, port, dataset_path=dataset_path, user_auth_path=user_auth_path)

    twigetcli = TwiGetCLIBase(bearer)

    twigetcli._twiget.add_callback('uploader', create_uploader(client))

    twigetcli.cmdloop()
