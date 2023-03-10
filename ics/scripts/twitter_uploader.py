import getpass
import sys
from pathlib import Path

from configargparse import ArgParser
from twiget_cli import TwiGetCLIBase

from ics.client import ClientSession


def create_uploader(client: ClientSession):
    def uploader(data):
        id = data['data']['id']
        text = data['data']['text']
        for rule in data['matching_rules']:
            try:
                client.dataset_add_document(rule['tag'], id, text)
            except Exception as e:
                print(e)

    return uploader


def main():
    parser = ArgParser()
    parser.add_argument('-c', '--config', help='configuration file', is_config_file=True)
    parser.add_argument('-s', '--save', help='saves configuration to a file', is_write_out_config_file_arg=True)
    parser.add_argument('--protocol', help='host protocol (http)', type=str, default='http')
    parser.add_argument('--host', help='host server address', type=str, default='127.0.0.1')
    parser.add_argument('--port', help='host server port', type=int, default=8080)
    parser.add_argument('--dataset_path', help='server path of the dataset web service', type=str,
                        default='/service/datasets/')
    parser.add_argument('--user_auth_path', help='server path of the user auth web service', type=str,
                        default='/service/userauth/')

    default_bearer_filename = '.twiget.conf'

    parser.add_argument('-b', '--bearer_filename', type=str, default=Path.home() / default_bearer_filename)

    args = parser.parse_args(sys.argv[1:])

    try:
        with open(args.bearer_filename, mode='rt', encoding='utf-8') as input_file:
            bearer = input_file.readline().strip()
    except:
        print(f'Cannot load the bearer token from {args.bearer_filename}')
        sys.exit(-1)

    client = ClientSession(args.protocol, args.host, args.port, dataset_path=args.dataset_path,
                           user_auth_path=args.user_auth_path)

    print(f'Logging into {args.protocol}://{args.host}:{args.port}{args.user_auth_path}')
    username = input('Username: ').strip()
    try:
        client.login(username, getpass.getpass('Password: '))
    except Exception as e:
        print(f'Login failed: {e.args[0]}')
        sys.exit(-1)

    twigetcli = TwiGetCLIBase(bearer)

    twigetcli._twiget.add_callback('uploader', create_uploader(client))

    twigetcli.cmdloop()


if __name__ == '__main__':
    main()
