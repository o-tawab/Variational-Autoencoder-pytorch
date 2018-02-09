import argparse
from bunch import Bunch
import json


def parse_args():
    """
    Parse the arguments of the program
    :return: (config_args)
    :type: tuple
    """
    # Create a parser
    parser = argparse.ArgumentParser(description='VAE MNIST Example')
    parser.add_argument('--batch-size', type=int, default=32, metavar='N',
                        help='input batch size for training (default: 128)')
    parser.add_argument('--epochs', type=int, default=200, metavar='N',
                        help='number of epochs to train (default: 200)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='enables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--config', default='./config.json', type=str, help='Configuration file')
    # Parse the arguments
    args = parser.parse_args()

    # parse the configurations from the config json file provided
    if args.config:
        with open(args.config, 'r') as config_file:
            config_args_dict = json.load(config_file)
        # convert the dictionary to a namespace using bunch lib
        config_args = Bunch(config_args_dict)

        return config_args

    else:
        return args