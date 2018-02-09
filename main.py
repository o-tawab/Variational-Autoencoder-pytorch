from __future__ import print_function
import torch.utils.data
import torch.nn.init as init

from utils import *
from model import *
from trainer import *
from weight_initializer import Initializer


def main():
    args = parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)
    if args.cuda:
        print('using CUDA...')
        torch.cuda.manual_seed(args.seed)

    model = VAE()
    Initializer.initialize(model, initialization=init.xavier_uniform, gain=init.calculate_gain('relu'))
    trainer = Trainer(model, args)

    trainer.train()


if __name__ == '__main__':
    main()
