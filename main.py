from __future__ import print_function
import torch.utils.data

from utils import *
from model import *
from trainer import *


def main():
    args = parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    model = VAE()
    trainer = Trainer(model, args)

    trainer.train()


if __name__ == '__main__':
    main()
