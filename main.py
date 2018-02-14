from __future__ import print_function
import torch.utils.data
import torch.nn.init as init
import torch.backends.cudnn as cudnn

from utils import *
from model import VAE
from loss import Loss
from trainer import Trainer
from cifar10_data_loader import CIFAR10DataLoader
from weight_initializer import Initializer


def main():
    # Parse the JSON arguments
    args = parse_args()

    # Create the experiment directories
    args.summary_dir, args.checkpoint_dir, args.test_results_dir, args.train_results_dir = create_experiment_dirs(
        args.experiment_dir)

    model = VAE()

    # to apply xavier_uniform:
    Initializer.initialize(model=model, initialization=init.xavier_uniform, gain=init.calculate_gain('relu'))

    loss = Loss()

    args.cuda = args.cuda and torch.cuda.is_available()
    if args.cuda:
        model.cuda()
        loss.cuda()
        cudnn.enabled = True
        cudnn.benchmark = True

    print("Loading Data...")
    data = CIFAR10DataLoader(args)
    print("Data loaded successfully\n")

    trainer = Trainer(model, loss, data.train_loader, data.test_loader, args)

    if args.to_train:
        try:
            print("Training...")
            trainer.train()
            print("Training Finished\n")
        except KeyboardInterrupt:
            pass

    if args.to_test:
        print("Testing on training data...")
        trainer.test_on_trainings_set()
        print("Testing Finished\n")


if __name__ == "__main__":
    main()
