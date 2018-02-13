from __future__ import print_function
import torch.utils.data
import torch.nn.init as init

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
    loss = Loss()

    args.cuda = not args.no_cuda and torch.cuda.is_available()
    if args.cuda:
        model.cuda()
        loss.cuda()
        cudnn.enabled = True
        cudnn.benchmark = True

    print("Loading Data...")
    data = CIFAR10DataLoader(args)
    print("Data loaded successfully\n")

    trainer = Train(model, loss, data.trainloader, data.testloader, args)

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
