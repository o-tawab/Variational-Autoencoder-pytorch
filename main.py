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
    config_args = parse_args()

    # Create the experiment directories
    config_args.summary_dir, config_args.checkpoint_dir, config_args.test_results_dir, config_args.train_results_dir = create_experiment_dirs(
        config_args.experiment_dir)

    model = VAE()
    loss = Loss()

    if config_args.cuda:
        model.cuda()
        loss.cuda()
        cudnn.enabled = True
        cudnn.benchmark = True

    print("Loading Data...")
    data = CIFAR10DataLoader(config_args)
    print("Data loaded successfully\n")

    trainer = Train(model, loss, data.trainloader, data.testloader, config_args)

    if config_args.to_train:
        try:
            print("Training...")
            trainer.train()
            print("Training Finished\n")
        except KeyboardInterrupt:
            pass

    if config_args.to_test:
        print("Testing on training data...")
        trainer.test_on_trainings_set()
        print("Testing Finished\n")


if __name__ == "__main__":
    main()
