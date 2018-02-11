from torchvision import datasets, transforms
from torch.utils.data import DataLoader


class CIFAR10DataLoader:
    def __init__(self, args):
        if args.dataset == 'CIFAR10':
            # Data Loading
            kwargs = {'num_workers': args.dataloader_workers, 'pin_memory': args.pin_memory} if args.cuda else {}

            transform_train = transforms.Compose([
                transforms.ToTensor()
                # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])

            transform_test = transforms.Compose([
                transforms.ToTensor()
                # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])

            train_set = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
            self.train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=args.shuffle,
                                           **kwargs)

            test_set = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
            self.test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False,
                                          **kwargs)

            self.classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

        else:
            raise ValueError('The dataset should be CIFAR10')
