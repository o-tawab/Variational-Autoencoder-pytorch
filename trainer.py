from torch.autograd import Variable
from torch import optim
import torch
import shutil
import torch.backends.cudnn as cudnn
from torch.nn import functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np


class Trainer:
    def __init__(self, model, args):
        self.model = model
        self.args = args
        self.args.start_epoch = 0
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.args.learning_rate,
                                    weight_decay=self.args.weight_decay)

        self.loss = self.loss_function

        # If NVIDIA CUDA is available.
        if self.args.cuda:
            self.model.cuda()
            # To select the best algorithms for training.
            cudnn.enabled = True
            cudnn.benchmark = True

        self.load_checkpoint()

        if args.dataset == 'CIFAR10':
            # Data Loading
            kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}

            transform_train = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])

            transform_test = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])

            train_set = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
            self.train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                                                            **kwargs)

            test_set = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
            self.test_loader = torch.utils.data.DataLoader(test_set, batch_size=args.batch_size, shuffle=False,
                                                           **kwargs)

            self.classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    def train(self):
        self.model.train()
        for epoch in range(self.args.start_epoch, self.args.num_epoch):
            loss_list = []
            print("epoch {}...".format(epoch))
            for batch_idx, (data, _) in enumerate(tqdm(self.train_loader)):
                if self.args.cuda:
                    data = data.cuda()
                data = Variable(data)
                self.optimizer.zero_grad()
                recon_batch, mu, logvar = self.model(data)
                loss = self.loss(recon_batch, data, mu, logvar)
                loss.backward()
                self.optimizer.step()
                loss_list.append(loss.data[0])

            print("epoch {}: - loss: {}".format(epoch, np.mean(loss_list)))
            if self.args.linear_scheduler:
                self.linear_scheduler(epoch)

    def test(self):
        self.model.eval()
        test_loss = 0
        for i, (data, _) in enumerate(self.test_loader):
            if self.args.cuda:
                data = data.cuda()
            data = Variable(data, volatile=True)
            recon_batch, mu, logvar = self.model(data)
            test_loss += self.loss(recon_batch, data, mu, logvar).data[0]
            if i == 0:
                n = min(data.size(0), 8)
                comparison = torch.cat([data[:n],
                                        recon_batch.view(self.args.batch_size, 1, 28, 28)[:n]])
                # save_image(comparison.data.cpu(),
                #            'results/reconstruction_' + str(epoch) + '.png', nrow=n)

        test_loss /= len(self.test_loader.dataset)
        print('====> Test set loss: {:.4f}'.format(test_loss))

    def validate(self):
        pass

    def loss_function(self, recon_x, x, mu, logvar):
        BCE = F.binary_cross_entropy(recon_x, x.view(-1, 784), size_average=False)

        # see Appendix B from VAE paper:
        # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
        # https://arxiv.org/abs/1312.6114
        # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

        return BCE + KLD

    def save_checkpoint(self, cur_epoch, is_best=False):
        """Saves checkpoint to disk"""
        filename = self.args.checkpoint_path
        state = {'epoch': cur_epoch + 1, 'state_dict': self.model.state_dict(),
                 'optimizer': self.optimizer.state_dict()}
        torch.save(state, filename)
        if is_best:
            shutil.copyfile(filename, 'model_best.pth.tar')

    def load_checkpoint(self):
        """Loads checkpoint from disk"""
        if self.args.resume:
            try:
                print("Loading checkpoint '{}'...".format(self.args.checkpoint_path))
                checkpoint = torch.load(self.args.checkpoint_path)
                self.args.start_epoch = checkpoint['epoch']
                self.model.load_state_dict(checkpoint['state_dict'])
                self.optimizer.load_state_dict(checkpoint['optimizer'])
                print("Checkpoint Loaded. '{}' at epoch {}\n\n".format(self.args.checkpoint_path, checkpoint['epoch']))
            except:
                print("No checkpoints exist. Aborting.\n\n")

    def linear_scheduler(self, epoch):
        """Decay learning rate linearly"""
        lr = self.args.learning_rate * (1 - epoch / self.args.num_epochs)

        # Stopping criterion for decaying.
        if lr > min(self.args.learning_rate / 100.0, 1e-6):
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
