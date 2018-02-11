from torch.autograd import Variable
from torch import optim
import torch
from torchvision.utils import save_image
from tqdm import tqdm
import numpy as np

from base_trainer import BaseTrainer


class Trainer(BaseTrainer):
    def __init__(self, model, loss, train_loader, test_loader, args):
        super(Trainer, self).__init__(model, loss, train_loader, test_loader, args)
        self.model = model
        self.args = args
        self.args.start_epoch = 0
        self.optimizer = self.get_optimizer()

    def train(self):
        self.model.train()
        for epoch in range(self.args.start_epoch, self.args.num_epochs):
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
            self.save_checkpoint(epoch)
            if self.args.linear_scheduler:
                new_lr = self.adjust_learning_rate(epoch)
                print('learning rate:', new_lr)

            if epoch % 20 == 0:
                self.test(epoch)

    def test(self, cur_epoch):
        print('testing...')
        self.model.eval()
        test_loss = 0
        for i, (data, _) in enumerate(self.test_loader):
            if self.args.cuda:
                data = data.cuda()
            data = Variable(data, volatile=True)
            recon_batch, mu, logvar = self.model(data)
            test_loss += self.loss(recon_batch, data, mu, logvar).data[0]
            _, indices = recon_batch.max(1)
            indices.data = indices.data.float() / 255
            if i == 0:
                n = min(data.size(0), 8)
                comparison = torch.cat([data[:n],
                                        indices.view(-1, 3, 32, 32)[:n]])
                save_image(comparison.data.cpu(),
                           self.args.exp_name + '/results/reconstruction_' + str(cur_epoch) + '.png', nrow=n)

        test_loss /= len(self.test_loader.dataset)
        print('====> Test set loss: {:.4f}'.format(test_loss))
        self.model.train()

    def test_on_trainings_set(self):
        print('testing...')
        self.model.eval()
        test_loss = 0
        for i, (data, _) in enumerate(self.train_loader):
            if self.args.cuda:
                data = data.cuda()
            data = Variable(data, volatile=True)
            recon_batch, mu, logvar = self.model(data)
            test_loss += self.loss(recon_batch, data, mu, logvar).data[0]
            _, indices = recon_batch.max(1)
            if i % 10 == 0:
                n = min(data.size(0), 8)
                comparison = torch.cat([data[:n],
                                        indices.view(-1, 3, 32, 32)[:n]])
                save_image(comparison.data.cpu(),
                           self.args.exp_name + '/train_results/reconstruction_' + str(i) + '.png', nrow=n)

        test_loss /= len(self.test_loader.dataset)
        print('====> Test set loss: {:.4f}'.format(test_loss))
        self.model.train()

    def get_optimizer(self):
        return optim.Adam(self.model.parameters(), lr=self.args.learning_rate,
                          weight_decay=self.args.weight_decay)
