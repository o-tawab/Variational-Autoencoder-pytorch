from torch.autograd import Variable
from torch import optim
import torch
from torchvision.utils import save_image
from tqdm import tqdm
import numpy as np

from train.base_trainer import BaseTrainer


class Trainer(BaseTrainer):
    def __init__(self, model, loss, train_loader, test_loader, args):
        super(Trainer, self).__init__(model, loss, train_loader, None, test_loader, args)
        self.model = model
        self.args = args
        self.args.start_epoch = 0
        self.optimizer = self.get_optimizer()

        # Model Loading
        if args.resume:
            self.load_checkpoint(self.args.resume_from)

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
            new_lr = self.adjust_learning_rate(epoch)
            print('learning rate:', new_lr)

            self.summary_writer.add_scalar('training/loss', np.mean(loss_list), epoch)
            self.summary_writer.add_scalar('training/learning_rate', new_lr, epoch)
            self.save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': self.model.state_dict(),
                'optimizer': self.optimizer.state_dict(),
            })
            if epoch % self.args.test_every == 0:
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
            if i == 0:
                n = min(data.size(0), 8)
                comparison = torch.cat([data[:n],
                                        recon_batch.view(-1, 3, 32, 32)[:n]])
                self.summary_writer.add_image('testing_set/image', comparison, cur_epoch)

        test_loss /= len(self.test_loader.dataset)
        print('====> Test set loss: {:.4f}'.format(test_loss))
        self.summary_writer.add_scalar('testing/loss', test_loss, cur_epoch)
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
            if i % 50 == 0:
                n = min(data.size(0), 8)
                comparison = torch.cat([data[:n],
                                        recon_batch.view(-1, 3, 32, 32)[:n]])
                self.summary_writer.add_image('training_set/image', comparison, i)

        test_loss /= len(self.test_loader.dataset)
        print('====> Test on training set loss: {:.4f}'.format(test_loss))
        self.model.train()

    def get_optimizer(self):
        return optim.Adam(self.model.parameters(), lr=self.args.learning_rate,
                          weight_decay=self.args.weight_decay)
