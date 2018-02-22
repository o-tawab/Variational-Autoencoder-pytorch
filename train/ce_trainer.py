from torch.autograd import Variable
from torch import optim
import torch

from tensorboardX import SummaryWriter
import shutil
from tqdm import tqdm
import numpy as np

class Trainer:
    def __init__(self, model, loss, train_loader, test_loader, args):
        self.model = model
        self.args = args
        self.args.start_epoch = 0

        self.train_loader = train_loader
        self.test_loader = test_loader

        # Loss function and Optimizer
        self.loss = loss
        self.optimizer = self.get_optimizer()

        # Tensorboard Writer
        self.summary_writer = SummaryWriter(log_dir=args.summary_dir)
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
            _, indices = recon_batch.max(1)
            indices.data = indices.data.float() / 255
            if i == 0:
                n = min(data.size(0), 8)
                comparison = torch.cat([data[:n],
                                        indices.view(-1, 3, 32, 32)[:n]])
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
            _, indices = recon_batch.max(1)
            indices.data = indices.data.float() / 255
            if i % 50 == 0:
                n = min(data.size(0), 8)
                comparison = torch.cat([data[:n],
                                        indices.view(-1, 3, 32, 32)[:n]])
                self.summary_writer.add_image('training_set/image', comparison, i)

        test_loss /= len(self.test_loader.dataset)
        print('====> Test on training set loss: {:.4f}'.format(test_loss))
        self.model.train()

    def get_optimizer(self):
        return optim.Adam(self.model.parameters(), lr=self.args.learning_rate,
                          weight_decay=self.args.weight_decay)

    def adjust_learning_rate(self, epoch):
        """Sets the learning rate to the initial LR multiplied by 0.98 every epoch"""
        learning_rate = self.args.learning_rate * (self.args.learning_rate_decay ** epoch)
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = learning_rate
        return learning_rate

    def save_checkpoint(self, state, is_best=False, filename='checkpoint.pth.tar'):
        '''
        a function to save checkpoint of the training
        :param state: {'epoch': cur_epoch + 1, 'state_dict': self.model.state_dict(),
                            'optimizer': self.optimizer.state_dict()}
        :param is_best: boolean to save the checkpoint aside if it has the best score so far
        :param filename: the name of the saved file
        '''
        torch.save(state, self.args.checkpoint_dir + filename)
        if is_best:
            shutil.copyfile(self.args.checkpoint_dir + filename,
                            self.args.checkpoint_dir + 'model_best.pth.tar')

    def load_checkpoint(self, filename):
        filename = self.args.checkpoint_dir + filename
        try:
            print("Loading checkpoint '{}'".format(filename))
            checkpoint = torch.load(filename)
            self.args.start_epoch = checkpoint['epoch']
            self.model.load_state_dict(checkpoint['state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            print("Checkpoint loaded successfully from '{}' at (epoch {})\n"
                  .format(self.args.checkpoint_dir, checkpoint['epoch']))
        except:
            print("No checkpoint exists from '{}'. Skipping...\n".format(self.args.checkpoint_dir))
