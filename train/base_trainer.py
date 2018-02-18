import shutil

import torch.optim
from tensorboardX import SummaryWriter


class BaseTrainer:
    def __init__(self, model, loss, train_loader, val_loader, test_loader, args):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.args = args

        # Loss function and Optimizer
        self.loss = loss
        self.optimizer = None

        # Tensorboard Writer
        self.summary_writer = SummaryWriter(log_dir=args.summary_dir)

    def train(self):
        raise NotImplementedError

    def test(self, cur_epoch):
        raise NotImplementedError

    def get_optimizer(self):
        raise NotImplementedError

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
