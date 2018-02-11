from torch import nn
import torch

class Loss(nn.Module):
    def __init__(self):
        super(Loss, self).__init__()
        self.ce_loss = nn.CrossEntropyLoss(size_average=False)

    def forward(self, recon_x, x, mu, logvar):
        # BCE = F.mse_loss(recon_x, x, size_average=False)
        x = x * 255
        x.data = x.data.int().long().view(-1)
        recon_x = recon_x.view(-1, 256)

        CE = self.ce_loss(recon_x, x)

        # see Appendix B from VAE paper:    https://arxiv.org/abs/1312.6114
        # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

        return CE + KLD
