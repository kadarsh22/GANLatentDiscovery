import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm
import torch.nn.functional as F

class Reshape(nn.Module):
    def __init__(self, *args):
        super(Reshape, self).__init__()
        self.shape = args

    def forward(self, x):
        return x.view(self.shape)

class CRDiscriminator(nn.Module):
    '''Shared Part of Discriminator and Recognition Model'''

    def __init__(self, dim_c_cont):
        super(CRDiscriminator, self).__init__()
        # self.dim_c_disc = dim_c_disc
        self.dim_c_cont = dim_c_cont
        # self.n_c_disc = n_c_disc
        # Shared layers
        self.module = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels=2,
                                    out_channels=32,
                                    kernel_size=4,
                                    stride=2,
                                    padding=1)),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            spectral_norm(nn.Conv2d(in_channels=32,
                                    out_channels=32,
                                    kernel_size=4,
                                    stride=2,
                                    padding=1)),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            spectral_norm(nn.Conv2d(in_channels=32,
                                    out_channels=64,
                                    kernel_size=4,
                                    stride=2,
                                    padding=1)),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            spectral_norm(nn.Conv2d(in_channels=64,
                                    out_channels=64,
                                    kernel_size=4,
                                    stride=2,
                                    padding=1)),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            Reshape(-1, 256),
            spectral_norm(nn.Linear(in_features=256, out_features=128)),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Linear(in_features=128, out_features=2),
        )

    def forward(self, x1, x2):
        x = torch.cat((x1, x2), dim=1)
        out = self.module(x)
        return out

class Classifier(nn.Module):
	def __init__(self, output_dim=2):
		super(Classifier, self).__init__()

		self.latent_dim = 2
		self.nc = 2

		self.cnn1_en = nn.Conv2d(self.nc, 32, 4, 2, 1)
		self.cnn2_en = nn.Conv2d(32, 32, 4, 2, 1)
		self.cnn3_en = nn.Conv2d(32, 32, 4, 2, 1)
		self.cnn4_en = nn.Conv2d(32, 32, 4, 2, 1)
		self.linear1_en = nn.Linear(32 * 4 * 4, 256)
		self.linear2_en = nn.Linear(256, 256)
		self.z_mean = nn.Linear(256, self.latent_dim)
		self.act = nn.ReLU(inplace=True)

	def encoder(self, x):
		x = x.type(torch.cuda.FloatTensor)
		x = x.view(-1, self.nc, 64, 64)
		out = self.act(self.cnn1_en(x))
		out = self.act(self.cnn2_en(out))
		out = self.act(self.cnn3_en(out))
		out = self.act(self.cnn4_en(out)).view(-1, 32 * 4 * 4)
		out = self.act(self.linear1_en(out))
		out = self.act(self.linear2_en(out))
		z_parameters = self.z_mean(out)
		return z_parameters

	def forward(self, x1,x2):
		x = torch.cat((x1, x2), dim=1)
		z_mean = self.encoder(x)
		return z_mean