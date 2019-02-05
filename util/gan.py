from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable

from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
from torch.optim import lr_scheduler
import os
import copy

plt.ion()

########################################################################

batchSize = 64
imageSize = 64

data_dir = 'bird_dataset'
TRAIN = 'train_images'

data_transforms = {
	TRAIN: transforms.Compose([
		transforms.Resize(80),
		transforms.RandomResizedCrop(64),
		transforms.RandomHorizontalFlip(),
		transforms.ToTensor(),
		transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
	])
}

dataset = {
	x: datasets.ImageFolder(
		os.path.join(data_dir, x), 
		transform=data_transforms[x]
	)
	for x in [TRAIN]
}

dataloader = {
	x: torch.utils.data.DataLoader(
		dataset[x], batch_size=8,
		shuffle=True, num_workers=4
	)
	for x in [TRAIN]
}

def weights_init(m):
	classname = m.__class__.__name__
	if classname.find('Conv') != -1:
		m.weight.data.normal_(0.0, 0.02)
	elif classname.find('BatchNorm') != -1:
		m.weight.data.normal_(1.0, 0.02)
		m.bias.data.fill_(0)

class G(nn.Module):
	def __init__(self):
		super(G, self).__init__()
		self.main = nn.Sequential(
			nn.ConvTranspose2d(100, 512, 4, 1, 0, bias = False),
			nn.BatchNorm2d(512),
			nn.ReLU(True),
			nn.ConvTranspose2d(512, 256, 4, 2, 1, bias = False),
			nn.BatchNorm2d(256),
			nn.ReLU(True),
			nn.ConvTranspose2d(256, 128, 4, 2, 1, bias = False),
			nn.BatchNorm2d(128),
			nn.ReLU(True),
			nn.ConvTranspose2d(128, 64, 4, 2, 1, bias = False),
			nn.BatchNorm2d(64),
			nn.ReLU(True),
			nn.ConvTranspose2d(64, 3, 4, 2, 1, bias = False),
			nn.Tanh()
		)
	def forward(self, input):
		output = self.main(input)
		return output

netG = G()
netG.apply(weights_init)

class D(nn.Module):
	def __init__(self):
		super(D, self).__init__()
		self.main = nn.Sequential(
			nn.Conv2d(3, 64, 4, 2, 1, bias = False),
			nn.LeakyReLU(0.2, inplace = True),
			nn.Conv2d(64, 128, 4, 2, 1, bias = False),
			nn.BatchNorm2d(128),
			nn.LeakyReLU(0.2, inplace = True),
			nn.Conv2d(128, 256, 4, 2, 1, bias = False),
			nn.BatchNorm2d(256),
			nn.LeakyReLU(0.2, inplace = True),
			nn.Conv2d(256, 512, 4, 2, 1, bias = False),
			nn.BatchNorm2d(512),
			nn.LeakyReLU(0.2, inplace = True),
			nn.Conv2d(512, 1, 4, 1, 0, bias = False),
			nn.Sigmoid()
		)
	def forward(self, input):
		output = self.main(input)
		return output.view(-1)

netD = D()
netD.apply(weights_init)
criterion = nn.BCELoss()
optimizerD = optim.Adam(netD.parameters(), lr = 0.0002, betas = (0.5, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr = 0.0002, betas = (0.5, 0.999))

for epoch in range(25):
	for i, data in enumerate(dataloader[TRAIN]):
		
		netD.zero_grad()
		real, label = data
		input = Variable(real)
		target = Variable(torch.ones(input.size()[0]))
		output = netD(input)
		errD_real = criterion(output, target)
		
		noise = Variable(torch.randn(input.size()[0], 100, 1, 1))
		fake = netG(noise)
		target = Variable(torch.zeros(input.size()[0]))
		output = netD(fake.detach())
		errD_fake = criterion(output, target)
		
		errD = errD_real + errD_fake
		errD.backward()
		optimizerD.step()

		netG.zero_grad()
		target = Variable(torch.ones(input.size()[0]))
		output = netD(fake)
		errG = criterion(output, target)
		errG.backward()
		optimizerG.step()
		
		print('[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f' % (epoch, 25, i, len(dataloader), errD.data[0], errG.data[0]))
		if i % 100 == 0:
			vutils.save_image(real, '%s/real_samples.png' % "./", normalize = True)
			fake = netG(noise)
			vutils.save_image(fake.data, '%s/fake_samples_epoch_%03d.png' % ("./", epoch), normalize = True)