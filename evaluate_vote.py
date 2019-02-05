from __future__ import print_function, division

import argparse
from tqdm import tqdm
import os
import PIL.Image as Image
import torch
import torch.nn as nn
import torchvision
from torchvision import models, transforms
import numpy as np

parser = argparse.ArgumentParser(description='RecVis A3 evaluation script')
parser.add_argument('--data', type=str, default='bird_dataset', metavar='D',
					help="folder where data is located. test_images/ need to be found in the folder")
parser.add_argument('--model', type=str, default='RSN152_unfreeze.pth', metavar='M',
					help="the model file to be evaluated. Usually it is of the form model_X.pth")
parser.add_argument('--outfile', type=str, default='kaggle.csv', metavar='D',
					help="name of the output csv file")
args = parser.parse_args()

### the 3 best ResNet152 that main.py has produced
noms_des_modeles = ['RSN152_unfreeze_1.pth',
'RSN152_unfreeze_augmentor.pth',
'RSN152_unfreeze_2.pth']

nb_de_modeles = len(noms_des_modeles)
model = []
for i in range(0,nb_de_modeles):
	model.append(models.resnet152())
	num_ftrs = model[i].fc.in_features
	model[i].fc = nn.Linear(num_ftrs, 20)
	state_dict = torch.load(noms_des_modeles[i])
	model[i].load_state_dict(state_dict)
	model[i].eval()

from data import data_transforms
test_dir = args.data + '/test_images/mistery_category'
def pil_loader(path):
	with open(path, 'rb') as f:
		with Image.open(f) as img:
			return img.convert('RGB')
			
softy = nn.Softmax(dim=1)

output_file = open(args.outfile, "w")
output_file.write("Id,Category\n")
for f in tqdm(os.listdir(test_dir)):
	if 'jpg' in f:
		data = data_transforms(pil_loader(test_dir + '/' + f))
		data = data.view(1, data.size(0), data.size(1), data.size(2))
		
		voting_list = np.zeros(20).astype(int)
		weighted_voting_list = np.zeros(20)
		
		### vote among the models
		for i in range(0,nb_de_modeles):
			output = model[i](data)
			pred_model = np.asarray(output.data.max(1, keepdim=True)[1])[0][0]
			softy_model = np.asarray(softy(output.data))[0]
			
			voting_list[pred_model] = voting_list[pred_model] + 1
			#weighted_voting_list[pred_model] = weighted_voting_list[pred_model] + softy_model[pred_model]
			
		pred = np.argmax(voting_list)
		#pred = np.argmax(weighted_voting_list)
		
		output_file.write("%s,%d\n" % (f[:-4], pred))
output_file.close()

print("Succesfully wrote " + args.outfile + ', you can upload this file to the kaggle competition website')